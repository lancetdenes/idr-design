"""Modal CPU worker exposing the FINCHES intermap design path as an HTTP API.

    POST /submit   -> {"job_id": "..."}        start a run, returns immediately
    GET  /result   -> {"status": ..., ...}     poll until status != "running"

One design per request. A run is minutes, not seconds, so submission and
collection are separate calls and the browser polls.

No GPU: the FINCHES intermaps are analytical numpy, so there is no model to put
on one. torch still gets installed because sparrow pulls in metapredict, but the
CPU wheel is enough and is an order of magnitude smaller than the CUDA build.

The STARLING encoder path is preserved in `modal_app_starling.py` and can be
deployed instead; see the README.

Deploy:  modal deploy worker/modal_app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "idrdesign"

# Pinned to a known-good combination. goose, sparrow and finches have no usable
# PyPI release, so they are pinned by commit rather than version.
GOOSE_SHA = "4298c2f95483311c55129ade1865fdd5d337a582"
SPARROW_SHA = "9b49f77f797c42c1de61e6002a05fac6ffad0a2c"
FINCHES_SHA = "5b52ba4023e9faf45e4e172e7681c05f86895b14"

REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    # CPU-only torch. sparrow -> metapredict needs it; nothing here uses a GPU.
    .pip_install(
        "torch==2.11.0",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "numpy==2.2.6",
        "cython==3.2.4",
        "scipy",
        "pandas",
        "matplotlib",
        "pyyaml",
        "tqdm",
        f"sparrow @ git+https://github.com/idptools/sparrow.git@{SPARROW_SHA}",
        f"goose @ git+https://github.com/idptools/goose.git@{GOOSE_SHA}",
        f"finches @ git+https://github.com/idptools/finches.git@{FINCHES_SHA}",
    )
    .env({"PYTHONPATH": "/root/src", "MPLBACKEND": "Agg"})
    .add_local_dir(REPO_ROOT / "src", remote_path="/root/src")
)

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

MAX_SEQUENCE_LENGTH = 380
MIN_SEQUENCE_LENGTH = 20   # below ~2x the window the map is barely defined
AA = set("ACDEFGHIKLMNPQRSTVWY")

VALID_COMPOSITION_MODES = ("none", "exact", "properties")
VALID_FORCEFIELDS = ("mpipi", "calvados")
VALID_WINDOW_SIZES = (17, 31)

# Web-facing caps, set by wall time rather than taste: at window 17 with both
# forcefields a run costs roughly 2 s per iteration on a few cores.
LIMITS = {"max_iterations": (20, 300)}
MAX_WALL_SECONDS = 1500.0


class BadRequest(Exception):
    """Client error; surfaced to the browser with its message intact."""


def clean_sequence(raw: str) -> str:
    """Strip FASTA headers, whitespace and numbering; uppercase; validate."""
    if not isinstance(raw, str):
        raise BadRequest("sequence must be a string")

    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith(">")]
    seq = "".join(lines)
    seq = "".join(ch for ch in seq if not ch.isspace() and not ch.isdigit())
    seq = seq.replace("-", "").replace("*", "").upper()

    if not seq:
        raise BadRequest("no sequence found in the pasted text")

    bad = sorted(set(seq) - AA)
    if bad:
        raise BadRequest(
            f"unrecognized residue(s): {', '.join(bad)}. "
            "Only the 20 standard amino acids are supported."
        )
    if len(seq) < MIN_SEQUENCE_LENGTH:
        raise BadRequest(
            f"sequence is {len(seq)} residues; minimum is {MIN_SEQUENCE_LENGTH}"
        )
    if len(seq) > MAX_SEQUENCE_LENGTH:
        raise BadRequest(
            f"sequence is {len(seq)} residues; the limit is {MAX_SEQUENCE_LENGTH}. "
            "Trim it or split it into domains."
        )
    return seq


def parse_payload(payload: dict) -> dict:
    """Validate and normalize the browser's JSON into design_intermap kwargs."""
    if not isinstance(payload, dict):
        raise BadRequest("expected a JSON object")

    seq = clean_sequence(payload.get("sequence", ""))

    mode = payload.get("composition_mode", "none")
    if mode not in VALID_COMPOSITION_MODES:
        raise BadRequest(
            f"composition_mode must be one of {VALID_COMPOSITION_MODES}, got {mode!r}"
        )

    ffs = payload.get("forcefields", list(VALID_FORCEFIELDS))
    if isinstance(ffs, str):
        ffs = [ffs]
    if not isinstance(ffs, (list, tuple)) or not ffs:
        raise BadRequest("select at least one forcefield")
    bad = [f for f in ffs if f not in VALID_FORCEFIELDS]
    if bad:
        raise BadRequest(f"unknown forcefield(s): {', '.join(map(str, bad))}")
    # Preserve a stable order regardless of what the browser sent.
    ffs = [f for f in VALID_FORCEFIELDS if f in ffs]

    try:
        window_size = int(payload.get("window_size", 17))
    except (TypeError, ValueError):
        raise BadRequest("window_size must be an integer")
    if window_size not in VALID_WINDOW_SIZES:
        raise BadRequest(f"window_size must be one of {VALID_WINDOW_SIZES}")
    if len(seq) < window_size * 2:
        raise BadRequest(
            f"sequence is {len(seq)} residues, too short for window {window_size}; "
            f"need at least {window_size * 2}"
        )

    try:
        iters = int(payload.get("max_iterations", 150))
    except (TypeError, ValueError):
        raise BadRequest("max_iterations must be an integer")
    lo, hi = LIMITS["max_iterations"]
    if not (lo <= iters <= hi):
        raise BadRequest(f"max_iterations must be between {lo} and {hi}")

    seed = payload.get("seed")
    if seed in (None, ""):
        seed = 0
    else:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise BadRequest("seed must be an integer")

    return {
        "sequence": seq,
        "composition_mode": mode,
        "forcefields": ffs,
        "window_size": window_size,
        "max_iterations": iters,
        "render_maps": bool(payload.get("render_maps", True)),
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------

@app.function(
    cpu=8.0,
    memory=8192,
    timeout=1800,
    max_containers=4,
)
def run_design(req: dict) -> dict:
    """Design one sequence against the intermap objective(s), then render maps."""
    from idrdesign.intermap_design import design_intermap, target_maps, pearson, rmse

    result = design_intermap(
        target_seq=req["sequence"],
        forcefields=req["forcefields"],
        window_size=req["window_size"],
        composition_mode=req["composition_mode"],
        max_iterations=req["max_iterations"],
        max_wall_seconds=MAX_WALL_SECONDS,
        seed=req["seed"],
    )

    out = {
        "status": "done",
        "target": {"sequence": req["sequence"], "length": len(req["sequence"])},
        "params": {k: v for k, v in req.items() if k != "sequence"},
        "design": {
            "sequence": result.sequence,
            "starting_sequence": result.starting_sequence,
            "iterations": result.iterations,
            "wall_seconds": round(result.wall_seconds, 1),
            "convergence_status": result.convergence_status,
            "composition_mode": result.composition_mode,
            "window_size": result.window_size,
        },
        "scores": {
            ff: {
                "rmse": round(result.rmse[ff], 4),
                "rmse_start": round(result.rmse_start[ff], 4),
                "pearson": round(result.pearson[ff], 4),
            }
            for ff in result.forcefields
        },
        "maps": None,
    }

    if not req["render_maps"]:
        return out

    from idrdesign.intermap_figure import render_intermap_figure

    tmaps = target_maps(req["sequence"], result.forcefields, result.window_size)
    dmaps = target_maps(result.sequence, result.forcefields, result.window_size)
    out["maps"] = {
        ff: render_intermap_figure(
            tmaps[ff], dmaps[ff],
            forcefield=ff, window_size=result.window_size,
        )
        for ff in result.forcefields
    }
    return out


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------

def _cors() -> dict:
    return {
        "Access-Control-Allow-Origin": os.environ.get("ALLOWED_ORIGIN", "*"),
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


@app.function(image=image)
@modal.fastapi_endpoint(method="POST", docs=False)
def submit(payload: dict):
    """Validate, spawn the job, hand back a job id."""
    from fastapi.responses import JSONResponse

    try:
        req = parse_payload(payload)
    except BadRequest as exc:
        return JSONResponse(
            {"status": "error", "error": str(exc)}, status_code=400, headers=_cors()
        )

    call = run_design.spawn(req)
    return JSONResponse(
        {"status": "running", "job_id": call.object_id},
        status_code=202,
        headers=_cors(),
    )


@app.function(image=image)
@modal.fastapi_endpoint(method="GET", docs=False)
def result(job_id: str):
    """Poll a job. Returns status "running", "done" or "error"."""
    from fastapi.responses import JSONResponse

    try:
        call = modal.FunctionCall.from_id(job_id)
    except Exception:
        return JSONResponse(
            {"status": "error", "error": "unknown job id"},
            status_code=404,
            headers=_cors(),
        )

    try:
        return JSONResponse(call.get(timeout=0), headers=_cors())
    except TimeoutError:
        return JSONResponse({"status": "running"}, headers=_cors())
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
            status_code=500,
            headers=_cors(),
        )
