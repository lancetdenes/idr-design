"""Modal GPU worker exposing idrdesign as an async HTTP API.

Two endpoints, both public:

    POST /design   -> {"job_id": "..."}        submit a run, returns immediately
    GET  /result   -> {"status": ..., ...}     poll until status != "running"

A design run is tens of seconds to a few minutes, which is far past any
serverless request timeout, so submission and collection are separate calls.
`submit` spawns the GPU function and hands back Modal's FunctionCall id; the
browser polls `result` with it.

Deploy:  modal deploy worker/modal_app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "idrdesign"

# Pinned to a known-good combination. goose and sparrow have no usable PyPI
# release, so they are pinned by commit rather than version.
GOOSE_SHA = "4298c2f95483311c55129ade1865fdd5d337a582"
SPARROW_SHA = "9b49f77f797c42c1de61e6002a05fac6ffad0a2c"

REPO_ROOT = Path(__file__).resolve().parent.parent

# STARLING downloads model weights on first use. Without a persistent cache
# every cold start re-downloads them, which dominates latency.
weights_cache = modal.Volume.from_name("idrdesign-weights", create_if_missing=True)
CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    # CUDA build of torch.
    .pip_install(
        "torch==2.11.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "numpy==2.2.6",
        "cython==3.2.4",
        "mdtraj==1.10.3",
        "soursop==0.2.6",
        "pandas",
        "pyyaml",
        "jinja2",
        "tqdm",
        "idptools-starling==2.0.0a3",
        f"sparrow @ git+https://github.com/idptools/sparrow.git@{SPARROW_SHA}",
        f"goose @ git+https://github.com/idptools/goose.git@{GOOSE_SHA}",
    )
    .env(
        {
            # Point every weight/model cache at the persistent volume.
            "TORCH_HOME": f"{CACHE_DIR}/torch",
            "HF_HOME": f"{CACHE_DIR}/hf",
            "XDG_CACHE_HOME": f"{CACHE_DIR}/xdg",
            "PYTHONPATH": "/root/src",
        }
    )
    .add_local_dir(REPO_ROOT / "src", remote_path="/root/src")
)

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

# starling.configs.MAX_SEQUENCE_LENGTH, mirrored in idrdesign.ensemble.
MAX_SEQUENCE_LENGTH = 380
MIN_SEQUENCE_LENGTH = 10
AA = set("ACDEFGHIKLMNPQRSTVWY")

VALID_COMPOSITION_MODES = ("none", "exact", "properties")

# Web-facing caps. These exist to bound cost and latency for an anonymous
# public endpoint, and are deliberately tighter than the CLI defaults.
LIMITS = {
    "n_designs": (1, 5),
    "n_conformers": (50, 400),
    "target_similarity": (0.80, 0.995),
    "ionic_strength": (0.0, 500.0),
}
MAX_WALL_SECONDS = 420.0
MAX_ITERATIONS = 3000


class BadRequest(Exception):
    """Client error; surfaced to the browser with its message intact."""


def clean_sequence(raw: str) -> str:
    """Strip FASTA headers, whitespace and numbering; uppercase; validate.

    Accepts a bare sequence, a pasted FASTA record, or sequence text with
    residue numbers and spaces in it (as copied out of a viewer).
    """
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
            f"sequence is {len(seq)} residues; STARLING's maximum is "
            f"{MAX_SEQUENCE_LENGTH}. Trim it or split it into domains."
        )
    return seq


def _num(payload: dict, key: str, cast, default):
    if key not in payload or payload[key] is None:
        return default
    try:
        val = cast(payload[key])
    except (TypeError, ValueError):
        raise BadRequest(f"{key} must be a number")
    if key in LIMITS:
        lo, hi = LIMITS[key]
        if not (lo <= val <= hi):
            raise BadRequest(f"{key} must be between {lo} and {hi}")
    return val


def parse_payload(payload: dict) -> dict:
    """Validate and normalize the browser's JSON into design_idr kwargs."""
    if not isinstance(payload, dict):
        raise BadRequest("expected a JSON object")

    mode = payload.get("composition_mode", "none")
    if mode not in VALID_COMPOSITION_MODES:
        raise BadRequest(
            f"composition_mode must be one of {VALID_COMPOSITION_MODES}, got {mode!r}"
        )

    seed = payload.get("seed")
    if seed in (None, ""):
        seed = 0
    else:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise BadRequest("seed must be an integer")

    return {
        "sequence": clean_sequence(payload.get("sequence", "")),
        "composition_mode": mode,
        "target_similarity": _num(payload, "target_similarity", float, 0.95),
        "ionic_strength": _num(payload, "ionic_strength", float, 150.0),
        "n_designs": _num(payload, "n_designs", int, 3),
        "n_conformers": _num(payload, "n_conformers", int, 100),
        "compute_metrics": bool(payload.get("compute_metrics", True)),
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# The GPU job
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    timeout=900,
    volumes={CACHE_DIR: weights_cache},
    max_containers=4,
)
def run_design(req: dict) -> dict:
    """Run the design loop, then optionally the ensemble/metrics pass."""
    import numpy as np

    from idrdesign.optimize import design_idr

    designs = design_idr(
        target_seq=req["sequence"],
        n_designs=req["n_designs"],
        target_similarity=req["target_similarity"],
        composition_mode=req["composition_mode"],
        max_iterations=MAX_ITERATIONS,
        max_wall_seconds=MAX_WALL_SECONDS,
        seed=req["seed"],
        ionic_strength=req["ionic_strength"],
    )

    out = {
        "status": "done",
        "target": {"sequence": req["sequence"], "length": len(req["sequence"])},
        "params": {k: v for k, v in req.items() if k != "sequence"},
        "designs": [
            {
                "sequence": d.sequence,
                "cosine_similarity": round(float(d.final_cosine_similarity), 4),
                "iterations": int(d.iterations),
                "wall_seconds": round(float(d.wall_seconds), 1),
                "convergence_status": d.convergence_status,
                "seed": int(d.seed),
                "composition_mode": d.composition_mode,
            }
            for d in designs
        ],
        "metrics": None,
    }

    if not req["compute_metrics"]:
        return out

    from idrdesign.ensemble import extract_metrics, generate_ensembles

    records = {"target": req["sequence"]}
    for i, d in enumerate(designs):
        records[f"design_{i:02d}"] = d.sequence

    ensembles = generate_ensembles(
        records,
        conformations=req["n_conformers"],
        ionic_strength=req["ionic_strength"],
        device="cuda",
        verbose=False,
    )

    # Summary scalars only. Distance and contact maps are (L, L) float arrays;
    # for a 380-residue sequence that is ~580 kB of JSON per ensemble, so they
    # are deliberately not serialized here.
    summary = {}
    for name, ens in ensembles.items():
        m = extract_metrics(ens)
        summary[name] = {
            "rg_mean": round(float(np.mean(m["rg"])), 2),
            "rg_std": round(float(np.std(m["rg"])), 2),
            "end_to_end_mean": round(float(np.mean(m["end_to_end"])), 2),
            "end_to_end_std": round(float(np.std(m["end_to_end"])), 2),
            "asphericity_mean": round(float(np.mean(m["asphericity"])), 4),
        }

    out["metrics"] = summary
    return out


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------

def _cors(origin: str | None = None) -> dict:
    allowed = os.environ.get("ALLOWED_ORIGIN", "*")
    return {
        "Access-Control-Allow-Origin": allowed,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


@app.function(image=image)
@modal.fastapi_endpoint(method="POST", docs=False)
def submit(payload: dict):
    """Validate, spawn the GPU job, hand back a job id."""
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
        # timeout=0 polls without blocking the request.
        return JSONResponse(call.get(timeout=0), headers=_cors())
    except TimeoutError:
        return JSONResponse({"status": "running"}, headers=_cors())
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
            status_code=500,
            headers=_cors(),
        )
