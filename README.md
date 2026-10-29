# idr-design

Design intrinsically disordered protein sequences whose predicted
inter-residue interaction map matches a target.

Paste a disordered region into the page, choose how much of the original
amino-acid composition to preserve, and get back a new sequence optimized to
reproduce the input's interaction map — with the maps rendered side by side.

## Method

Every run starts from a length-matched GS repeat, so nothing is carried over
from the input sequence except the constraints you set explicitly. Candidate
sequences are proposed by [GOOSE](https://github.com/idptools/goose) and scored
by RMSE against the target's inter-residue interaction map, computed by
[FINCHES](https://github.com/idptools/finches) from the Mpipi and CALVADOS
forcefields. A run stops when it stops improving, hits its iteration cap, or
hits its wall-clock cap.

With both forcefields selected the objective is their weighted sum. CALVADOS
maps run roughly twice the magnitude of Mpipi ones, so each is weighted by the
reciprocal of its target map's mean absolute value; otherwise CALVADOS would
dominate. That weighting is a project choice, not something the method
prescribes.

Scoring is deterministic: the maps are analytical, so a given sequence always
scores identically.

## Layout

```
src/idrdesign/
  intermap_design.py     design driver for the FINCHES objectives (active)
  intermap_objective.py  the Mpipi and CALVADOS GOOSE objectives
  intermap_figure.py     renders the map comparison PNG
  optimize.py            STARLING encoder design driver (not currently used)
  encode.py, ensemble.py STARLING encoder and diffusion wrappers
tests/                   test suite
worker/
  modal_app.py           CPU worker for the FINCHES path — the HTTP API
  modal_app_starling.py  GPU worker for the STARLING path, kept deployable
web/                     static page: design.html, design.css, design.js
docs/DEPLOY.md           how to get it online
```

## Architecture

```
browser  ──POST──▶  submit endpoint  ──spawn──▶  CPU container
   │                (returns job id)              design_intermap()
   └──GET result──▶  result endpoint  ◀─────────   render_intermap_figure()
       (polls every 3s)
```

A run takes minutes, which is past the request limit of any serverless
platform, so submission and collection are separate calls and the browser
polls.

**No GPU.** The interaction maps are analytical numpy — there is no model to put
on one. torch is still installed because sparrow pulls in metapredict, but the
CPU wheel suffices.

The front end is three static files served from the website alongside
`index.html`. The worker runs on [Modal](https://modal.com) and is reached over
CORS. No Python runs on the web host: the dependency set is far over the bundle
limit of a typical serverless function.

## Controls

| On the page | Parameter | Effect |
|---|---|---|
| What to keep → Nothing | `composition_mode="none"` | free search from a GS repeat |
| What to keep → Bulk properties | `composition_mode="properties"` | soft FCR/NCPR/hydrophobicity anchors |
| What to keep → Exact composition | `composition_mode="exact"` | shuffle only; amino-acid counts preserved |
| What to match | `forcefields` | Mpipi, CALVADOS, or both |
| Window | `window_size` | 17 or 31 residues; 31 costs ~2.2× per step |
| Effort | `max_iterations` | 20–300; also stops early on stagnation |
| Show the interaction maps | `render_maps` | renders input, design and difference per forcefield |
| Seed | `seed` | see below |

Input length is capped at 380 residues and floored at 20, and must be at least
twice the window. FASTA headers, residue numbering and whitespace are stripped
on paste.

## Reproducibility

Runs are reproducible: the same sequence, settings and seed give the same design.

That takes some care. `goose.optimize.SequenceOptimizer` accepts no seed, and
its candidate generator `cython_shuffle_sequence` draws from the C library's
`rand()` — which neither `random.seed()` nor `numpy.random.seed()` reaches.
Seeding only those two leaves the search unseeded while still appearing to
accept a seed. `idrdesign.seeding.seed_everything` seeds libc through
`ctypes.srand()` alongside Python, numpy and torch.

`IntermapDesignResult.reproducible` records whether libc was actually reachable.
If it is ever False, the search ran unseeded and the result should not be
treated as reproducible.

## The STARLING path

`optimize.py`, `encode.py` and `ensemble.py` design against STARLING encoder
embeddings instead, and `worker/modal_app_starling.py` deploys that version on a
GPU. It is kept intact and is not used by the current page.

**A STARLING bug this repo works around:** its models load in training mode.
`ModelManager.load_models` never calls `.eval()`, and `torch.inference_mode()`
only disables autograd — it does not change module mode. Left alone, 511 of 712
submodules including the sequence encoder stay in training mode, which keeps a
20% ionic-strength dropout live in `SequenceEncoder.forward`: roughly one call
in five returned an embedding displaced by about 37% of the full 0→500 mM range,
as though no ionic strength had been given, and the encoder was not
deterministic on a fixed sequence.

`idrdesign.encode.ensure_eval_mode` fixes this, and is called before every
encode and every ensemble generation. After it, 0 of 712 submodules are in
training mode and repeated calls are bitwise identical. The fix belongs
upstream in STARLING; until it lands there, anything calling STARLING directly
rather than through this package is still affected.

## Limits

The worker is reachable by anyone with the URL, so it ships with caps:
`max_wall_seconds=1500`, `max_iterations≤300`, one design per request, and at
most 4 containers at once. See `docs/DEPLOY.md` for restricting access.

## Command line

The STARLING pipeline is usable directly:

```bash
pip install -e .
idrdesign --target-fasta seqs.fasta --target-name MYSEQ --output-dir out/
```

The FINCHES path has no CLI yet; call `idrdesign.intermap_design.design_intermap`
directly. goose, sparrow, finches and torch must be installed separately — see
the pinned commits in `worker/modal_app.py`.

## License

MIT — see [LICENSE](LICENSE). STARLING, GOOSE, sparrow and FINCHES are separate
projects with their own licenses.
