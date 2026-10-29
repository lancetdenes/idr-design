# idrdesign-web

Design intrinsically disordered protein sequences whose predicted
conformational ensemble matches a target.

Paste a disordered region into the page, choose how much of the original
amino-acid composition to preserve, and get back new sequences optimized to
reproduce the input's conformational ensemble.

## Method

Every run starts from a length-matched GS repeat, so nothing is carried over
from the input sequence except the constraints you set explicitly. Candidate
sequences are proposed by [GOOSE](https://github.com/idptools/goose) and scored
by cosine similarity to the target's
[STARLING](https://github.com/idptools/starling) encoder embedding. A run stops
when it reaches the similarity threshold, stagnates, or hits its iteration or
wall-clock cap.

Optionally, full conformational ensembles are then generated for the input and
every design, and radius of gyration, end-to-end distance and asphericity are
reported side by side.

## Layout

```
src/idrdesign/        the design pipeline
tests/                its test suite
worker/modal_app.py   GPU worker — the HTTP API
web/                  static page: design.html, design.css, design.js
docs/DEPLOY.md        how to get it online
```

## Architecture

```
browser  ──POST──▶  submit endpoint  ──spawn──▶  GPU container
   │                (returns job id)              design_idr()
   └──GET result──▶  result endpoint  ◀─────────   generate_ensembles()
       (polls every 2s)
```

A design run takes tens of seconds and the ensemble pass adds more, which is
past the request limit of any serverless platform. Submission and collection
are therefore separate calls, and the browser polls for the result.

The front end is three static files served from the website alongside
`index.html`. The GPU worker runs on [Modal](https://modal.com) and is reached
over CORS. No Python runs on the web host: torch alone exceeds the bundle limit
of a typical serverless function by several times.

## Controls

| On the page | Parameter | Effect |
|---|---|---|
| What to keep → Nothing | `composition_mode="none"` | free search from a GS repeat |
| What to keep → Chemical classes | `composition_mode="properties"` | per-class residue counts held fixed |
| What to keep → Exact composition | `composition_mode="exact"` | shuffle only; amino-acid counts preserved |
| Ensemble match | `target_similarity` | 0.80–0.995 acceptance threshold |
| Ionic strength | `ionic_strength` | 0–500 mM; 150 is physiological |
| Sequences returned | `n_designs` | 1–5 independent optimizations |
| Measure the ensembles | `compute_metrics` | off skips the ensemble pass entirely |
| Conformers | `n_conformers` | 50–400 per sequence |
| Seed | `seed` | sequence + settings + seed reproduce a run |

Input length is capped at 380 residues, STARLING's `MAX_SEQUENCE_LENGTH`, and
floored at 10. FASTA headers, residue numbering and whitespace are stripped on
paste.

## Limits

The worker is reachable by anyone with the URL, so it ships with limits well
below what the pipeline supports: `max_wall_seconds=420`,
`max_iterations=3000`, at most 5 designs per request, and at most 4 GPU
containers at once. Raising them is reasonable behind authentication — see
`docs/DEPLOY.md`.

## Command line

The pipeline is usable directly, without the web layer:

```bash
pip install -e .
idrdesign --target-fasta seqs.fasta --target-name MYSEQ --output-dir out/
```

This writes designs, ensembles, a metrics table and an HTML report. STARLING,
GOOSE, sparrow and torch must already be installed; see the pinned versions in
`worker/modal_app.py`.

## License

MIT — see [LICENSE](LICENSE). STARLING, GOOSE, sparrow and finches are separate
projects with their own licenses.
