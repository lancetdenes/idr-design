"""Command-line interface for idrdesign."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import yaml

from idrdesign.config import Config, VALID_COMPOSITION_MODES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="idrdesign",
        description="Design IDR sequences targeting a STARLING-ensemble match.",
    )
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file. CLI flags override values from this file.")
    p.add_argument("--target-fasta", type=str, default=None)
    p.add_argument("--target-name", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--n-designs", type=int, default=None)
    p.add_argument("--target-similarity", type=float, default=None)
    p.add_argument("--composition-mode", choices=VALID_COMPOSITION_MODES, default=None)
    p.add_argument("--n-conformers", type=int, default=None)
    p.add_argument("--ionic-strength", type=float, default=None)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--max-wall-seconds", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--pilot", action="store_true",
                   help="Single-design verbose pilot run (implies --n-designs 1).")

    args = p.parse_args(argv)

    # Layer YAML defaults under CLI flags.
    yaml_data: dict = {}
    if args.config:
        with open(args.config) as fh:
            yaml_data = yaml.safe_load(fh) or {}

    defaults = Config(
        target_fasta=yaml_data.get("target_fasta", args.target_fasta or ""),
        target_name=yaml_data.get("target_name", args.target_name or ""),
        output_dir=yaml_data.get("output_dir", args.output_dir or ""),
    )

    def _pick(name, default):
        cli = getattr(args, name.replace("-", "_"))
        if cli is not None:
            return cli
        if name in yaml_data:
            return yaml_data[name]
        return default

    args.target_fasta = _pick("target_fasta", defaults.target_fasta)
    args.target_name = _pick("target_name", defaults.target_name)
    args.output_dir = _pick("output_dir", defaults.output_dir)
    args.n_designs = _pick("n_designs", defaults.n_designs)
    args.target_similarity = _pick("target_similarity", defaults.target_similarity)
    args.composition_mode = _pick("composition_mode", defaults.composition_mode)
    args.n_conformers = _pick("n_conformers", defaults.n_conformers)
    args.ionic_strength = _pick("ionic_strength", defaults.ionic_strength)
    args.max_iterations = _pick("max_iterations", defaults.max_iterations)
    args.max_wall_seconds = _pick("max_wall_seconds", defaults.max_wall_seconds)
    args.seed = _pick("seed", defaults.seed)
    args.device = _pick("device", defaults.device)

    if args.pilot:
        args.n_designs = 1

    required = ("target_fasta", "target_name", "output_dir")
    missing = [r for r in required if not getattr(args, r)]
    if missing:
        p.error(f"missing required arguments: {', '.join(missing)}")

    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


def run(args: argparse.Namespace) -> int:
    """Full design pipeline: load target, optimize, generate ensembles, write report."""
    from idrdesign.io import read_fasta, select_record, write_fasta
    from idrdesign.optimize import design_idr
    from idrdesign.ensemble import (
        generate_ensembles,
        save_ensembles,
        extract_metrics,
        validate_sequence_length,
    )
    from idrdesign.report import render_report, write_metrics_csv
    from idrdesign.config import Config, save_config, env_fingerprint, sequence_sha256

    # 1. Load target
    records = read_fasta(args.target_fasta)
    target_seq = select_record(records, args.target_name)
    validate_sequence_length(target_seq)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ensembles").mkdir(exist_ok=True)

    # 2. Save config early (reproducibility before the long compute starts)
    cfg = Config(
        target_fasta=args.target_fasta,
        target_name=args.target_name,
        output_dir=args.output_dir,
        n_designs=args.n_designs,
        target_similarity=args.target_similarity,
        composition_mode=args.composition_mode,
        n_conformers=args.n_conformers,
        ionic_strength=args.ionic_strength,
        max_iterations=args.max_iterations,
        max_wall_seconds=args.max_wall_seconds,
        seed=args.seed,
        device=args.device,
    )
    cfg.env = env_fingerprint()
    cfg.target_seq_sha256 = sequence_sha256(target_seq)
    save_config(cfg, output_dir / "config.yaml")

    # 3. Design
    print(f"[idrdesign] designing {args.n_designs} sequences against {args.target_name} "
          f"(len={len(target_seq)}) ...", flush=True)
    designs = design_idr(
        target_seq=target_seq,
        n_designs=args.n_designs,
        target_similarity=args.target_similarity,
        composition_mode=args.composition_mode,
        max_iterations=args.max_iterations,
        max_wall_seconds=args.max_wall_seconds,
        seed=args.seed,
        ionic_strength=args.ionic_strength,
    )

    # 4. Write designs FASTA
    design_records = {"target": target_seq}
    for i, d in enumerate(designs):
        design_records[f"design_{i:02d}"] = d.sequence
    write_fasta(output_dir / "designs.fasta", design_records)

    # 5. Ensembles
    print(f"[idrdesign] generating ensembles ({args.n_conformers} conformers each) ...", flush=True)
    ensembles = generate_ensembles(
        design_records,
        conformations=args.n_conformers,
        ionic_strength=args.ionic_strength,
        device=args.device,
        verbose=False,
    )
    save_ensembles(ensembles, output_dir / "ensembles")

    # 6. Metrics
    metrics = {name: extract_metrics(ens) for name, ens in ensembles.items()}
    write_metrics_csv(
        path=output_dir / "metrics.csv",
        target_seq=target_seq,
        designs=designs,
        metrics=metrics,
    )

    # 7. Report
    trajectories: dict = {}  # Real per-iter traces aren't captured in DesignResult;
                             # we populate this when we later extend GOOSE logging.
                             # For now, show final point only.
    for i, d in enumerate(designs):
        trajectories[f"design_{i:02d}"] = [(d.iterations, d.final_cosine_similarity)]

    render_report(
        path=output_dir / "report.html",
        target_name=args.target_name,
        target_seq=target_seq,
        designs=designs,
        metrics=metrics,
        trajectories=trajectories,
        config_summary={
            "n_designs": args.n_designs,
            "target_similarity": args.target_similarity,
            "composition_mode": args.composition_mode,
            "n_conformers": args.n_conformers,
            "max_iterations": args.max_iterations,
            "max_wall_seconds": args.max_wall_seconds,
            "seed": args.seed,
        },
    )

    print(f"[idrdesign] done — outputs in {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
