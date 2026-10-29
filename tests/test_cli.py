"""Tests for idrdesign.cli."""
from pathlib import Path

import pytest


def test_cli_help_runs():
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-m", "idrdesign.cli", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "target-fasta" in r.stdout


def test_cli_parse_args_defaults():
    from idrdesign.cli import parse_args
    args = parse_args([
        "--target-fasta", "in.fa",
        "--target-name", "WT",
        "--output-dir", "out",
    ])
    assert args.target_fasta == "in.fa"
    assert args.target_name == "WT"
    assert args.output_dir == "out"
    assert args.n_designs == 10
    assert args.target_similarity == 0.95
    assert args.composition_mode == "none"
    assert args.pilot is False


def test_cli_parse_args_pilot_sets_n_designs_to_one():
    from idrdesign.cli import parse_args
    args = parse_args([
        "--target-fasta", "in.fa",
        "--target-name", "WT",
        "--output-dir", "out",
        "--pilot",
    ])
    assert args.pilot is True
    assert args.n_designs == 1


def test_cli_parse_args_from_yaml(tmp_path: Path):
    from idrdesign.cli import parse_args
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "target_fasta: in.fa\n"
        "target_name: WT\n"
        "output_dir: out\n"
        "n_designs: 5\n"
        "target_similarity: 0.85\n"
    )
    args = parse_args(["--config", str(cfg)])
    assert args.target_fasta == "in.fa"
    assert args.n_designs == 5
    assert args.target_similarity == 0.85
