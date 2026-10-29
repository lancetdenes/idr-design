"""End-to-end smoke test: run the CLI against a tiny synthetic target."""
from pathlib import Path

import pytest


@pytest.mark.gpu
@pytest.mark.slow
def test_cli_smoke_runs_end_to_end(tmp_path: Path):
    from idrdesign.cli import main

    # Write a tiny FASTA.
    fa = tmp_path / "tiny.fasta"
    fa.write_text(">tiny\nACDEFGHIKLMNPQRSTVWY\n")

    out = tmp_path / "run_out"

    exit_code = main([
        "--target-fasta", str(fa),
        "--target-name", "tiny",
        "--output-dir", str(out),
        "--n-designs", "1",
        "--target-similarity", "0.80",
        "--composition-mode", "none",
        "--n-conformers", "10",
        "--max-iterations", "20",
        "--max-wall-seconds", "60",
        "--seed", "0",
    ])

    assert exit_code == 0
    assert (out / "config.yaml").exists()
    assert (out / "designs.fasta").exists()
    assert (out / "metrics.csv").exists()
    assert (out / "report.html").exists()
    # Ensembles saved
    ensembles_dir = out / "ensembles"
    assert ensembles_dir.exists()
    assert any(ensembles_dir.iterdir())

    # Minimal sanity on the report
    html = (out / "report.html").read_text()
    assert "tiny" in html
    assert "<svg" in html
