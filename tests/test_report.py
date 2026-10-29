"""Tests for idrdesign.report."""
from pathlib import Path

import numpy as np
import pandas as pd

from idrdesign.optimize import DesignResult


def _fake_design(name: str, seq: str, cos_sim: float) -> DesignResult:
    return DesignResult(
        sequence=seq,
        final_cosine_similarity=cos_sim,
        iterations=100,
        wall_seconds=10.0,
        convergence_status="converged",
        seed=0,
        starting_sequence="GS" * (len(seq) // 2),
        composition_mode="none",
    )


def _fake_metrics() -> dict:
    rng = np.random.default_rng(0)
    return {
        "rg": rng.uniform(15, 20, size=5).astype(np.float32),
        "end_to_end": rng.uniform(30, 50, size=5).astype(np.float32),
        "mean_distance_map": rng.uniform(5, 10, size=(10, 10)).astype(np.float32),
        "contact_map": rng.uniform(0, 1, size=(10, 10)).astype(np.float32),
        "asphericity": rng.uniform(0, 0.5, size=5).astype(np.float32),
    }


def test_write_metrics_csv_has_expected_columns(tmp_path: Path):
    from idrdesign.report import write_metrics_csv

    designs = [_fake_design(f"design_{i:02d}", "A" * 10, 0.9 + i * 0.01) for i in range(3)]
    metrics = {
        "target": _fake_metrics(),
        "design_00": _fake_metrics(),
        "design_01": _fake_metrics(),
        "design_02": _fake_metrics(),
    }
    target_seq = "A" * 10

    csv_path = tmp_path / "metrics.csv"
    write_metrics_csv(
        path=csv_path,
        target_seq=target_seq,
        designs=designs,
        metrics=metrics,
    )
    df = pd.read_csv(csv_path)
    assert "name" in df.columns
    assert "final_cosine_similarity" in df.columns
    assert "rg_mean" in df.columns
    assert "rg_std" in df.columns
    assert "end_to_end_mean" in df.columns
    assert "asphericity_mean" in df.columns
    assert "convergence_status" in df.columns
    assert len(df) == 4   # target + 3 designs
    # Target row first
    assert df.iloc[0]["name"] == "target"


def test_plot_rg_violins_returns_svg_string():
    from idrdesign.report import plot_rg_violins

    rng = np.random.default_rng(0)
    ensemble_rgs = {
        "target": rng.uniform(15, 20, 100),
        "design_00": rng.uniform(16, 19, 100),
    }
    svg = plot_rg_violins(ensemble_rgs)
    assert isinstance(svg, str)
    assert svg.lstrip().startswith("<svg") or "<svg" in svg


def test_plot_distance_delta_map_returns_svg_string():
    from idrdesign.report import plot_distance_delta_map

    target = np.random.default_rng(0).uniform(5, 15, (10, 10)).astype(np.float32)
    design = target + np.random.default_rng(1).normal(0, 1, (10, 10)).astype(np.float32)
    svg = plot_distance_delta_map(target, design)
    assert "<svg" in svg


def test_plot_trajectory_returns_svg_string():
    from idrdesign.report import plot_trajectories

    # fake iter, cos_sim pairs
    trajectories = {
        "design_00": [(i, 1.0 - 1.0 / (i + 1)) for i in range(20)],
        "design_01": [(i, 1.0 - 0.5 / (i + 1)) for i in range(20)],
    }
    svg = plot_trajectories(trajectories)
    assert "<svg" in svg


def test_render_report_writes_self_contained_html(tmp_path: Path):
    from idrdesign.report import render_report

    designs = [_fake_design(f"design_{i:02d}", "A" * 10, 0.9 + i * 0.01) for i in range(2)]
    metrics = {
        "target": _fake_metrics(),
        "design_00": _fake_metrics(),
        "design_01": _fake_metrics(),
    }
    target_seq = "A" * 10
    trajectories = {
        "design_00": [(i, 1.0 - 1.0 / (i + 1)) for i in range(20)],
        "design_01": [(i, 1.0 - 0.5 / (i + 1)) for i in range(20)],
    }

    out = tmp_path / "report.html"
    render_report(
        path=out,
        target_name="WT",
        target_seq=target_seq,
        designs=designs,
        metrics=metrics,
        trajectories=trajectories,
        config_summary={"n_designs": 2, "target_similarity": 0.95},
    )
    html = out.read_text()
    assert "<html" in html
    assert "WT" in html
    assert "<svg" in html     # inline SVGs
    # No external JS/CSS refs (self-contained)
    assert "src=\"http" not in html
    assert "href=\"http" not in html
