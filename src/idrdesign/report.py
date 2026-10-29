"""HTML report + CSV writer for idrdesign."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from idrdesign.optimize import DesignResult


def _metric_summary_row(
    name: str, metrics: dict[str, np.ndarray]
) -> dict[str, float]:
    return {
        "rg_mean": float(np.mean(metrics["rg"])),
        "rg_std": float(np.std(metrics["rg"])),
        "end_to_end_mean": float(np.mean(metrics["end_to_end"])),
        "end_to_end_std": float(np.std(metrics["end_to_end"])),
        "asphericity_mean": float(np.mean(metrics["asphericity"])),
        "asphericity_std": float(np.std(metrics["asphericity"])),
    }


def write_metrics_csv(
    *,
    path: str | Path,
    target_seq: str,
    designs: list[DesignResult],
    metrics: dict[str, dict[str, np.ndarray]],
) -> None:
    """Write a flat per-design CSV with scalar summary metrics.

    Row 0: target. Rows 1..N: designs in order.
    """
    rows: list[dict] = []

    target_row = {
        "name": "target",
        "length": len(target_seq),
        "final_cosine_similarity": 1.0,
        "convergence_status": "target",
        "seed": "",
        "iterations": 0,
        "wall_seconds": 0.0,
        "composition_mode": "",
    }
    target_row.update(_metric_summary_row("target", metrics["target"]))
    rows.append(target_row)

    for i, d in enumerate(designs):
        name = f"design_{i:02d}"
        row = {
            "name": name,
            "length": len(d.sequence),
            "final_cosine_similarity": d.final_cosine_similarity,
            "convergence_status": d.convergence_status,
            "seed": d.seed,
            "iterations": d.iterations,
            "wall_seconds": d.wall_seconds,
            "composition_mode": d.composition_mode,
        }
        row.update(_metric_summary_row(name, metrics[name]))
        rows.append(row)

    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_to_svg(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue().decode("utf-8")


def plot_rg_violins(ensemble_rgs: dict[str, np.ndarray]) -> str:
    """Overlaid violin plot of Rg distributions, target first."""
    names = list(ensemble_rgs.keys())
    data = [ensemble_rgs[n] for n in names]
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(names)), 4))
    parts = ax.violinplot(data, showmeans=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor("#E94E4E" if names[i] == "target" else "#4B8BBE")
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Rg (Å)")
    ax.set_title("Radius of gyration distribution")
    return _fig_to_svg(fig)


def plot_distance_delta_map(target_map: np.ndarray, design_map: np.ndarray) -> str:
    """Heatmap of (design - target) mean distance map."""
    delta = design_map - target_map
    vmax = float(np.abs(delta).max())
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("residue i")
    ax.set_ylabel("residue j")
    ax.set_title("Mean distance map: design − target (Å)")
    fig.colorbar(im, ax=ax)
    return _fig_to_svg(fig)


def plot_trajectories(trajectories: dict[str, list[tuple[int, float]]]) -> str:
    """Plot optimization trajectories (iteration, cosine_similarity) per design."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, pts in trajectories.items():
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=name, alpha=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cosine similarity to target")
    ax.set_title("Optimization trajectories")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    return _fig_to_svg(fig)


import datetime

from jinja2 import Environment, FileSystemLoader, select_autoescape


def _load_template_env():
    tpl_dir = Path(__file__).parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        autoescape=select_autoescape(enabled_extensions=("html", "j2")),
    )


def render_report(
    *,
    path: str | Path,
    target_name: str,
    target_seq: str,
    designs: list[DesignResult],
    metrics: dict[str, dict[str, np.ndarray]],
    trajectories: dict[str, list[tuple[int, float]]],
    config_summary: dict,
) -> None:
    """Render and write the self-contained HTML report."""
    rg_data = {"target": metrics["target"]["rg"]}
    ree_data = {"target": metrics["target"]["end_to_end"]}
    asph_data = {"target": metrics["target"]["asphericity"]}
    for i, d in enumerate(designs):
        name = f"design_{i:02d}"
        rg_data[name] = metrics[name]["rg"]
        ree_data[name] = metrics[name]["end_to_end"]
        asph_data[name] = metrics[name]["asphericity"]

    rg_svg = plot_rg_violins(rg_data)
    ree_svg = plot_rg_violins(ree_data)    # reuse violin plotter; relabels would need own fn
    asph_svg = plot_rg_violins(asph_data)
    trajectories_svg = plot_trajectories(trajectories)

    delta_maps: list[tuple[str, str]] = []
    target_map = metrics["target"]["mean_distance_map"]
    for i, _ in enumerate(designs):
        name = f"design_{i:02d}"
        svg = plot_distance_delta_map(target_map, metrics[name]["mean_distance_map"])
        delta_maps.append((name, svg))

    env = _load_template_env()
    tpl = env.get_template("report.html.j2")
    html = tpl.render(
        target_name=target_name,
        target_seq=target_seq,
        designs=designs,
        config_summary=config_summary,
        rg_svg=rg_svg,
        ree_svg=ree_svg,
        asph_svg=asph_svg,
        trajectories_svg=trajectories_svg,
        delta_maps=delta_maps,
        generated_at=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
