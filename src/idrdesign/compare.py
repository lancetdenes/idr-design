"""Cross-design comparison for idrdesign.

Pure numeric helpers (composition + map agreement) are unit-tested in
tests/test_compare.py. The I/O + plotting driver (added below the helpers) is
validated by the end-to-end design run and the plot-review loop.
"""
from __future__ import annotations

import numpy as np

AA20 = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX = {a: i for i, a in enumerate(AA20)}


def aa_frequency_vector(seq: str) -> np.ndarray:
    """Amino-acid frequency vector over the 20 standard residues (sums to 1).

    Non-standard characters are ignored; normalization is over recognised
    residues so the vector sums to 1 for any sequence with >= 1 standard aa.
    """
    counts = np.zeros(20, dtype=float)
    for ch in seq:
        idx = _AA_INDEX.get(ch)
        if idx is not None:
            counts[idx] += 1.0
    total = counts.sum()
    if total == 0:
        raise ValueError("sequence contains no standard amino acids")
    return counts / total


def composition_l1(a: str, b: str) -> float:
    """L1 distance between amino-acid frequency vectors (range 0..2)."""
    return float(np.abs(aa_frequency_vector(a) - aa_frequency_vector(b)).sum())


def composition_jsd(a: str, b: str) -> float:
    """Jensen-Shannon divergence (base 2, range 0..1) between aa frequencies."""
    p = aa_frequency_vector(a)
    q = aa_frequency_vector(b)
    m = 0.5 * (p + q)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def mean_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error between two equal-shaped arrays."""
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two arrays (flattened)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# I/O + driver (integration code; validated by the end-to-end run, not unit
# tests). Loads two idrdesign run directories (free + exact), reloads their
# saved STARLING ensembles, and emits a single merged comparison deliverable.
# ---------------------------------------------------------------------------

from pathlib import Path

CONTACT_THRESH = 11  # match idrdesign.ensemble.extract_metrics

# Plotting colors (consistent across all figures).
C_WT = "#333333"
C_FREE = "#3B7DD8"
C_EXACT = "#E2872B"
_GROUP_COLOR = {"target": C_WT, "free": C_FREE, "exact": C_EXACT}


def _load_ensemble_metrics(starling_path) -> dict:
    """Reload a .starling file and pull distance-map-derived observables.

    All of Rg, Ree, the mean distance map, and the mean contact map derive
    directly from the stored distance maps, so no 3D reconstruction occurs
    (ignore_structures=True). Asphericity is NOT recomputed here (it needs
    coords); it is read from the run's metrics.csv instead.
    """
    import starling

    ens = starling.load_ensemble(str(starling_path), ignore_structures=True)
    return {
        "rg": np.asarray(ens.radius_of_gyration(), dtype=float).ravel(),
        "ree": np.asarray(ens.end_to_end_distance(), dtype=float).ravel(),
        "mean_dmap": np.asarray(ens.distance_maps(return_mean=True), dtype=float),
        "mean_cmap": np.asarray(
            ens.contact_map(contact_thresh=CONTACT_THRESH, return_mean=True), dtype=float
        ),
    }


def _collect_group(run_dir, label: str) -> tuple[list[dict], str]:
    """Collect per-design records from one idrdesign run directory.

    Returns (design_records, target_sequence). Each design record relabels
    design_NN -> {label}_NN and attaches its reloaded ensemble observables.
    """
    import pandas as pd

    from idrdesign.io import read_fasta

    run_dir = Path(run_dir)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    fasta = read_fasta(run_dir / "designs.fasta")
    ens_dir = run_dir / "ensembles"

    records: list[dict] = []
    design_rows = metrics[metrics["name"] != "target"].reset_index(drop=True)
    for _, r in design_rows.iterrows():
        src = str(r["name"])  # e.g. design_00
        suffix = src.split("_")[1]
        new_name = f"{label}_{suffix}"
        em = _load_ensemble_metrics(ens_dir / f"{src}.starling")
        records.append(
            {
                "name": new_name,
                "group": label,
                "sequence": fasta[src],
                "composition_mode": str(r["composition_mode"]),
                "final_cosine_similarity": float(r["final_cosine_similarity"]),
                "convergence_status": str(r["convergence_status"]),
                "iterations": int(r["iterations"]),
                "asphericity_mean": float(r["asphericity_mean"]),
                "asphericity_std": float(r["asphericity_std"]),
                "ens": em,
            }
        )
    return records, fasta["target"]


def build_records(free_dir, exact_dir) -> tuple[dict, list[dict], str]:
    """Build the target reference + ordered design records for both groups.

    Returns (target, designs, target_seq). target carries the WT reference
    ensemble (taken from the free run) plus the exact run's target Rg/Ree means
    as a run-to-run stochastic baseline.
    """
    free, target_seq = _collect_group(free_dir, "free")
    exact, target_seq_exact = _collect_group(exact_dir, "exact")
    if target_seq != target_seq_exact:
        raise ValueError("target sequence differs between the free and exact runs")

    target_ens = _load_ensemble_metrics(Path(free_dir) / "ensembles" / "target.starling")
    exact_target_ens = _load_ensemble_metrics(
        Path(exact_dir) / "ensembles" / "target.starling"
    )
    import pandas as pd

    fm = pd.read_csv(Path(free_dir) / "metrics.csv")
    target_asph = fm[fm["name"] == "target"].iloc[0]
    target = {
        "name": "target",
        "group": "target",
        "sequence": target_seq,
        "ens": target_ens,
        "asphericity_mean": float(target_asph["asphericity_mean"]),
        "asphericity_std": float(target_asph["asphericity_std"]),
        "exact_run_rg_mean": float(np.mean(exact_target_ens["rg"])),
        "exact_run_ree_mean": float(np.mean(exact_target_ens["ree"])),
        # Independent STARLING generation of the SAME WT sequence. The MAE
        # between this and target["ens"]["mean_dmap"] is the run-to-run
        # stochastic noise floor: no design can match WT closer than this.
        "exact_run_mean_dmap": exact_target_ens["mean_dmap"],
    }
    return target, free + exact, target_seq


def compute_comparison_rows(target: dict, designs: list[dict], target_seq: str) -> list[dict]:
    """One flat dict per design with encoder, ensemble, and composition metrics."""
    t = target["ens"]
    t_rg, t_ree = float(np.mean(t["rg"])), float(np.mean(t["ree"]))
    rows: list[dict] = []
    for d in designs:
        e = d["ens"]
        rows.append(
            {
                "name": d["name"],
                "group": d["group"],
                "composition_mode": d["composition_mode"],
                "length": len(d["sequence"]),
                "final_cosine_similarity": d["final_cosine_similarity"],
                "convergence_status": d["convergence_status"],
                "iterations": d["iterations"],
                "rg_mean": float(np.mean(e["rg"])),
                "rg_std": float(np.std(e["rg"])),
                "ree_mean": float(np.mean(e["ree"])),
                "ree_std": float(np.std(e["ree"])),
                "asphericity_mean": d["asphericity_mean"],
                "asphericity_std": d["asphericity_std"],
                "delta_rg_mean": float(np.mean(e["rg"])) - t_rg,
                "delta_ree_mean": float(np.mean(e["ree"])) - t_ree,
                "delta_asphericity_mean": d["asphericity_mean"] - target["asphericity_mean"],
                "distance_map_mae": mean_abs_error(e["mean_dmap"], t["mean_dmap"]),
                "contact_map_pearson_r": pearson_r(e["mean_cmap"], t["mean_cmap"]),
                "composition_l1_vs_wt": composition_l1(d["sequence"], target_seq),
                "composition_jsd_vs_wt": composition_jsd(d["sequence"], target_seq),
            }
        )
    return rows


# ----------------------------- plotting ------------------------------------

def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans", "DejaVu Sans"]
    plt.rcParams["font.size"] = 6
    plt.rcParams["axes.titlesize"] = 6
    plt.rcParams["axes.labelsize"] = 6
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6
    plt.rcParams["legend.fontsize"] = 6
    return plt


def _group_legend(fig, plt, groups=("target", "free", "exact")):
    from matplotlib.patches import Patch

    labels = {
        "target": "WT (target)",
        "free": "Free (none)",
        "exact": "Exact (shuffle)",
    }
    handles = [Patch(facecolor=_GROUP_COLOR[g], label=labels[g]) for g in groups]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.0, 0.98),
        frameon=False,
        borderaxespad=0.0,
    )


def _design_order(designs: list[dict]) -> list[dict]:
    return [d for d in designs if d["group"] == "free"] + [
        d for d in designs if d["group"] == "exact"
    ]


def plot_distributions(target, designs, out_path):
    """Rg and Ree per-conformer distributions: WT + 6 designs (violins)."""
    plt = _setup_mpl()
    from matplotlib.ticker import MaxNLocator

    ordered = _design_order(designs)
    entries = [target] + ordered
    names = [e["name"] for e in entries]
    colors = [_GROUP_COLOR[e["group"]] for e in entries]
    pos = list(range(1, len(entries) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), constrained_layout=True)
    for ax, key, ylabel in zip(axes, ("rg", "ree"), ("Rg (Å)", "Ree (Å)")):
        data = [e["ens"][key] for e in entries]
        parts = ax.violinplot(data, positions=pos, showmeans=True, widths=0.8)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.6)
            pc.set_edgecolor("none")
        for k in ("cmeans", "cmaxes", "cmins", "cbars"):
            if k in parts:
                parts[k].set_color("#555555")
                parts[k].set_linewidth(0.6)
        ax.set_xticks(pos)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.margins(y=0.05)
    axes[0].set_title("Radius of gyration")
    axes[1].set_title("End-to-end distance")
    _group_legend(fig, plt)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_agreement(rows, out_path, noise_floor_mae=None):
    """2x2: ΔRg/ΔRee, asphericity mean±std, distance-map MAE, contact-map r."""
    plt = _setup_mpl()
    import numpy as _np
    from matplotlib.ticker import MaxNLocator

    names = [r["name"] for r in rows]
    colors = [_GROUP_COLOR[r["group"]] for r in rows]
    x = _np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.4), constrained_layout=True)

    # (0,0) ΔRg and ΔRee grouped bars
    ax = axes[0, 0]
    w = 0.4
    ax.bar(x - w / 2, [r["delta_rg_mean"] for r in rows], w, color=colors, label="ΔRg")
    ax.bar(
        x + w / 2,
        [r["delta_ree_mean"] for r in rows],
        w,
        color=colors,
        alpha=0.5,
        hatch="//",
        label="ΔRee",
    )
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_ylabel("design − WT (Å)")
    ax.set_title("Mean Rg / Ree deviation (solid=Rg, hatched=Ree)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # (0,1) asphericity mean ± std
    ax = axes[0, 1]
    ax.errorbar(
        x,
        [r["asphericity_mean"] for r in rows],
        yerr=[r["asphericity_std"] for r in rows],
        fmt="o",
        ms=3,
        capsize=2,
        lw=0.8,
        ecolor="#888888",
        mfc="none",
        mec="#333333",
    )
    for xi, r in zip(x, rows):
        ax.plot(xi, r["asphericity_mean"], "o", ms=3, color=_GROUP_COLOR[r["group"]])
    ax.set_ylabel("asphericity")
    ax.set_title("Asphericity (mean ± std)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # (1,0) distance-map MAE, with the WT run-to-run noise floor as a reference
    ax = axes[1, 0]
    ax.bar(x, [r["distance_map_mae"] for r in rows], color=colors)
    if noise_floor_mae is not None:
        ax.axhline(noise_floor_mae, ls="--", lw=0.8, color="#888888")
        ax.text(
            len(rows) - 0.5,
            noise_floor_mae,
            f" noise floor {noise_floor_mae:.2f} Å",
            va="bottom",
            ha="right",
            color="#666666",
        )
    ax.set_ylabel("MAE (Å)")
    ax.set_title("Mean distance-map MAE vs WT")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # (1,1) contact-map Pearson r
    ax = axes[1, 1]
    ax.bar(x, [r["contact_map_pearson_r"] for r in rows], color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Pearson r")
    ax.set_title("Mean contact-map agreement vs WT")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    _group_legend(fig, plt, groups=("free", "exact"))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_composition(rows, out_path):
    """Composition distance to WT per design (the free-vs-exact control)."""
    plt = _setup_mpl()
    import numpy as _np
    from matplotlib.ticker import MaxNLocator

    names = [r["name"] for r in rows]
    colors = [_GROUP_COLOR[r["group"]] for r in rows]
    x = _np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), constrained_layout=True)
    axes[0].bar(x, [r["composition_l1_vs_wt"] for r in rows], color=colors)
    axes[0].set_ylabel("composition L1 vs WT")
    axes[0].set_title("Amino-acid composition L1 distance")
    axes[1].bar(x, [r["composition_jsd_vs_wt"] for r in rows], color=colors)
    axes[1].set_ylabel("composition JSD vs WT (bits)")
    axes[1].set_title("Amino-acid composition JS divergence")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.margins(y=0.05)
    _group_legend(fig, plt, groups=("free", "exact"))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distance_delta_maps(target, designs, out_path):
    """2x3 grid of (design − WT) mean distance-map heatmaps, shared color scale."""
    plt = _setup_mpl()

    ordered = _design_order(designs)
    t_map = target["ens"]["mean_dmap"]
    deltas = [d["ens"]["mean_dmap"] - t_map for d in ordered]
    vmax = max(float(np.abs(dm).max()) for dm in deltas)

    fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.4), constrained_layout=True)
    im = None
    for ax, d, dm in zip(axes.ravel(), ordered, deltas):
        im = ax.imshow(dm, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
        ax.set_title(d["name"])
        ax.set_xlabel("residue i")
        ax.set_ylabel("residue j")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("design − WT mean distance (Å)")
    fig.suptitle("Mean distance-map difference vs WT", fontsize=6)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distance_maps_raw(target, designs, out_path, noise_floor_mae=None):
    """Raw mean distance maps on a shared absolute scale: WT + the 6 designs.

    2x4 grid. WT (target) is repeated as the leading reference in each row so
    every design sits next to the map it is meant to match; top row = free
    designs, bottom row = exact designs. All panels share one color scale
    (0..global max). Each design title carries its mean distance-map MAE vs WT;
    the WT run-to-run noise floor (irreducible STARLING sampling noise) is
    reported in the suptitle so the MAEs are interpretable.
    """
    plt = _setup_mpl()
    from matplotlib.ticker import FixedLocator

    free = [d for d in designs if d["group"] == "free"]
    exact = [d for d in designs if d["group"] == "exact"]
    t_map = target["ens"]["mean_dmap"]
    n = t_map.shape[0]
    vmax = max(
        float(t_map.max()),
        max(float(d["ens"]["mean_dmap"].max()) for d in designs),
    )
    ticks = [t for t in (0, 100, 200, 300) if t <= n - 1]

    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.1), constrained_layout=True)
    im = None
    for r, group in enumerate((free, exact)):
        # col 0: WT reference (same map both rows; marked with a dark border).
        ax = axes[r, 0]
        im = ax.imshow(t_map, cmap="viridis", vmin=0, vmax=vmax, origin="upper")
        ax.set_title("WT (target)", color=C_WT)
        for s in ax.spines.values():
            s.set_edgecolor(C_WT)
            s.set_linewidth(1.4)
        # cols 1..3: the three designs of this group
        for c in range(1, 4):
            d = group[c - 1]
            dm = d["ens"]["mean_dmap"]
            im = axes[r, c].imshow(dm, cmap="viridis", vmin=0, vmax=vmax, origin="upper")
            axes[r, c].set_title(
                f"{d['name']}  (MAE {mean_abs_error(dm, t_map):.2f} Å)",
                color=_GROUP_COLOR[d["group"]],
            )
        for c in range(4):
            ax = axes[r, c]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            if r == 1:
                ax.set_xlabel("residue i")
            if c == 0:
                ax.set_ylabel("residue j")
            else:
                ax.set_yticklabels([])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("mean distance (Å)")
    sup = "STARLING mean distance maps — WT vs ensemble-matched designs"
    if noise_floor_mae is not None:
        sup += f"\n(WT vs WT run-to-run noise floor: MAE {noise_floor_mae:.2f} Å — designs can't beat this)"
    fig.suptitle(sup, fontsize=6)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------- AFRC null-model normalization ----------------------
# STARLING-paper "Null model normalization": normalize inter-residue distances
# by the Analytical Flory Random Coil (AFRC) model — the expected distances for
# a sequence-specific Gaussian (phantom) chain — to divide out the
# sequence-separation gradient and expose compaction/expansion. The resulting
# "scaling map" D_obs / D_AFRC is dimensionless: <1 = more compact than a random
# coil, >1 = more expanded. Each sequence is normalized by its OWN AFRC, as in
# the paper. (STARLING maps are Cα-Cα; AFRC distances are residue-COM, so the
# ratio is a visualization normalization, not an exact physical quantity.)
_AFRC_CACHE: dict = {}


def afrc_distance_map(seq: str) -> np.ndarray:
    """Symmetric AFRC mean inter-residue distance map (Å) for `seq` (cached)."""
    cached = _AFRC_CACHE.get(seq)
    if cached is not None:
        return cached
    from afrc import AnalyticalFRC

    amap = np.asarray(
        AnalyticalFRC(seq).get_distance_map(symmetric_map=True), dtype=float
    )
    _AFRC_CACHE[seq] = amap
    return amap


def scaling_map(mean_dmap: np.ndarray, seq: str) -> np.ndarray:
    """D_STARLING / D_AFRC scaling map; diagonal (0/0) set to NaN."""
    amap = afrc_distance_map(seq)
    with np.errstate(divide="ignore", invalid="ignore"):
        sm = np.asarray(mean_dmap, dtype=float) / amap
    np.fill_diagonal(sm, np.nan)
    return sm


def plot_distance_maps_afrc(target, designs, out_path):
    """AFRC-normalized scaling maps (paper null-model normalization): WT + designs.

    Each panel is D_STARLING / D_AFRC for that sequence, so the
    sequence-separation gradient is divided out and compaction (<1, blue) vs
    expansion (>1, red) relative to a Gaussian random coil is directly visible.
    WT (target) leads each row; all panels share one diverging scale centered at
    1 so designs are directly comparable to WT and to each other.
    """
    plt = _setup_mpl()
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import FixedLocator

    free = [d for d in designs if d["group"] == "free"]
    exact = [d for d in designs if d["group"] == "exact"]
    sms = {"WT": scaling_map(target["ens"]["mean_dmap"], target["sequence"])}
    for d in designs:
        sms[d["name"]] = scaling_map(d["ens"]["mean_dmap"], d["sequence"])

    dev = max(float(np.nanpercentile(np.abs(sm - 1.0), 99)) for sm in sms.values())
    dev = min(max(dev, 0.05), 0.6)  # robust, capped
    norm = TwoSlopeNorm(vmin=1 - dev, vcenter=1.0, vmax=1 + dev)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")
    t_sm = sms["WT"]
    n = t_sm.shape[0]
    ticks = [t for t in (0, 100, 200, 300) if t <= n - 1]

    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.1), constrained_layout=True)
    im = None
    for r, group in enumerate((free, exact)):
        ax = axes[r, 0]
        im = ax.imshow(t_sm, cmap=cmap, norm=norm, origin="upper")
        ax.set_title("WT (target)", color=C_WT)
        for s in ax.spines.values():
            s.set_edgecolor(C_WT)
            s.set_linewidth(1.4)
        for c in range(1, 4):
            d = group[c - 1]
            im = axes[r, c].imshow(sms[d["name"]], cmap=cmap, norm=norm, origin="upper")
            axes[r, c].set_title(d["name"], color=_GROUP_COLOR[d["group"]])
        for c in range(4):
            ax = axes[r, c]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            if r == 1:
                ax.set_xlabel("residue i")
            if c == 0:
                ax.set_ylabel("residue j")
            else:
                ax.set_yticklabels([])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, extend="both")
    cbar.set_label("STARLING / AFRC distance (×)")
    fig.suptitle(
        "AFRC-normalized distance maps (scaling maps) — WT vs designs\n"
        "blue = compact vs random coil, red = expanded; sequence-separation gradient divided out",
        fontsize=6,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distance_delta_maps_afrc(target, designs, out_path):
    """AFRC-normalized difference maps: (design − WT) scaling-map difference.

    For each design: (D_design/AFRC_design) − (D_WT/AFRC_WT), centered at 0 on a
    shared diverging scale. Red = design more expanded than WT relative to the
    random-coil baseline; blue = more compact. Removes the gradient from the
    difference so short- and long-range deviations are comparable.
    """
    plt = _setup_mpl()
    from matplotlib.ticker import FixedLocator

    ordered = _design_order(designs)
    t_sm = scaling_map(target["ens"]["mean_dmap"], target["sequence"])
    deltas = [
        (d, scaling_map(d["ens"]["mean_dmap"], d["sequence"]) - t_sm) for d in ordered
    ]
    vmax = max(float(np.nanpercentile(np.abs(dm), 99)) for _, dm in deltas)
    vmax = min(max(vmax, 0.02), 0.6)
    n = t_sm.shape[0]
    ticks = [t for t in (0, 100, 200, 300) if t <= n - 1]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4), constrained_layout=True)
    im = None
    for ax, (d, dm) in zip(axes.ravel(), deltas):
        im = ax.imshow(dm, cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper")
        ax.set_title(d["name"], color=_GROUP_COLOR[d["group"]])
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xlabel("residue i")
        ax.set_ylabel("residue j")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, extend="both")
    cbar.set_label("(design − WT) scaling-map difference")
    fig.suptitle("AFRC-normalized mean distance-map difference vs WT", fontsize=6)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------- GS-control (empirical null) normalization ---------------
# Like the AFRC normalization, but the reference is an EMPIRICAL STARLING
# ensemble of a length-matched pure GS-repeat (Gly-Ser) control instead of the
# analytical random coil. Same method (STARLING) and length as the data, and a
# single shared reference for every panel, so D_obs / D_GS is directly
# comparable across WT and all designs. <1 = more compact than the GS control,
# >1 = more expanded.


def gs_scaling_map(mean_dmap: np.ndarray, gs_dmap: np.ndarray) -> np.ndarray:
    """D_STARLING / D_GS scaling map vs the GS-repeat control; diagonal -> NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        sm = np.asarray(mean_dmap, dtype=float) / np.asarray(gs_dmap, dtype=float)
    np.fill_diagonal(sm, np.nan)
    return sm


def plot_distance_maps_gsnorm(target, designs, gs_dmap, out_path):
    """GS-control normalized scaling maps: WT + designs, all / the same GS-repeat
    STARLING control. Removes the sequence-separation gradient and makes the
    panels directly comparable (blue = compact vs GS, red = expanded vs GS)."""
    plt = _setup_mpl()
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import FixedLocator

    free = [d for d in designs if d["group"] == "free"]
    exact = [d for d in designs if d["group"] == "exact"]
    sms = {"WT": gs_scaling_map(target["ens"]["mean_dmap"], gs_dmap)}
    for d in designs:
        sms[d["name"]] = gs_scaling_map(d["ens"]["mean_dmap"], gs_dmap)

    dev = max(float(np.nanpercentile(np.abs(sm - 1.0), 99)) for sm in sms.values())
    dev = min(max(dev, 0.05), 0.6)
    norm = TwoSlopeNorm(vmin=1 - dev, vcenter=1.0, vmax=1 + dev)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")
    t_sm = sms["WT"]
    n = t_sm.shape[0]
    ticks = [t for t in (0, 100, 200, 300) if t <= n - 1]

    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.1), constrained_layout=True)
    im = None
    for r, group in enumerate((free, exact)):
        ax = axes[r, 0]
        im = ax.imshow(t_sm, cmap=cmap, norm=norm, origin="upper")
        ax.set_title("WT (target)", color=C_WT)
        for s in ax.spines.values():
            s.set_edgecolor(C_WT)
            s.set_linewidth(1.4)
        for c in range(1, 4):
            d = group[c - 1]
            im = axes[r, c].imshow(sms[d["name"]], cmap=cmap, norm=norm, origin="upper")
            axes[r, c].set_title(d["name"], color=_GROUP_COLOR[d["group"]])
        for c in range(4):
            ax = axes[r, c]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            if r == 1:
                ax.set_xlabel("residue i")
            if c == 0:
                ax.set_ylabel("residue j")
            else:
                ax.set_yticklabels([])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, extend="both")
    cbar.set_label("STARLING / GS-control distance (×)")
    fig.suptitle(
        "GS-control–normalized distance maps — WT vs designs\n"
        "blue = compact vs GS repeat, red = expanded; sequence-separation gradient divided out",
        fontsize=6,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(rows, target, path, noise_floor_mae=None):
    """Markdown summary table + a data-derived takeaway line."""
    import numpy as _np

    lines = ["# RUNX2 WT — six ensemble-matched designs", ""]
    lines.append(
        f"Target: WT ({len(target['sequence'])} aa). Reference ensemble Rg mean "
        f"{_np.mean(target['ens']['rg']):.1f} Å, Ree mean "
        f"{_np.mean(target['ens']['ree']):.1f} Å "
        f"(exact-run target baseline: Rg {target['exact_run_rg_mean']:.1f} Å, "
        f"Ree {target['exact_run_ree_mean']:.1f} Å).",
    )
    if noise_floor_mae is not None:
        lines.append("")
        lines.append(
            f"Distance-map MAE noise floor (WT vs an independent WT generation): "
            f"**{noise_floor_mae:.2f} Å**. This is the irreducible STARLING "
            f"run-to-run sampling noise — no design can match WT below it."
        )
    lines.append("")
    hdr = (
        "| design | mode | cos sim | conv | ΔRg (Å) | ΔRee (Å) | dmap MAE (Å) | "
        "cmap r | comp L1 | comp JSD |"
    )
    lines.append(hdr)
    lines.append("|" + "---|" * 10)
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['composition_mode']} | "
            f"{r['final_cosine_similarity']:.4f} | {r['convergence_status']} | "
            f"{r['delta_rg_mean']:+.2f} | {r['delta_ree_mean']:+.2f} | "
            f"{r['distance_map_mae']:.2f} | {r['contact_map_pearson_r']:.4f} | "
            f"{r['composition_l1_vs_wt']:.4f} | {r['composition_jsd_vs_wt']:.4f} |"
        )
    free = [r for r in rows if r["group"] == "free"]
    exact = [r for r in rows if r["group"] == "exact"]
    lines += [
        "",
        "## Takeaway",
        "",
        f"- All {len(rows)} designs reached cosine similarity "
        f"{min(r['final_cosine_similarity'] for r in rows):.3f}–"
        f"{max(r['final_cosine_similarity'] for r in rows):.3f} to the WT encoder embedding.",
        f"- Free designs diverge in composition (L1 to WT "
        f"{min(r['composition_l1_vs_wt'] for r in free):.3f}–"
        f"{max(r['composition_l1_vs_wt'] for r in free):.3f}); "
        f"exact designs preserve it (L1 max {max(r['composition_l1_vs_wt'] for r in exact):.1e}).",
        f"- Ensemble agreement: |ΔRg| ≤ {max(abs(r['delta_rg_mean']) for r in rows):.2f} Å, "
        f"|ΔRee| ≤ {max(abs(r['delta_ree_mean']) for r in rows):.2f} Å, "
        f"distance-map MAE ≤ {max(r['distance_map_mae'] for r in rows):.2f} Å, "
        f"contact-map r ≥ {min(r['contact_map_pearson_r'] for r in rows):.3f}.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def build_comparison(free_dir, exact_dir, output_dir, target_name: str = "WT", gs_ref=None) -> Path:
    """Full combine step: load both runs, compute metrics, write CSV + figs + summary."""
    import pandas as pd

    from idrdesign.io import write_fasta

    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    target, designs, target_seq = build_records(free_dir, exact_dir)
    rows = compute_comparison_rows(target, designs, target_seq)

    # WT-vs-WT distance-map MAE across two independent STARLING generations:
    # the irreducible run-to-run noise floor for the dmap-MAE metric.
    noise_floor_mae = mean_abs_error(
        target["ens"]["mean_dmap"], target["exact_run_mean_dmap"]
    )

    # combined_metrics.csv (target row first, then designs)
    target_row = {
        "name": "target",
        "group": "target",
        "composition_mode": "target",
        "length": len(target_seq),
        "final_cosine_similarity": 1.0,
        "convergence_status": "target",
        "iterations": 0,
        "rg_mean": float(np.mean(target["ens"]["rg"])),
        "rg_std": float(np.std(target["ens"]["rg"])),
        "ree_mean": float(np.mean(target["ens"]["ree"])),
        "ree_std": float(np.std(target["ens"]["ree"])),
        "asphericity_mean": target["asphericity_mean"],
        "asphericity_std": target["asphericity_std"],
        "delta_rg_mean": 0.0,
        "delta_ree_mean": 0.0,
        "delta_asphericity_mean": 0.0,
        "distance_map_mae": 0.0,
        "contact_map_pearson_r": 1.0,
        "composition_l1_vs_wt": 0.0,
        "composition_jsd_vs_wt": 0.0,
    }
    pd.DataFrame([target_row] + rows).to_csv(output_dir / "combined_metrics.csv", index=False)

    # combined designs.fasta: WT + free_00..02 + exact_00..02
    records = {"WT": target_seq}
    for d in _design_order(designs):
        records[d["name"]] = d["sequence"]
    write_fasta(output_dir / "designs.fasta", records)

    # figures
    plot_distributions(target, designs, fig_dir / "fig_rg_ree_distributions.png")
    plot_ensemble_agreement(
        rows, fig_dir / "fig_ensemble_agreement.png", noise_floor_mae=noise_floor_mae
    )
    plot_composition(rows, fig_dir / "fig_composition.png")
    plot_distance_delta_maps(target, designs, fig_dir / "fig_distance_delta_maps.png")
    plot_distance_maps_raw(
        target, designs, fig_dir / "fig_distance_maps_raw.png", noise_floor_mae=noise_floor_mae
    )
    # AFRC null-model normalized versions (paper "Null model normalization"):
    # divide out the sequence-separation gradient to make maps comparable.
    plot_distance_maps_afrc(target, designs, fig_dir / "fig_distance_maps_afrc.png")
    plot_distance_delta_maps_afrc(
        target, designs, fig_dir / "fig_distance_delta_maps_afrc.png"
    )
    # GS-control (empirical STARLING GS-repeat) normalization, if a reference
    # ensemble is supplied.
    if gs_ref is not None:
        import starling

        gs_ens = starling.load_ensemble(str(gs_ref), ignore_structures=True)
        gs_dmap = np.asarray(gs_ens.distance_maps(return_mean=True), dtype=float)
        plot_distance_maps_gsnorm(
            target, designs, gs_dmap, fig_dir / "fig_distance_maps_gsnorm.png"
        )

    write_summary_md(rows, target, output_dir / "summary.md", noise_floor_mae=noise_floor_mae)
    return output_dir
