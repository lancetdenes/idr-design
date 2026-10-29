"""Render FINCHES intermaps as a PNG for the web page.

One figure per forcefield: target | design | difference.

Two deliberate choices:

* Target and design share one symmetric color scale, so the two panels can be
  compared by eye. A per-panel scale would make a close design look different
  from its target purely through rescaling.
* The difference panel gets its own symmetric scale, because its range is much
  smaller than the maps it came from and would otherwise be a flat gray square.

Signed data, so the palette is diverging: cool and warm poles with a neutral
gray midpoint, which keeps "no interaction" reading as nothing. Negative is
blue, positive is red, matching the usual convention for these maps.
"""
from __future__ import annotations

import base64
import io

import numpy as np

# Diverging pair and chart ink. Kept as named constants so the whole set can be
# retuned in one place.
COOL = "#2a78d6"
MID = "#f0efec"
WARM = "#e34948"
INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"


def _configure(mpl) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "DejaVu Sans"],
        "font.size": 6,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.titlesize": 6,
        "axes.linewidth": 0.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("finches_div", [COOL, MID, WARM], N=256)


def render_intermap_figure(
    target_map: np.ndarray,
    design_map: np.ndarray,
    *,
    forcefield: str,
    window_size: int,
    dpi: int = 200,
) -> str:
    """Return a base64-encoded PNG comparing two intermaps.

    The returned string is the payload of a `data:image/png;base64,` URI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    _configure(matplotlib)

    target_map = np.asarray(target_map, dtype=float)
    design_map = np.asarray(design_map, dtype=float)
    if target_map.shape != design_map.shape:
        raise ValueError(
            f"map shapes differ: {target_map.shape} vs {design_map.shape}"
        )

    delta = design_map - target_map

    # Shared symmetric limits for the two maps; separate ones for the delta.
    lim = float(np.abs(np.concatenate([target_map.ravel(), design_map.ravel()])).max())
    dlim = float(np.abs(delta).max()) or 1.0

    cmap = _cmap()
    n = target_map.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.0), constrained_layout=True)

    panels = [
        (axes[0], target_map, "Input", lim),
        (axes[1], design_map, "Design", lim),
        (axes[2], delta, "Design − input", dlim),
    ]
    images = []
    for ax, data, title, vlim in panels:
        im = ax.imshow(
            data, cmap=cmap, vmin=-vlim, vmax=vlim,
            origin="lower", interpolation="nearest", aspect="equal",
        )
        images.append(im)
        ax.set_title(title, pad=3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.tick_params(length=2, width=0.5, pad=1.5)
        for s in ax.spines.values():
            s.set_linewidth(0.5)

    axes[0].set_ylabel("window start")
    for ax in axes:
        ax.set_xlabel("window start")

    # One bar for the shared pair, one for the delta.
    cb1 = fig.colorbar(images[0], ax=[axes[0], axes[1]], fraction=0.046, pad=0.02)
    cb1.set_label(f"{forcefield} interaction", labelpad=2)
    cb1.ax.tick_params(length=2, width=0.5, pad=1.5)
    cb1.outline.set_linewidth(0.5)

    cb2 = fig.colorbar(images[2], ax=axes[2], fraction=0.046, pad=0.02)
    cb2.set_label("difference", labelpad=2)
    cb2.ax.tick_params(length=2, width=0.5, pad=1.5)
    cb2.outline.set_linewidth(0.5)

    fig.suptitle(
        f"{forcefield}  ·  window {window_size}  ·  {n}×{n} windows",
        fontsize=6, y=1.04,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
