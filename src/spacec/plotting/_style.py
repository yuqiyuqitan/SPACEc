"""Shared publication-ready plotting style and export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import seaborn as sns

DEFAULT_FONT_FAMILIES = ["Inter", "Source Sans Pro", "Helvetica", "Arial", "sans-serif"]
DEFAULT_DISCRETE_PALETTE = "colorblind"

PUBLICATION_RCPARAMS = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": DEFAULT_FONT_FAMILIES,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.25,
    "lines.markersize": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply_publication_style() -> None:
    """Apply a unified publication-grade style across all matplotlib/seaborn plots."""
    mpl.rcParams.update(PUBLICATION_RCPARAMS)
    sns.set_theme(
        style="ticks",
        context="paper",
        palette=DEFAULT_DISCRETE_PALETTE,
        rc=PUBLICATION_RCPARAMS,
    )


def get_categorical_palette(
    n_colors: int, palette: str = DEFAULT_DISCRETE_PALETTE
) -> list[tuple[float, float, float]]:
    """Return a deterministic categorical palette."""
    return sns.color_palette(palette, n_colors=n_colors)


def save_figure(
    fig,
    output_dir: str | Path,
    output_fname: str,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 300,
    transparent: bool = True,
    bbox_inches: str = "tight",
) -> list[Path]:
    """
    Save one figure into multiple formats with publication defaults.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure instance to save.
    output_dir : str or pathlib.Path
        Output directory.
    output_fname : str
        File stem (without extension).
    formats : iterable of str, optional
        Target formats, by default ("pdf", "svg", "png").
    dpi : int, optional
        Raster DPI, by default 300.
    transparent : bool, optional
        Whether the background is transparent, by default True.
    bbox_inches : str, optional
        Bounding box mode, by default "tight".

    Returns
    -------
    list[pathlib.Path]
        Paths of all written figure files.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(output_fname).stem if output_fname else "figure"

    saved_paths: list[Path] = []
    for fmt in formats:
        fmt_clean = str(fmt).lower().lstrip(".")
        outpath = outdir / f"{stem}.{fmt_clean}"
        fig.savefig(
            outpath,
            format=fmt_clean,
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
        )
        saved_paths.append(outpath)
    return saved_paths
