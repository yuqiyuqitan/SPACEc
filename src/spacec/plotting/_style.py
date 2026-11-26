from __future__ import annotations

import contextlib
import contextvars
import copy
import json
import pathlib
import warnings
from dataclasses import asdict, dataclass, field
from typing import ClassVar, Dict, List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

_active_style: contextvars.ContextVar[Optional["SPACEcStyle"]] = contextvars.ContextVar(
    "_active_style", default=None
)
_default_style: Optional["SPACEcStyle"] = None


@dataclass
class SPACEcStyle:
    """Style configuration for spacec plotting functions.

    All parameters have publication-ready defaults calibrated for Nature journals
    (single column: 89 mm, double column: 183 mm; 7–8 pt fonts; 300 DPI).

    Usage
    -----
    >>> import spacec.plotting as pl
    >>> pl.set_style(pl.SPACEcStyle.publication())  # apply globally
    >>> pl.SPACEcStyle.publication().to_yaml("my_style.yaml")  # export for editing
    >>> pl.set_style("my_style.yaml")                          # load from file
    """

    # --- Typography ---
    font_family: str = "sans-serif"
    font_size_tiny: float = 6.0
    font_size_small: float = 7.0
    font_size_medium: float = 8.0
    font_size_large: float = 10.0
    font_size_annotation: float = 7.0

    # --- Figure geometry (inches; Nature column widths: 89/120/183 mm) ---
    figure_width_single: float = 3.50
    figure_width_onehalf: float = 4.72
    figure_width_double: float = 7.20
    figure_height_default: float = 3.50
    figure_dpi: int = 300
    figure_dpi_lineart: int = 600

    # --- Line widths ---
    axes_linewidth: float = 0.75
    tick_length: float = 3.0
    tick_width: float = 0.5
    data_linewidth: float = 0.75

    # --- Axes appearance ---
    tick_direction: str = "in"
    remove_top_spine: bool = True
    remove_right_spine: bool = True
    background_color: str = "white"
    text_color: str = "black"
    grid: bool = False
    grid_alpha: float = 0.15
    grid_color: str = "#cccccc"

    # --- Colors (Wong 2011 colorblind-safe 8-color palette) ---
    categorical_palette: List[str] = field(
        default_factory=lambda: [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#999999",
        ]
    )
    diverging_cmap: str = "RdBu_r"
    sequential_cmap: str = "viridis"
    correlation_cmap: str = "coolwarm"
    large_categorical_palette: str = "tab20"

    # --- Legend ---
    legend_frameon: bool = False
    legend_fontsize: Optional[float] = None
    legend_markerscale: float = 1.0

    # --- Save behavior ---
    save_format: str = "pdf"
    save_transparent: bool = True
    save_bbox_inches: str = "tight"

    # --- Seaborn base style ---
    seaborn_style: str = "ticks"

    # --- Spatial scatter ---
    scatter_size_cell: float = 3.0
    data_alpha_spatial: float = 0.9

    # --- Data marks ---
    data_alpha_bar: float = 0.85
    data_alpha_box: float = 0.40
    data_alpha_background: float = 0.40
    data_linewidth_box: float = 0.75
    edge_color_default: str = "#1a1a1a"

    # --- Accent colors (directional / statistical) ---
    accent_color_positive: str = "#006BA2"
    accent_color_negative: str = "#DB444B"
    accent_color_neutral: str = "#758D99"

    # --- Network graphs ---
    network_node_size: float = 300.0
    network_node_default_color: str = "#d0d0d0"

    # --- Heatmaps ---
    heatmap_vmin: float = -3.0
    heatmap_vmax: float = 3.0
    colorbar_fraction: float = 0.03
    colorbar_pad: float = 0.05

    # --- Multi-panel layout ---
    subplot_wspace: float = 0.4
    subplot_hspace: float = 0.4

    # --- Scale bars ---
    scalebar_color: str = "white"
    scalebar_location: str = "lower right"
    scalebar_fraction: float = (
        0.04  # bar height = this × displayed image height (data units)
    )
    scalebar_label_top: bool = False  # True: scale-bar text above the bar

    # ------------------------------------------------------------------ #
    # Named sub-palettes (class-level constants, not YAML-serialized)     #
    # ------------------------------------------------------------------ #
    PALETTE_CELL_TYPES: ClassVar[List[str]] = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#999999",
    ]
    PALETTE_NEIGHBORHOODS: ClassVar[List[str]] = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]
    PALETTE_ACCENT: ClassVar[Dict[str, str]] = {
        "positive": "#006BA2",
        "negative": "#DB444B",
        "neutral": "#758D99",
        "highlight": "#F0E442",
    }

    # ------------------------------------------------------------------ #
    # Factory presets                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def publication(cls) -> "SPACEcStyle":
        """Nature-publication-ready defaults (all field defaults)."""
        return cls()

    @classmethod
    def presentation(cls) -> "SPACEcStyle":
        """Larger fonts and lines for slide/poster presentations."""
        return cls(
            font_size_tiny=12.0,
            font_size_small=14.0,
            font_size_medium=16.0,
            font_size_large=18.0,
            font_size_annotation=12.0,
            figure_width_single=6.0,
            figure_width_onehalf=8.0,
            figure_width_double=12.0,
            axes_linewidth=1.5,
            data_linewidth=2.0,
            tick_length=5.0,
            tick_width=1.0,
            legend_frameon=True,
            figure_dpi=150,
        )

    @classmethod
    def default(cls) -> "SPACEcStyle":
        """Legacy spacec defaults (matches the old hardcoded 14/16/18 pt values)."""
        return cls(
            font_size_tiny=14.0,
            font_size_small=14.0,
            font_size_medium=16.0,
            font_size_large=18.0,
            font_size_annotation=14.0,
            figure_dpi=300,
            seaborn_style="ticks",
        )

    @classmethod
    def tissue(cls) -> "SPACEcStyle":
        """Dark-background preset for tissue/spatial overlay plots.

        Large dots, high opacity, dark canvas — optimised for microscopy images
        and spatial scatter maps displayed as figures or on screen.
        """
        return cls(
            background_color="#111111",
            text_color="#eeeeee",
            scatter_size_cell=6.0,
            data_alpha_spatial=0.85,
            remove_top_spine=True,
            remove_right_spine=True,
            grid=False,
            legend_frameon=False,
            font_size_tiny=6.0,
            font_size_small=7.0,
            font_size_medium=8.0,
            seaborn_style="dark",
            scalebar_color="#eeeeee",
        )

    # ------------------------------------------------------------------ #
    # rcParams                                                              #
    # ------------------------------------------------------------------ #

    def to_rcparams(self) -> dict:
        """Return a dict suitable for ``mpl.rcParams.update()``."""
        legend_fs = (
            self.legend_fontsize
            if self.legend_fontsize is not None
            else self.font_size_small
        )
        return {
            "font.family": self.font_family,
            "font.size": self.font_size_small,
            "axes.titlesize": self.font_size_medium,
            "axes.labelsize": self.font_size_small,
            "xtick.labelsize": self.font_size_tiny,
            "ytick.labelsize": self.font_size_tiny,
            "xtick.direction": self.tick_direction,
            "ytick.direction": self.tick_direction,
            "xtick.major.size": self.tick_length,
            "ytick.major.size": self.tick_length,
            "xtick.major.width": self.tick_width,
            "ytick.major.width": self.tick_width,
            "axes.linewidth": self.axes_linewidth,
            "lines.linewidth": self.data_linewidth,
            "figure.dpi": self.figure_dpi,
            "savefig.dpi": self.figure_dpi,
            "figure.facecolor": self.background_color,
            "axes.facecolor": self.background_color,
            "text.color": self.text_color,
            "axes.labelcolor": self.text_color,
            "axes.titlecolor": self.text_color,
            "xtick.color": self.text_color,
            "ytick.color": self.text_color,
            "axes.grid": self.grid,
            "grid.alpha": self.grid_alpha,
            "grid.color": self.grid_color,
            "legend.frameon": self.legend_frameon,
            "legend.fontsize": legend_fs,
            "legend.markerscale": self.legend_markerscale,
            "figure.titlesize": self.font_size_large,
            "lines.markersize": self.scatter_size_cell,
            "patch.edgecolor": self.edge_color_default,
            "patch.linewidth": self.data_linewidth_box,
            "image.cmap": self.sequential_cmap,
        }

    # ------------------------------------------------------------------ #
    # Context manager                                                       #
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def context(self):
        """Apply this style for the duration of the ``with`` block."""
        token = _active_style.set(self)
        with mpl.rc_context():
            sns.set_style(self.seaborn_style)
            mpl.rcParams.update(self.to_rcparams())
            try:
                yield self
            finally:
                _active_style.reset(token)

    # ------------------------------------------------------------------ #
    # Serialization                                                         #
    # ------------------------------------------------------------------ #

    def to_yaml(self, path: Union[str, pathlib.Path]) -> None:
        """Write this style to a YAML file the user can edit and reload."""
        with open(path, "w") as f:
            f.write(self.to_yaml_string())

    def to_yaml_string(self) -> str:
        """Return the style as an annotated YAML string (without writing to disk)."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for to_yaml_string(). Run: pip install pyyaml"
            ) from exc
        header = (
            "# SPACEc Style Configuration\n"
            '# Generated by SPACEc — edit freely, reload with pl.set_style("<path>")\n'
            "#\n"
        )
        body = yaml.dump(asdict(self), default_flow_style=False, sort_keys=True)
        return header + "\n" + body

    @classmethod
    def from_yaml(cls, path: Union[str, pathlib.Path]) -> "SPACEcStyle":
        """Load a style from a YAML file."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for from_yaml(). Run: pip install pyyaml"
            ) from exc
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            **{k: v for k, v in data.items() if k in SPACEcStyle.__dataclass_fields__}
        )

    def to_json(self, path: Union[str, pathlib.Path]) -> None:
        """Write this style to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, pathlib.Path]) -> "SPACEcStyle":
        """Load a style from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            **{k: v for k, v in data.items() if k in SPACEcStyle.__dataclass_fields__}
        )

    @classmethod
    def from_dict(cls, d: dict) -> "SPACEcStyle":
        """Construct from an arbitrary dict (e.g. a widget state dict)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ------------------------------------------------------------------ #
# Register bundled .mplstyle files so users can call                    #
# plt.style.use("spacec_publication") / "spacec_presentation" etc.     #
# ------------------------------------------------------------------ #


def _register_mplstyles() -> None:
    """Register spacec .mplstyle files with matplotlib's style library."""
    try:
        import matplotlib.style.core as _msc

        _styles_dir = pathlib.Path(__file__).parent / "styles"
        if _styles_dir.is_dir():
            _new = _msc.read_style_directory(str(_styles_dir))
            _msc.library.update(_new)
            # Same refresh matplotlib's reload_library() does; without it the
            # styles work in style.use() but are missing from style.available.
            _msc.available[:] = sorted(_msc.library.keys())
    except Exception:
        pass  # Never hard-fail at import time


_register_mplstyles()


# ------------------------------------------------------------------ #
# Module-level helpers                                                  #
# ------------------------------------------------------------------ #


def get_active_style() -> SPACEcStyle:
    """Return the active style (context > module default > publication)."""
    active = _active_style.get()
    if active is not None:
        return active
    if _default_style is not None:
        return _default_style
    return SPACEcStyle.publication()


def set_style(s: Union[SPACEcStyle, str, pathlib.Path]) -> None:
    """Set the module-level default style. Persists until changed.

    Parameters
    ----------
    s:
        A ``SPACEcStyle`` instance, or a path to a ``.yaml`` / ``.json`` file.
    """
    global _default_style
    if isinstance(s, SPACEcStyle):
        _default_style = s
    else:
        p = pathlib.Path(s)
        if p.suffix in (".yaml", ".yml"):
            _default_style = SPACEcStyle.from_yaml(p)
        elif p.suffix == ".json":
            _default_style = SPACEcStyle.from_json(p)
        else:
            raise ValueError(
                f"Unsupported style file format: {p.suffix}. Use .yaml or .json."
            )
    sns.set_style(_default_style.seaborn_style)
    mpl.rcParams.update(_default_style.to_rcparams())


@contextlib.contextmanager
def style(s: Union[SPACEcStyle, str, pathlib.Path]):
    """Context manager: apply *s* as the active style for the block.

    Parameters
    ----------
    s:
        A ``SPACEcStyle`` instance, or a path to a ``.yaml`` / ``.json`` file.
    """
    if not isinstance(s, SPACEcStyle):
        p = pathlib.Path(s)
        if p.suffix in (".yaml", ".yml"):
            s = SPACEcStyle.from_yaml(p)
        else:
            s = SPACEcStyle.from_json(p)
    with s.context():
        yield s


# ------------------------------------------------------------------ #
# Internal helpers used by plotting functions                           #
# ------------------------------------------------------------------ #


def _apply_style(
    style_arg: Optional[Union[SPACEcStyle, str, pathlib.Path]],
) -> SPACEcStyle:
    """Resolve the style to use for a single function call."""
    if style_arg is None:
        return get_active_style()
    if isinstance(style_arg, SPACEcStyle):
        return style_arg
    p = pathlib.Path(str(style_arg))
    if p.suffix in (".yaml", ".yml"):
        return SPACEcStyle.from_yaml(p)
    return SPACEcStyle.from_json(p)


def _savefig(
    fig_or_none,
    output_dir: Union[str, pathlib.Path, None],
    output_fname: str,
    suffix: str,
    s: SPACEcStyle,
    dpi: Optional[int] = None,
) -> None:
    """Save *fig_or_none* using style settings.

    Parameters
    ----------
    fig_or_none:
        A ``matplotlib.figure.Figure``, a seaborn ``ClusterGrid`` / ``FacetGrid``
        (any object with a ``.figure`` attribute), or ``None`` to call
        ``plt.savefig``.
    output_dir:
        Directory to save into. If ``None``, nothing is saved.
    output_fname:
        Base filename without extension.
    suffix:
        String appended after *output_fname* (e.g. ``"_stacked_barplot"``).
    s:
        The active ``SPACEcStyle``.
    dpi:
        Override DPI. Defaults to ``s.figure_dpi``.
    """
    if output_dir is None:
        return
    save_dpi = dpi if dpi is not None else s.figure_dpi
    out_path = pathlib.Path(output_dir) / f"{output_fname}{suffix}.{s.save_format}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = getattr(fig_or_none, "figure", fig_or_none)
    kwargs = dict(
        dpi=save_dpi, transparent=s.save_transparent, bbox_inches=s.save_bbox_inches
    )
    if fig is not None:
        fig.savefig(out_path, **kwargs)
    else:
        plt.savefig(out_path, **kwargs)


def _apply_spines(ax, s: SPACEcStyle) -> None:
    """Remove top/right spines according to *s*."""
    sns.despine(ax=ax, top=s.remove_top_spine, right=s.remove_right_spine)


def introduce_line_breaks(
    text,
    max_width_px,
    renderer,
    fontsize,
    fontfamily="sans-serif",
    fontweight="normal",
):
    """Word-wrap *text* so no line's rendered width exceeds *max_width_px*.

    Measures each candidate line's actual width via *renderer* at the given
    font size/family/weight, so wrapping is conditional on the real
    panel-width-vs-text-length relationship for the active style -- not a
    fixed character count -- and only activates when a title/label would
    otherwise overflow its panel. A title that already fits (e.g. a smaller
    publication font, or a wider panel) is returned unchanged. Existing
    newlines in *text* are treated as fixed breaks; each resulting segment is
    wrapped independently.

    Parameters
    ----------
    text : str
        The text to wrap (may already contain ``\\n``).
    max_width_px : float
        Maximum allowed line width, in display pixels (e.g.
        ``ax.get_window_extent(renderer).width`` times a safety margin).
    renderer : matplotlib RendererBase
        From ``fig.canvas.get_renderer()`` after at least one draw, so axes
        sizes are final.
    fontsize : float
        Font size (points) to measure and wrap at -- pass the same size the
        text will actually be drawn with.
    fontfamily : str, optional
        Font family to measure with. Default ``"sans-serif"``.
    fontweight : str, optional
        Font weight to measure with. Default ``"normal"``.

    Returns
    -------
    str
        *text* with line breaks inserted where needed (unchanged if it
        already fits).
    """
    from matplotlib.font_manager import FontProperties

    prop = FontProperties(family=fontfamily, weight=fontweight, size=fontsize)

    def _width_px(s):
        w, _, _ = renderer.get_text_width_height_descent(s, prop, ismath=False)
        return w

    out_paragraphs = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        lines = []
        cur = words[0]
        for word in words[1:]:
            candidate = f"{cur} {word}"
            if _width_px(candidate) <= max_width_px:
                cur = candidate
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
        out_paragraphs.append("\n".join(lines))
    return "\n".join(out_paragraphs)


def _get_palette(
    s: SPACEcStyle,
    n: int,
    adata=None,
    col: Optional[str] = None,
) -> list:
    """Resolve a categorical color palette of length *n*.

    Priority: ``adata.uns[col + '_colors']`` → ``s.categorical_palette`` →
    ``s.large_categorical_palette`` (seaborn name).
    """
    if adata is not None and col is not None:
        uns_key = col + "_colors"
        if uns_key in adata.uns:
            stored = list(adata.uns[uns_key])
            if len(stored) >= n:
                return stored[:n]

    base = list(s.categorical_palette)
    if n <= len(base):
        return base[:n]
    return list(sns.color_palette(s.large_categorical_palette, n))


def _resolve_categorical_palette(
    s: SPACEcStyle,
    adata,
    col: Optional[str],
    categories,
    palette=None,
    missing_color: str = "grey",
) -> dict:
    """Resolve a ``{category: color}`` dict for *col*, consistent across every function
    call for the same ``adata``/*col* -- regardless of what internal display order any
    one caller happens to use (e.g. a dendrogram leaf order, vs. plain sorted order).

    *categories* is the set of values THIS call needs a color for (any order -- only
    membership matters). Three cases:

    - ``palette`` is a ``dict``: used for every category it covers. Anything in
      *categories* missing from it is colored *missing_color* (default ``"grey"``),
      and a single ``UserWarning`` lists exactly which categories were uncovered --
      an explicit palette is treated as a deliberate, best-practice override, not
      required to be exhaustive. Not written to ``adata.uns`` (a local override
      shouldn't redefine the round's shared colors for every other caller).
    - ``palette`` is a ``str``: a matplotlib colormap name, sampled evenly across
      ``sorted(str(c) for c in categories)``.
    - ``palette`` is ``None``: if *adata* and *col* are both given, resolves a persistent
      mapping for *every* value ``adata.obs[col]`` actually has (not just *categories* --
      so a later, broader call doesn't fall outside an earlier, narrower call's cache),
      keyed to ``sorted()`` order to match the convention scanpy itself uses for
      ``adata.uns[col + "_colors"]`` (e.g. after ``sc.pl.umap``): reuse it if already
      present and long enough, else generate fresh via :func:`_get_palette` and store
      it back -- so whichever function is called first in a "round" (annotation,
      dotplot, spatial plot, ...) establishes the round's colors, and every later
      call, in any function, in any internal order, reuses exactly those colors by
      name instead of silently regenerating a differently-ordered palette. If *adata*
      is ``None`` (a function with no AnnData to cache into, e.g. one that plots a
      plain DataFrame), colors *categories* directly with no caching.

    *adata*/*col* may be ``None`` together (no caching possible); *categories* is
    then colored directly, in ``sorted()`` order.
    """
    if isinstance(palette, dict):
        missing = sorted({str(c) for c in categories if c not in palette})
        if missing:
            warnings.warn(
                f"palette has no color for {missing} in '{col}' -- "
                f"colored {missing_color!r}. Pass a complete dict, or leave "
                "palette=None to auto-generate/reuse one for every value.",
                UserWarning,
                stacklevel=3,
            )
        return {c: palette.get(c, missing_color) for c in categories}

    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        str_categories = sorted(str(c) for c in categories)
        colors = cmap(
            [i / max(len(str_categories) - 1, 1) for i in range(len(str_categories))]
        )
        return dict(zip(str_categories, colors))

    if adata is None or col is None:
        full_categories = sorted(categories)
        return dict(zip(full_categories, _get_palette(s, len(full_categories))))

    full_categories = sorted(adata.obs[col].dropna().unique().tolist())
    uns_key = col + "_colors"
    stored = list(adata.uns[uns_key]) if uns_key in adata.uns else []
    if len(stored) < len(full_categories):
        stored = _get_palette(s, len(full_categories))
        adata.uns[uns_key] = stored
    return dict(zip(full_categories, stored))


def _style_legend(ax, s: SPACEcStyle, **kws) -> None:
    """Apply standard legend formatting from *s* to the axes legend.

    Silently does nothing if the axes has no legend yet.
    """
    leg = ax.get_legend()
    if leg is None:
        return
    leg.set_frame_on(s.legend_frameon)
    fs = s.legend_fontsize if s.legend_fontsize is not None else s.font_size_small
    for text in leg.get_texts():
        text.set_fontsize(fs)
    if leg.get_title() and leg.get_title().get_text():
        leg.get_title().set_fontsize(s.font_size_small)


def _style_colorbar(cbar, s: SPACEcStyle) -> None:
    """Apply standard colorbar formatting from *s*.

    Parameters
    ----------
    cbar:
        A ``matplotlib.colorbar.Colorbar`` instance.
    s:
        The active ``SPACEcStyle``.
    """
    cbar.ax.tick_params(labelsize=s.font_size_tiny)
    label = cbar.ax.get_ylabel() or cbar.ax.get_xlabel()
    if label:
        cbar.set_label(label, fontsize=s.font_size_small)


def _nice_scalebar_length(image_width_px: int) -> int:
    """Return a round pixel length ~20% of image width (from the 1,2,5,10,20,50,... sequence)."""
    target = image_width_px * 0.20
    magnitude = 1
    while True:
        for i, factor in enumerate((1, 2, 5)):
            candidate = factor * magnitude
            if candidate > target:
                if i == 0:
                    magnitude //= 10
                    return 5 * magnitude
                return (1, 2, 5)[i - 1] * magnitude
        magnitude *= 10


def _add_scalebar(
    ax,
    image_width_px: int,
    s: SPACEcStyle,
    scalebar_length_px=None,
    um_per_px=None,
    show_text: bool = True,
    color_override=None,
) -> None:
    """Add a scale bar to an image axes.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes on which imshow was called.
    image_width_px : int
        Width of the displayed image in pixels (array shape[1]).
    s : SPACEcStyle
        Active style (provides color, location, fraction, font settings).
    scalebar_length_px : int, optional
        Explicit bar length in pixels. Auto-picked (~20% of width) if None.
    um_per_px : float, optional
        Conversion factor (micrometres per pixel). If provided, label shows µm.
    show_text : bool
        Whether to show the text label. Default True.
    color_override : str, optional
        Override s.scalebar_color for this call.
    """
    import matplotlib.font_manager as fm
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    color = color_override if color_override is not None else s.scalebar_color
    bar_px = (
        scalebar_length_px
        if scalebar_length_px is not None
        else _nice_scalebar_length(image_width_px)
    )

    if show_text:
        if um_per_px is not None:
            um_value = bar_px * um_per_px
            label = f"{round(um_value, 1):g} µm"  # 200.00000001 -> "200 µm", 12.5 -> "12.5 µm"
        else:
            label = f"{bar_px} px"
    else:
        label = ""

    y_min, y_max = ax.get_ylim()
    size_vertical = abs(y_max - y_min) * s.scalebar_fraction

    fontprops = fm.FontProperties(size=s.font_size_annotation, family=s.font_family)
    scalebar = AnchoredSizeBar(
        ax.transData,
        size=bar_px,
        label=label,
        loc=s.scalebar_location,
        pad=0.3,
        borderpad=0.5,
        sep=3,
        frameon=False,
        size_vertical=size_vertical,
        color=color,
        fontproperties=fontprops,
        label_top=s.scalebar_label_top,
    )
    ax.add_artist(scalebar)


def preview_style(style=None):
    """Render a 2×2 preview figure demonstrating the active style.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered preview figure. Call ``plt.show()`` or ``display(fig)``
        in a notebook to display it.

    Parameters
    ----------
    style:
        A ``SPACEcStyle`` instance, path to a ``.yaml`` / ``.json`` file,
        or ``None`` to use the active/default style.

    Examples
    --------
    >>> import spacec.plotting as pl
    >>> fig = pl.preview_style(pl.SPACEcStyle.presentation())
    >>> fig = pl.preview_style("my_style.yaml")
    """
    import numpy as np

    _s = _apply_style(style)

    rng = np.random.default_rng(42)
    _bar_cats = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    _bar_vals = [0.72, 0.45, 0.88, 0.33, 0.61]
    _bar_err = [0.06, 0.04, 0.07, 0.05, 0.06]
    _sc_x = rng.standard_normal(80)
    _sc_y = _sc_x * 0.75 + rng.standard_normal(80) * 0.6
    _sc_groups = rng.integers(0, 4, size=80)
    _hm_data = rng.standard_normal((10, 10))

    def _draw_bar(ax):
        pal = _s.categorical_palette
        colors = [pal[i % len(pal)] for i in range(len(_bar_cats))]
        ax.bar(
            _bar_cats,
            _bar_vals,
            color=colors,
            alpha=_s.data_alpha_bar,
            edgecolor=_s.edge_color_default,
            linewidth=_s.data_linewidth_box,
            yerr=_bar_err,
            error_kw=dict(
                elinewidth=_s.data_linewidth, capsize=2, ecolor=_s.edge_color_default
            ),
            zorder=3,
        )
        ax.axhline(
            0.5,
            color=_s.accent_color_neutral,
            linewidth=_s.data_linewidth,
            linestyle="--",
            zorder=2,
        )
        ax.set_xlabel(
            "Condition", fontsize=_s.font_size_small, fontfamily=_s.font_family
        )
        ax.set_ylabel(
            "Normalised score", fontsize=_s.font_size_small, fontfamily=_s.font_family
        )
        ax.set_title(
            "Bar chart", fontsize=_s.font_size_medium, fontfamily=_s.font_family
        )
        ax.annotate(
            "ref",
            xy=(4.5, 0.51),
            fontsize=_s.font_size_annotation,
            color=_s.accent_color_neutral,
            fontfamily=_s.font_family,
        )
        _apply_spines(ax, _s)

    def _draw_scatter(ax):
        pal = _s.categorical_palette
        for g in range(4):
            mask = _sc_groups == g
            ax.scatter(
                _sc_x[mask],
                _sc_y[mask],
                s=_s.scatter_size_cell * 8,
                alpha=_s.data_alpha_spatial,
                color=pal[g % len(pal)],
                edgecolors=_s.edge_color_default,
                linewidths=_s.data_linewidth * 0.5,
                label=f"Cluster {g + 1}",
                zorder=3,
            )
        m, b = np.polyfit(_sc_x, _sc_y, 1)
        xr = np.linspace(_sc_x.min(), _sc_x.max(), 100)
        ax.plot(
            xr,
            m * xr + b,
            color=_s.accent_color_positive,
            linewidth=_s.data_linewidth,
            zorder=2,
        )
        ax.set_xlabel("PC 1", fontsize=_s.font_size_small, fontfamily=_s.font_family)
        ax.set_ylabel("PC 2", fontsize=_s.font_size_small, fontfamily=_s.font_family)
        ax.set_title(
            "Scatter / UMAP", fontsize=_s.font_size_medium, fontfamily=_s.font_family
        )
        ax.legend(
            fontsize=_s.font_size_small,
            frameon=_s.legend_frameon,
            markerscale=_s.legend_markerscale,
            handlelength=0.8,
        )
        _style_legend(ax, _s)
        _apply_spines(ax, _s)

    def _draw_heatmap(ax):
        im = ax.imshow(
            _hm_data,
            cmap=_s.diverging_cmap,
            vmin=_s.heatmap_vmin,
            vmax=_s.heatmap_vmax,
            aspect="auto",
        )
        ax.set_title("Heatmap", fontsize=_s.font_size_medium, fontfamily=_s.font_family)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = ax.figure.colorbar(
            im, ax=ax, fraction=_s.colorbar_fraction, pad=_s.colorbar_pad
        )
        cbar.ax.tick_params(labelsize=_s.font_size_tiny)
        cbar.set_label(r"$\log_2\mathrm{FC}$", fontsize=_s.font_size_small)

    def _draw_network(ax):
        try:
            import matplotlib.patches as mpatches
            import networkx as nx
        except ImportError:
            ax.text(
                0.5,
                0.5,
                "networkx not installed\n(pip install networkx)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=_s.font_size_small,
                fontfamily=_s.font_family,
            )
            ax.set_title(
                "Network graph", fontsize=_s.font_size_medium, fontfamily=_s.font_family
            )
            ax.axis("off")
            return
        g = nx.karate_club_graph()
        g = nx.subgraph(g, list(g.nodes)[:14]).copy()
        pos = nx.spring_layout(g, seed=7)
        degs = dict(g.degree())
        pal = _s.categorical_palette
        communities = list(nx.community.greedy_modularity_communities(g))
        node_colors = {
            n: pal[i % len(pal)] for i, comm in enumerate(communities) for n in comm
        }
        node_size_list = [degs[n] * _s.network_node_size * 0.25 for n in g.nodes()]
        color_list = [
            node_colors.get(n, _s.network_node_default_color) for n in g.nodes()
        ]
        nx.draw_networkx(
            g,
            pos=pos,
            ax=ax,
            node_size=node_size_list,
            node_color=color_list,
            edge_color=_s.accent_color_neutral,
            linewidths=_s.data_linewidth_box,
            edgecolors=_s.edge_color_default,
            with_labels=False,
            width=_s.data_linewidth,
            alpha=0.95,
        )
        ax.set_title(
            "Network graph", fontsize=_s.font_size_medium, fontfamily=_s.font_family
        )
        ax.set_facecolor(_s.background_color)
        ax.axis("off")
        patches = [
            mpatches.Patch(color=pal[i % len(pal)], label=f"Module {i + 1}")
            for i in range(len(communities))
        ]
        ax.legend(
            handles=patches,
            fontsize=_s.font_size_small,
            frameon=_s.legend_frameon,
            loc="lower left",
        )

    with _s.context():
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(_s.figure_width_double, _s.figure_width_double * 0.85),
            facecolor=_s.background_color,
        )
        fig.subplots_adjust(wspace=_s.subplot_wspace, hspace=_s.subplot_hspace)
        _draw_bar(axes[0, 0])
        _draw_scatter(axes[0, 1])
        _draw_heatmap(axes[1, 0])
        _draw_network(axes[1, 1])

    return fig
