"""Interactive spatial cell-type highlighting widget for SPACEc.

Highlights selected cell types in a spatial scatter plot while rendering all
other cells in grey — useful for understanding the spatial distribution of
individual or grouped cell types within a tissue region.

Public entry point::

    sp.pl.spatial_cell_type_widget(adata)

The work is done by :class:`_SpatialCellTypeWidget`; the public function
validates input, builds the instance, displays the UI, and returns the instance
for headless / programmatic access (e.g. in unit tests).
"""

from __future__ import annotations

import ipywidgets as widgets
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from ._style import get_active_style


def _build_color_map(cell_types: list[str], palette: str) -> dict[str, tuple]:
    """Assign a stable color to every cell type from *palette*.

    Uses the full list so that colors do not shift when the user de-selects
    some types — only the selected ones get painted, but the mapping is fixed.
    """
    cmap = plt.colormaps[palette]
    n = len(cell_types)
    if hasattr(cmap, "colors"):  # ListedColormap (tab10, tab20, …)
        base_colors = list(cmap.colors)
    else:  # continuous colormap — sample evenly
        base_colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    return {
        ct: base_colors[i % len(base_colors)] for i, ct in enumerate(sorted(cell_types))
    }


class _SpatialCellTypeWidget:
    """State + widgets + render logic for the spatial cell-type widget.

    Callback methods double as headless entry points exercised by the unit
    tests (no live Jupyter frontend required).
    """

    def __init__(
        self,
        adata,
        cell_type_col: str,
        region_col: str,
        x_col: str,
        y_col: str,
        palette: str,
        point_size: float,
        grey_alpha: float,
        color_alpha: float,
        figsize: tuple,
    ):
        self.adata = adata
        self.cell_type_col = cell_type_col
        self.region_col = region_col
        self.x_col = x_col
        self.y_col = y_col
        self.palette = palette
        self.point_size = point_size
        self.grey_alpha = grey_alpha
        self.color_alpha = color_alpha
        self.figsize = figsize

        self._regions: list[str] = sorted(adata.obs[region_col].astype(str).unique())
        self._color_map: dict[str, tuple] = {}
        self._updating: bool = False  # guard against duplicate renders

        self._build_widgets()
        self._link_callbacks()
        # Populate options only — do NOT render yet.  The first render happens
        # inside display(), after the widget tree has been sent to the frontend,
        # so the figure goes into the Output widget and not into the raw cell
        # output stream.
        self._populate_region(self._regions[0] if self._regions else None)

    # ------------------------------------------------------------------ #
    # Widget construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_widgets(self):
        L = widgets.Layout

        self.status = widgets.HTML(value="")

        self.region_dd = widgets.Dropdown(
            options=self._regions,
            value=self._regions[0] if self._regions else None,
            description="",
            layout=L(width="240px"),
        )

        self.cell_type_sm = widgets.SelectMultiple(
            options=[],
            value=[],
            rows=12,
            description="",
            layout=L(width="210px"),
        )

        self.select_all_btn = widgets.Button(
            description="Select All",
            button_style="",
            icon="check",
            layout=L(width="100px"),
        )
        self.clear_btn = widgets.Button(
            description="Clear",
            button_style="",
            icon="times",
            layout=L(width="75px"),
        )

        self.size_slider = widgets.FloatSlider(
            value=self.point_size,
            min=0.5,
            max=50.0,
            step=0.5,
            description="",
            continuous_update=False,
            readout_format=".1f",
            layout=L(width="210px"),
        )
        self.alpha_slider = widgets.FloatSlider(
            value=self.color_alpha,
            min=0.05,
            max=1.0,
            step=0.05,
            description="",
            continuous_update=False,
            readout_format=".2f",
            layout=L(width="210px"),
        )

        self.plot_out = widgets.Output(
            layout=L(border="1px solid #ddd", min_height="420px", flex="1")
        )

    def _link_callbacks(self):
        self.region_dd.observe(self._on_region_change, names="value")
        self.cell_type_sm.observe(self._on_selection_change, names="value")
        self.size_slider.observe(self._on_selection_change, names="value")
        self.alpha_slider.observe(self._on_selection_change, names="value")
        self.select_all_btn.on_click(self._select_all)
        self.clear_btn.on_click(self._clear_selection)

    # ------------------------------------------------------------------ #
    # Layout                                                               #
    # ------------------------------------------------------------------ #

    def _ui(self) -> widgets.Widget:
        """Assemble and return the full widget layout."""
        L = widgets.Layout

        def row(label: str, *ws):
            """
            Build a labelled horizontal row of widgets.

            Parameters
            ----------
            label : str
                Text to display in a fixed-width label on the left.
            *ws : widgets.Widget
                One or more ipywidgets to place to the right of the label.

            Returns
            -------
            widgets.HBox
                Horizontal box containing the label followed by ``ws``.
            """
            return widgets.HBox([widgets.Label(label, layout=L(width="80px")), *ws])

        controls = widgets.VBox(
            [
                widgets.HTML("<b>Cell Type Explorer</b>"),
                row("Region:", self.region_dd),
                widgets.HTML("<hr style='margin:4px 0'>"),
                row(
                    "Highlight:",
                    widgets.VBox(
                        [
                            self.cell_type_sm,
                            widgets.HBox([self.select_all_btn, self.clear_btn]),
                        ]
                    ),
                ),
                widgets.HTML("<hr style='margin:4px 0'>"),
                row("Pt size:", self.size_slider),
                row("Alpha:", self.alpha_slider),
                self.status,
            ],
            layout=L(width="310px", min_width="310px", padding="6px"),
        )
        return widgets.HBox(
            [controls, self.plot_out],
            layout=L(width="100%", align_items="flex-start"),
        )

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #

    def _populate_region(self, region):
        """Update color map and cell-type options for *region* without rendering."""
        if region is None:
            return
        obs = self.adata.obs
        mask = obs[self.region_col].astype(str) == str(region)
        types_in_region: list[str] = sorted(
            obs.loc[mask, self.cell_type_col].astype(str).unique()
        )
        self._color_map = _build_color_map(types_in_region, self.palette)

        prev = set(self.cell_type_sm.value)
        # Block the selection observer while batch-updating options + value so
        # the render is triggered exactly once (explicitly, at call-site).
        self._updating = True
        try:
            self.cell_type_sm.options = types_in_region
            still_valid = [t for t in types_in_region if t in prev]
            self.cell_type_sm.value = still_valid if still_valid else types_in_region
        finally:
            self._updating = False

    def _on_region_change(self, change):
        self._populate_region(change["new"])
        self._render_plot()

    def _on_selection_change(self, _=None):
        if self._updating:
            return
        self._render_plot()

    def _select_all(self, _=None):
        self.cell_type_sm.value = list(self.cell_type_sm.options)

    def _clear_selection(self, _=None):
        self.cell_type_sm.value = ()

    # ------------------------------------------------------------------ #
    # Rendering                                                            #
    # ------------------------------------------------------------------ #

    def _render_plot(self):
        """Redraw the scatter plot into the persistent Output widget.

        The figure is built and closed *outside* the Output context so that
        the Jupyter inline backend never adds it to its auto-display queue.
        Only the already-closed figure object is then passed into the Output
        widget via ``display(fig)``, matching the pattern used by the
        clustering GUI's ``_show`` helper.
        """
        region = self.region_dd.value
        selected = list(self.cell_type_sm.value)
        size = self.size_slider.value
        alpha = self.alpha_slider.value

        obs = self.adata.obs
        mask = obs[self.region_col].astype(str) == str(region)
        df = obs[mask]

        # ── Build figure (outside Output context) ──────────────────────── #
        with get_active_style().context():
            fig, ax = plt.subplots(figsize=self.figsize)

            # Grey background — all cells in the region.
            ax.scatter(
                df[self.x_col],
                df[self.y_col],
                c="#CCCCCC",
                s=size,
                alpha=self.grey_alpha,
                linewidths=0,
                rasterized=True,
            )

            # Colored foreground — one scatter call per selected cell type.
            legend_handles = []
            for ct in sorted(selected):
                ct_mask = df[self.cell_type_col].astype(str) == ct
                color = self._color_map.get(ct, "#333333")
                ax.scatter(
                    df.loc[ct_mask, self.x_col],
                    df.loc[ct_mask, self.y_col],
                    c=[color],
                    s=size,
                    alpha=alpha,
                    linewidths=0,
                    rasterized=True,
                )
                legend_handles.append(mpatches.Patch(color=color, label=ct))

            ax.set_title(str(region))
            ax.set_xlabel(self.x_col)
            ax.set_ylabel(self.y_col)
            ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="datalim")

            if legend_handles:
                ax.legend(
                    handles=legend_handles,
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    borderaxespad=0,
                    frameon=False,
                )
            elif not selected:
                ax.text(
                    0.5,
                    0.5,
                    "No cell types selected — all cells shown in grey",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#888888",
                )

            plt.tight_layout()

        # Remove from pyplot's active-figure list so the inline backend will
        # not auto-display it — the explicit display() call below is the only
        # path that should render the figure.
        plt.close(fig)

        # ── Push closed figure into the Output widget ───────────────────── #
        with self.plot_out:
            clear_output(wait=True)
            display(fig)

    def _ipython_display_(self, **kwargs):
        """Render the widget in the current Jupyter cell.

        Called automatically by Jupyter when it displays the return value of a
        cell.  The widget tree is sent to the frontend first; the initial plot
        is drawn afterwards so the figure goes into the Output widget rather
        than into the raw cell output stream.
        """
        display(self._ui())
        self._render_plot()

    def display(self):
        """Public alias for headless / programmatic use."""
        self._ipython_display_()


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #


def spatial_cell_type_widget(
    adata,
    cell_type_col: str = "cell_type",
    region_col: str = "unique_region",
    x_col: str = "x",
    y_col: str = "y",
    palette: str = "tab20",
    point_size: float = 5.0,
    grey_alpha: float = 0.2,
    color_alpha: float = 0.8,
    figsize: tuple = (7, 7),
) -> _SpatialCellTypeWidget:
    """Launch an interactive spatial cell-type highlighting widget.

    Displays a scatter plot of cells in a selected tissue region.  Selected cell
    types are rendered in distinct colors; all others appear in grey.  Controls
    update the plot immediately on change.

    Parameters
    ----------
    adata:
        Annotated data matrix with spatial coordinates and cell type labels in
        ``adata.obs``.
    cell_type_col:
        ``adata.obs`` column containing cell type labels.
    region_col:
        ``adata.obs`` column containing region / sample identifiers.
    x_col, y_col:
        ``adata.obs`` columns for spatial X and Y coordinates.
    palette:
        Matplotlib colormap name used to assign distinct colors to cell types.
        Qualitative maps (``"tab20"``, ``"tab10"``, ``"Set1"``) work best.
    point_size:
        Initial scatter point size (matplotlib ``s`` parameter).
    grey_alpha:
        Initial opacity of unselected (grey) cells.
    color_alpha:
        Initial opacity of highlighted cells.
    figsize:
        Figure size in inches ``(width, height)``.

    Returns
    -------
    _SpatialCellTypeWidget
        The widget instance — useful for headless / programmatic access in tests.

    Examples
    --------
    >>> import spacec as sp
    >>> sp.pl.spatial_cell_type_widget(adata)
    >>> # with custom defaults
    >>> sp.pl.spatial_cell_type_widget(
    ...     adata,
    ...     cell_type_col="leiden_1",
    ...     region_col="Tissue_location",
    ...     palette="tab10",
    ...     point_size=3,
    ... )
    """
    _validate(adata, cell_type_col, region_col, x_col, y_col)
    w = _SpatialCellTypeWidget(
        adata=adata,
        cell_type_col=cell_type_col,
        region_col=region_col,
        x_col=x_col,
        y_col=y_col,
        palette=palette,
        point_size=point_size,
        grey_alpha=grey_alpha,
        color_alpha=color_alpha,
        figsize=figsize,
    )
    return w


def _validate(adata, cell_type_col, region_col, x_col, y_col):
    missing = [
        c for c in (cell_type_col, region_col, x_col, y_col) if c not in adata.obs
    ]
    if missing:
        raise ValueError(f"The following columns are missing from adata.obs: {missing}")
    if adata.obs[region_col].nunique() == 0:
        raise ValueError(f"No regions found in adata.obs['{region_col}'].")
