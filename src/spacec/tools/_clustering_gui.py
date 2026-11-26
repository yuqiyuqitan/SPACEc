"""Interactive clustering GUI for SPACEc, built on ipywidgets.

This is a pure ``ipywidgets`` reimplementation of the former Panel-based
``launch_interactive_clustering``. It removes the ``panel`` / ``watchfiles``
dependency, fixes the tab-leaking plot area (a single persistent ``Output`` is
reused instead of appending a new tab per run), adds live cluster annotation,
integrates the SPACEc plotting style system, and surfaces progress / errors
inline.

The public entry point keeps the original name and signature so existing
notebooks (e.g. ``notebooks/tutorials/04_clustering_GUI_assisted.ipynb``)
continue to work unchanged::

    sp.tl.launch_interactive_clustering(adata=adata, output_dir=output_dir)

The work is done by :class:`_ClusteringGUI`; the public function builds it,
displays the UI, and returns the instance for programmatic / headless access.
"""

from __future__ import annotations

import base64
import contextlib
import io
import os
import traceback

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from ipyfilechooser import FileChooser
from IPython.display import HTML, clear_output, display

from ..plotting import catplot, extended_dotplot
from ..plotting import umap as _umap
from ..plotting._general import (
    _get_clustering_report_markers,
    _qc_default_flags,
    _qc_flag_label_overrides,
)
from ..plotting._style import SPACEcStyle, _apply_spines, get_active_style
from ._clustering_tree import clustering_tree
from ._general import annotate_clusters, clustering, subclustering

# Order in which cached figures are offered in the plot-view selector.
_VIEWS = ["UMAP", "Dotplot", "Cluster counts", "Spatial"]
_STYLE_CHOICES = ["active default", "publication", "presentation", "tissue"]

# On-screen embedding caps, independent of the active SPACEcStyle's save/print
# quality settings (which can be much larger/higher-dpi than fits a widget panel).
_SCREEN_DPI = 100
_MAX_FIG_WIDTH_IN = 6.5
_MAX_FIG_HEIGHT_IN = 5.5

# Dotplot is left uncapped on both dimensions (see _cap_fig_size), so its native
# pixel size grows directly with marker/cluster count. At _SCREEN_DPI it was
# already larger than the panel for a typical marker/cluster count, so the panel's
# scroll/overflow -- meant for genuinely large plots -- was kicking in almost
# always instead. A lower, Dotplot-only DPI shrinks it uniformly (no change to
# relative layout, font/dot size proportions, or the extended_dotplot() library
# default used by the static calls in tutorial 04) so a typical plot mostly fits
# without scrolling, while a large one (many markers/clusters) still exceeds the
# panel and scrolls as intended.
_DOTPLOT_SCREEN_DPI = 70

_SPINNER_HTML = """
<style>
@keyframes spacec-spin {{ to {{ transform: rotate(360deg); }} }}
.spacec-spinner {{
  display:inline-block; width:14px; height:14px;
  border:2px solid #cfcfcf; border-top-color:#1976d2;
  border-radius:50%; animation:spacec-spin 0.8s linear infinite;
  vertical-align:middle; margin-right:6px;
}}
</style>
<span class="spacec-spinner"></span><b>{label}</b>
"""


def _sort_clusters(values):
    """Sort cluster labels numerically when possible, else lexicographically."""
    try:
        return sorted(values, key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(values, key=str)


def _cap_fig_size(
    fig,
    max_height_in=_MAX_FIG_HEIGHT_IN,
    max_width_in=_MAX_FIG_WIDTH_IN,
    dpi=_SCREEN_DPI,
):
    """Shrink fig (in place) to fit the persistent plot panel; never upscales.

    max_height_in/max_width_in let callers opt a view out of capping a given
    dimension (e.g. the Dotplot view, whose content genuinely grows with
    marker/cluster count — capping either dimension to a fixed box shrinks
    the whole figure without rescaling font size, dot size, or line width,
    crushing content into illegibility rather than just making it smaller.
    Dotplot opts out of both and relies on plot_out's own scroll/overflow
    instead).

    dpi lets a caller shrink the final pixel size uniformly instead -- that
    changes nothing about relative layout/font/dot-size proportions, only how
    large the rendered image comes out (see _DOTPLOT_SCREEN_DPI).
    """
    w, h = fig.get_size_inches()
    scale = min(1.0, max_width_in / w, max_height_in / h)
    if scale < 1.0:
        fig.set_size_inches(w * scale, h * scale)
    fig.set_dpi(dpi)
    return fig


def _display_fig_native_size(fig):
    """Render fig as an <img> at its true pixel size, ignoring notebook CSS.

    Jupyter's own stylesheet applies `img { max-width: 100% }` to output content, which
    silently shrinks a figure larger than its container to fit -- exactly what plot_out's
    `overflow="auto"` (see `_build_widgets`) is meant to catch, so a bare `display(fig)`
    (whose inline PNG repr is just an `<img>` tag) never actually scrolls no matter how large
    the figure gets. Overriding `max-width` restores real scrolling for a figure that's
    genuinely bigger than the panel (e.g. Dotplot with many markers/clusters, see
    `_render_dotplot_fig`) while changing nothing for a figure that already fits.

    `bbox_inches="tight"` is required, not cosmetic: scanpy's UMAP legend (drawn via
    `legend_loc="right margin"`) extends past the axes' own bounding box, so a plain
    `savefig()` silently clips the label text off the right edge of the exported PNG (only
    the color swatches survive) -- `display(fig)`'s own inline-backend PNG repr applies
    `bbox_inches="tight"` for exactly this reason, which a bare `savefig()` here doesn't do
    by default.
    """
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=fig.dpi,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    display(HTML(f'<img src="data:image/png;base64,{b64}" style="max-width:none;">'))


class _WidgetLogWriter(io.TextIOBase):
    """Writable stream that pushes each write live into an Output widget."""

    def __init__(self, out_widget):
        self._out = out_widget

    def write(self, s):
        if s:
            self._out.append_stdout(s)
        return len(s)

    def flush(self):
        pass


class _ClusteringGUI:
    """State + widgets + callbacks for the interactive clustering app.

    Built once and reused; all plotting renders into a single persistent
    ``Output`` so figures never leak across runs. Callback methods double as the
    headless entry points exercised by the unit tests.
    """

    _BUSY_BUTTONS = (
        "run_button",
        "subcluster_button",
        "spatial_button",
        "save_button",
    )

    def __init__(self, adata=None, output_dir=None, style=None):
        # ---- input validation (preserves prior behaviour) ----------------
        if adata is not None and not output_dir:
            raise ValueError(
                "Please provide an output directory to save the annotated AnnData object."
            )
        if output_dir is not None and not isinstance(output_dir, str):
            raise ValueError("output_dir must be a string.")
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # ---- state -------------------------------------------------------
        self.adata = adata
        self.output_dir = output_dir
        self.style = style  # explicit SPACEcStyle override, or None
        self.last_key = None  # active clustering key in adata.obs
        self._last_marker_list = None  # markers used for the active key's plots
        self._figs = {}  # cached matplotlib figures per view
        self._anno_rows = {}  # cluster id -> annotation Text widget
        self._updating_dotplot_selectors = False  # guard against duplicate re-renders

        self._build_widgets()
        self._link_callbacks()

        # Auto-load when an AnnData was supplied directly.
        if self.adata is not None:
            self._load_data()

    # ------------------------------------------------------------------ #
    # Widget construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_widgets(self):
        L = widgets.Layout

        # Status + busy indicator
        self.status = widgets.HTML(value="")
        self.spinner = widgets.HTML(value="")  # CSS spinner; empty string = hidden
        self.log_out = widgets.Output(
            layout=L(
                width="100%",
                max_height="140px",
                overflow="auto",
                border="1px solid #ddd",
                padding="4px",
                display="none",
            )
        )

        # --- Data loading (only shown when no adata was passed) ---
        _start_dir = (
            self.output_dir
            if self.output_dir and os.path.isdir(self.output_dir)
            else os.getcwd()
        )
        self.input_path = FileChooser(_start_dir)
        self.input_path.title = "Path to .h5ad file"
        self.input_path.filter_pattern = ["*.h5ad"]
        self.load_button = widgets.Button(
            description="Load data", button_style="primary", icon="upload"
        )

        # --- Clustering params ---
        self.method = widgets.Dropdown(
            options=["leiden", "louvain", "flowSOM", "leiden_gpu"],
            value="leiden",
            description="",
            layout=L(width="180px"),
        )
        self.resolution = widgets.FloatText(value=1.0, layout=L(width="120px"))
        self.n_neighbors = widgets.IntText(value=10, layout=L(width="120px"))
        self.seed = widgets.IntText(value=42, layout=L(width="120px"))
        self.reclustering = widgets.Checkbox(
            value=False, description="Reclustering (skip neighbors/UMAP)", indent=False
        )
        self.key_added = widgets.Text(
            value="", placeholder="auto: method_resolution", layout=L(width="220px")
        )
        self.marker_list = widgets.SelectMultiple(
            options=[], rows=8, description="", layout=L(width="260px")
        )
        self.fs_xdim = widgets.IntText(value=10, layout=L(width="90px"))
        self.fs_ydim = widgets.IntText(value=10, layout=L(width="90px"))
        self.fs_rlen = widgets.IntText(value=10, layout=L(width="90px"))
        self.run_button = widgets.Button(
            description="Run clustering", button_style="primary", icon="play"
        )

        # --- Subclustering params ---
        self.subcluster_key = widgets.Text(
            value="",
            placeholder="key to filter on (e.g. leiden_1.0)",
            layout=L(width="240px"),
        )
        self.subcluster_values = widgets.Text(
            value="",
            placeholder="clusters to split, sequentially (comma-separated)",
            tooltip=(
                "Each cluster listed here is split one at a time: the first "
                "restricts on the subcluster key above, every later one "
                "restricts on this round's new column -- NOT one pooled "
                "split across all of them."
            ),
            layout=L(width="240px"),
        )
        self.subcluster_resolution = widgets.FloatText(
            value=0.3, layout=L(width="120px")
        )
        self.subcluster_button = widgets.Button(
            description="Run subclustering", button_style="primary", icon="play"
        )

        # --- Annotation ---
        self.annotation_box = widgets.VBox([widgets.HTML("Run clustering first.")])
        self.apply_annotations_button = widgets.Button(
            description="Apply annotations", button_style="success", icon="check"
        )
        # Rendered clustering/subclustering/annotation lineage tree, refreshed
        # after each successful "Apply annotations" (see clustering_tree()).
        self.annotation_tree_out = widgets.Output(
            layout=L(
                width="100%",
                max_height="260px",
                overflow="auto",
                border="1px solid #ddd",
                padding="4px",
            )
        )

        # --- Spatial visualization ---
        self.spatial_color = widgets.Dropdown(
            options=[""]
            + (list(self.adata.obs.columns) if self.adata is not None else []),
            value="",
            description="",
            layout=L(width="240px"),
        )
        self.spatial_unique_region = widgets.Text(
            value="unique_region", layout=L(width="180px")
        )
        self.spatial_x = widgets.Text(value="x", layout=L(width="90px"))
        self.spatial_y = widgets.Text(value="y", layout=L(width="90px"))
        self.spatial_n_columns = widgets.IntText(value=2, layout=L(width="90px"))
        self.spatial_palette = widgets.Text(value="tab20", layout=L(width="120px"))
        self.spatial_figsize = widgets.FloatText(value=6, layout=L(width="90px"))
        self.spatial_size = widgets.FloatText(value=20, layout=L(width="90px"))
        self.spatial_savefig = widgets.Checkbox(
            value=False, description="Save figure", indent=False
        )
        self.spatial_output_fname = widgets.Text(
            value="", placeholder="output filename", layout=L(width="180px")
        )
        self.spatial_button = widgets.Button(
            description="Run spatial plot", button_style="primary", icon="play"
        )

        # --- Save ---
        _start_dir = (
            self.output_dir
            if self.output_dir and os.path.isdir(self.output_dir)
            else os.getcwd()
        )
        self.output_dir_widget = FileChooser(_start_dir)
        self.output_dir_widget.show_only_dirs = True
        self.output_dir_widget.title = "Output directory"
        self.save_button = widgets.Button(
            description="Save AnnData", button_style="success", icon="save"
        )

        # --- Plot area (persistent) ---
        self.style_dd = widgets.Dropdown(
            options=_STYLE_CHOICES,
            value="active default",
            description="",
            layout=L(width="160px"),
        )
        self.view_selector = widgets.Dropdown(
            options=_VIEWS, value="UMAP", description="", layout=L(width="160px")
        )
        # Global cluster highlight, visible regardless of the active view: an
        # outline on UMAP, a red bar in Cluster counts, and a highlighted
        # row in Dotplot (see extended_dotplot's highlight_cluster param).
        # The region picker is Dotplot-specific: pick one, on top of a
        # cluster, to also show that cluster's spatial distribution there
        # (see extended_dotplot's spatial_region param).
        self.dotplot_cluster_dd = widgets.Dropdown(
            options=[("None", "")], value="", description="", layout=L(width="140px")
        )
        self.dotplot_region_dd = widgets.Dropdown(
            options=[("None", "")], value="", description="", layout=L(width="140px")
        )
        # max_height + overflow_y: a tall Dotplot (height uncapped, see
        # _cap_fig_size) scrolls inside the panel instead of pushing the
        # rest of the page down.
        self.plot_out = widgets.Output(
            layout=L(
                border="1px solid #ddd",
                min_height="360px",
                max_height="70vh",
                max_width="100%",
                overflow="auto",
            )
        )

    def _link_callbacks(self):
        self.load_button.on_click(self._load_data)
        self.run_button.on_click(self._run_clustering)
        self.subcluster_button.on_click(self._run_subclustering)
        self.apply_annotations_button.on_click(self._apply_annotations)
        self.spatial_button.on_click(self._run_spatial)
        self.save_button.on_click(self._save_adata)
        self.view_selector.observe(self._on_view_change, names="value")
        self.style_dd.observe(self._on_style_change, names="value")
        self.dotplot_cluster_dd.observe(
            self._on_dotplot_selection_change, names="value"
        )
        self.dotplot_region_dd.observe(self._on_dotplot_selection_change, names="value")

    # ------------------------------------------------------------------ #
    # Layout                                                               #
    # ------------------------------------------------------------------ #

    def _ui(self):
        """Assemble and return the full widget layout."""

        def row(label, *widgets_):
            return widgets.HBox(
                [widgets.Label(label, layout=widgets.Layout(width="150px")), *widgets_]
            )

        # Clustering tab
        if self.adata is None:
            load_section = [
                widgets.HTML("<b>Load data</b>"),
                row("AnnData path", self.input_path, self.load_button),
                widgets.HTML("<hr>"),
            ]
        else:
            load_section = [
                widgets.HTML("<b>AnnData object loaded.</b>"),
                widgets.HTML("<hr>"),
            ]

        clustering_tab = widgets.VBox(
            load_section
            + [
                widgets.HTML("<b>Clustering parameters</b>"),
                row("Method", self.method),
                row("Resolution", self.resolution),
                row("N neighbors", self.n_neighbors),
                row("Seed", self.seed),
                row("", self.reclustering),
                row("Key added", self.key_added),
                row("FlowSOM xdim/ydim/rlen", self.fs_xdim, self.fs_ydim, self.fs_rlen),
                row("Markers", self.marker_list),
                widgets.HTML("<hr>"),
                self.run_button,
            ]
        )

        subclustering_tab = widgets.VBox(
            [
                widgets.HTML("<b>Subclustering parameters</b>"),
                row("Subcluster key", self.subcluster_key),
                row("Subcluster values", self.subcluster_values),
                row("Resolution", self.subcluster_resolution),
                widgets.HTML("<hr>"),
                self.subcluster_button,
            ]
        )

        annotation_tab = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Cluster annotation</b> — map each cluster to a cell type"
                ),
                self.annotation_box,
                widgets.HTML("<hr>"),
                self.apply_annotations_button,
                widgets.HTML("<b>Clustering family tree</b>"),
                self.annotation_tree_out,
            ]
        )

        spatial_tab = widgets.VBox(
            [
                widgets.HTML("<b>Spatial visualization</b>"),
                row("Color by", self.spatial_color),
                row("Region column", self.spatial_unique_region),
                row("X / Y columns", self.spatial_x, self.spatial_y),
                row("Palette", self.spatial_palette),
                row("N columns", self.spatial_n_columns),
                row("Figure size / dot size", self.spatial_figsize, self.spatial_size),
                row("Save", self.spatial_savefig, self.spatial_output_fname),
                widgets.HTML("<hr>"),
                self.spatial_button,
            ]
        )

        save_tab = widgets.VBox(
            [
                widgets.HTML("<b>Save annotated AnnData</b>"),
                row("Output directory", self.output_dir_widget),
                widgets.HTML("<hr>"),
                self.save_button,
            ]
        )

        tabs = widgets.Tab(
            children=[
                clustering_tab,
                subclustering_tab,
                annotation_tab,
                spatial_tab,
                save_tab,
            ]
        )
        for i, title in enumerate(
            ["Clustering", "Subclustering", "Annotation", "Spatial", "Save"]
        ):
            tabs.set_title(i, title)

        controls = widgets.VBox([self.status, self.spinner, self.log_out, tabs])
        plot_panel = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.Label("View:", layout=widgets.Layout(width="40px")),
                        self.view_selector,
                        widgets.Label("Style:", layout=widgets.Layout(width="45px")),
                        self.style_dd,
                        widgets.Label(
                            "Highlight cluster/region:",
                            layout=widgets.Layout(width="130px"),
                        ),
                        self.dotplot_cluster_dd,
                        self.dotplot_region_dd,
                    ]
                ),
                self.plot_out,
            ]
        )
        return widgets.HBox(
            [controls, plot_panel],
            layout=widgets.Layout(align_items="flex-start", flex_flow="row wrap"),
        )

    # ------------------------------------------------------------------ #
    # Status helpers                                                       #
    # ------------------------------------------------------------------ #

    def _set_status(self, msg, level="info"):
        colors = {
            "info": "#333333",
            "success": "#2e7d32",
            "error": "#c62828",
            "warn": "#e65100",
        }
        self.status.value = (
            f"<span style='color:{colors.get(level, '#333')}'>{msg}</span>"
        )

    def _busy(self, on, label=""):
        self.spinner.value = _SPINNER_HTML.format(label=label) if on else ""
        if on:
            self.log_out.layout.display = "block"
        for name in self._BUSY_BUTTONS:
            getattr(self, name).disabled = on

    def _with_progress(self, label, fn):
        """Run fn() with a visible busy state, live stdout log, and uniform
        error handling (including SystemExit, which `except Exception` misses)."""
        self._busy(True, label)
        self.log_out.clear_output(wait=True)
        self._set_status(f"{label}…", "info")
        try:
            with contextlib.redirect_stdout(_WidgetLogWriter(self.log_out)):
                fn()
        except SystemExit as e:
            self._error(label, ValueError(str(e) or "invalid parameters"))
        except Exception as e:
            self._error(label, e)
        finally:
            self._busy(False)

    def _resolve_style(self):
        """Resolve the SPACEcStyle to use for plotting based on the dropdown."""
        choice = self.style_dd.value
        if choice == "active default":
            return self.style if self.style is not None else get_active_style()
        return {
            "publication": SPACEcStyle.publication,
            "presentation": SPACEcStyle.presentation,
            "tissue": SPACEcStyle.tissue,
        }[choice]()

    def _chosen_output_dir(self):
        """Return the folder picked in the Save tab's file browser, or None."""
        return getattr(self.output_dir_widget, "selected", None)

    def _chosen_input_path(self):
        """Return the .h5ad file picked in the Load Data file browser, or None."""
        return getattr(self.input_path, "selected", None)

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #

    def _load_data(self, _=None):
        try:
            if self.adata is None:
                path = self._chosen_input_path()
                if not path or not os.path.isfile(path):
                    self._set_status("Please choose a valid AnnData file.", "error")
                    return
                self.adata = sc.read_h5ad(path)
            self.marker_list.options = list(self.adata.var_names)
            self._refresh_color_options()
            self._set_status("AnnData loaded successfully.", "success")
        except Exception as e:  # pragma: no cover - defensive
            self._error("loading data", e)

    def _run_clustering(self, _=None):
        if self.adata is None:
            self._set_status("Please load an AnnData object first.", "error")
            return
        marker_list = list(self.marker_list.value) if self.marker_list.value else None
        key_added = (
            self.key_added.value or f"{self.method.value}_{self.resolution.value}"
        )

        def _do():
            self.adata = clustering(
                self.adata,
                clustering=self.method.value,
                marker_list=marker_list,
                resolution=self.resolution.value,
                n_neighbors=self.n_neighbors.value,
                reclustering=self.reclustering.value,
                key_added=key_added,
                seed=self.seed.value,
                fs_xdim=self.fs_xdim.value,
                fs_ydim=self.fs_ydim.value,
                fs_rlen=self.fs_rlen.value,
            )
            self.last_key = key_added
            self._make_cluster_plots(key_added, marker_list)
            self._build_annotation_table()
            self._refresh_color_options()
            n = self.adata.obs[key_added].nunique()
            self._set_status(
                f"Clustering complete — {n} clusters in '{key_added}'.", "success"
            )

        self._with_progress("Clustering", _do)

    def _run_subclustering(self, _=None):
        if self.adata is None:
            self._set_status("Please run clustering first.", "error")
            return
        base_key = self.subcluster_key.value
        if not base_key or not self.subcluster_values.value:
            self._set_status("Please provide a subcluster key and values.", "error")
            return
        if base_key not in self.adata.obs:
            self._set_status(f"Column '{base_key}' not found in adata.obs.", "error")
            return
        clusters = [
            c.strip() for c in self.subcluster_values.value.split(",") if c.strip()
        ]
        key_added = base_key + "_subcluster"

        def _do():
            # Splits each listed cluster sequentially -- the first restricts
            # on base_key, every later one restricts on key_added (this
            # round's accumulating column) -- not one pooled split across
            # all of them. See subclustering()'s docstring.
            self.adata = subclustering(
                self.adata,
                base_key,
                clusters,
                resolution=self.subcluster_resolution.value,
                key_added=key_added,
                seed=self.seed.value,
            )
            self.last_key = key_added
            marker_list = (
                list(self.marker_list.value) if self.marker_list.value else None
            )
            self._make_cluster_plots(key_added, marker_list)
            self._build_annotation_table()
            self._refresh_color_options()
            n = self.adata.obs[key_added].nunique()
            self._set_status(
                f"Subclustering complete — {n} clusters in '{key_added}'.", "success"
            )

        self._with_progress("Subclustering", _do)

    def _apply_annotations(self, _=None):
        if self.adata is None or self.last_key is None:
            self._set_status("Run clustering before annotating.", "error")
            return
        mapping = {
            cl: w.value.strip() for cl, w in self._anno_rows.items() if w.value.strip()
        }
        if not mapping:
            self._set_status("No annotations entered.", "warn")
            return

        def _do():
            # missing="keep": clusters left unannotated in the table keep
            # their original label rather than becoming NaN.
            self.adata = annotate_clusters(
                self.adata,
                self.last_key,
                mapping,
                key_added="cell_type",
                missing="keep",
            )
            self._refresh_color_options()
            self._render_tree()
            self._set_status(
                f"Applied {len(mapping)} annotation(s) to adata.obs['cell_type'].",
                "success",
            )

        self._with_progress("Annotating", _do)

    def _render_tree(self):
        """Render the clustering/subclustering/annotation lineage into the
        annotation tab's output area, so GUI users see the same lineage
        sp.tl.clustering_tree() would print in a notebook.

        Uses the Output widget's own append_stdout / outputs reset directly
        rather than ``with self.annotation_tree_out: print(...)`` -- that
        display-hook-based capture (like plain ``clear_output()``) is a
        no-op without a live kernel comm, which would make this silently do
        nothing both in headless tests and in some notebook front ends.
        """
        self.annotation_tree_out.outputs = ()
        tree_ascii = clustering_tree(self.adata, print_tree=False)
        if tree_ascii:
            self.annotation_tree_out.append_stdout(tree_ascii)

    def _render_spatial_fig(self, save):
        """Generate the spatial figure and cache it (does not show/set status)."""
        with self._resolve_style().context():
            catplot(
                self.adata,
                color=self.spatial_color.value,
                unique_region=self.spatial_unique_region.value,
                X=self.spatial_x.value,
                Y=self.spatial_y.value,
                n_columns=self.spatial_n_columns.value,
                palette=self.spatial_palette.value,
                savefig=save,
                output_fname=self.spatial_output_fname.value,
                output_dir=self._chosen_output_dir() or self.output_dir or "./",
                figsize=self.spatial_figsize.value,
                size=self.spatial_size.value,
            )
            fig = plt.gcf()
        self._figs["Spatial"] = _cap_fig_size(fig)
        plt.close(fig)

    def _run_spatial(self, _=None):
        if self.adata is None:
            self._set_status("Please load an AnnData object first.", "error")
            return
        if not self.spatial_color.value:
            self._set_status("Please specify a 'Color by' column.", "error")
            return

        def _do():
            self._render_spatial_fig(save=self.spatial_savefig.value)
            self._show("Spatial")
            self._set_status("Spatial visualization complete.", "success")

        self._with_progress("Spatial visualization", _do)

    def _save_adata(self, _=None):
        if self.adata is None:
            self._set_status("No AnnData object to save.", "error")
            return
        out_dir = self._chosen_output_dir() or self.output_dir
        if not out_dir:
            self._set_status("Please specify an output directory.", "error")
            return

        def _do():
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "adata_annotated.h5ad")
            self.adata.write(path)
            self._set_status(f"AnnData saved to {path}.", "success")

        self._with_progress("Saving", _do)

    # ------------------------------------------------------------------ #
    # Plotting + annotation helpers                                        #
    # ------------------------------------------------------------------ #

    def _make_cluster_plots(self, key, marker_list, show=True):
        """Generate UMAP / Cluster counts / Dotplot figures and cache them.

        Defaults to all markers for the dotplot when none were explicitly
        selected, so it renders automatically instead of silently skipping.

        The cluster picked in `dotplot_cluster_dd` (if any) is highlighted
        consistently across all three: an outline on its UMAP cells, a red
        bar in the cluster-counts chart, and (via `_render_dotplot_fig`) its
        own row/spatial panel in the Dotplot. The region picked in
        `dotplot_region_dd` (if any) subsets both UMAP and Cluster counts to
        that region's cells -- not just a highlight, an actual filter --
        matching what it already does for Dotplot's own spatial panel.
        """
        if not marker_list:
            marker_list = list(self.adata.var_names)
        self._last_marker_list = marker_list
        highlight_cluster = self.dotplot_cluster_dd.value or None
        highlight_region = self.dotplot_region_dd.value or None
        region_col = self.spatial_unique_region.value

        plot_adata = self.adata
        if highlight_region is not None and region_col and region_col in self.adata.obs:
            # .copy(), not a bare boolean-mask view: scanpy's own plotting can write a
            # resolved color mapping back to adata.uns, which raises
            # ImplicitModificationWarning on a view -- and this is a fresh subset built
            # for plotting only, never assigned back to self.adata, so copying it is cheap
            # and safe regardless.
            plot_adata = self.adata[
                self.adata.obs[region_col].astype(str) == str(highlight_region)
            ].copy()

        style = self._resolve_style()
        with style.context():
            ax = _umap(
                plot_adata, color=key, style=style, highlight_cluster=highlight_cluster
            )
            umap_fig = ax[0].figure if isinstance(ax, list) else ax.figure
            if highlight_region is not None:
                (ax[0] if isinstance(ax, list) else ax).set_title(
                    f"{key} -- {highlight_region}"
                )
            self._figs["UMAP"] = _cap_fig_size(umap_fig)
            plt.close(umap_fig)

            counts = plot_adata.obs[key].value_counts().sort_index()
            bar_colors = [
                (
                    style.accent_color_negative
                    if highlight_cluster is not None
                    and str(c) == str(highlight_cluster)
                    else style.accent_color_neutral
                )
                for c in counts.index
            ]
            # Explicit fig/ax + ax= (rather than relying on plt.gcf()/pandas' implicit
            # "current axes") -- with many figures open at once (a real, long GUI session
            # easily leaves dozens around despite each view's own plt.close()), pyplot's
            # notion of the "current" figure can silently drift, so a bare
            # counts.plot(kind="bar") can land on the wrong one.
            hist_fig, hist_ax = plt.subplots()
            counts.plot(kind="bar", color=bar_colors, ax=hist_ax)
            hist_ax.set_xlabel("Cluster")
            hist_ax.set_ylabel("Number of cells")
            title = f"Cluster counts for {key}"
            if highlight_region is not None:
                title += f" -- {highlight_region}"
            hist_ax.set_title(title)
            _apply_spines(hist_ax, style)
            self._figs["Cluster counts"] = _cap_fig_size(hist_fig)
            plt.close(hist_fig)

        self._refresh_dotplot_selectors(key)
        self._render_dotplot_fig()
        if show:
            self._show("UMAP")

    def _refresh_dotplot_selectors(self, key):
        """Repopulate the Dotplot view's cluster/region pickers for `key`.

        Guarded by `_updating_dotplot_selectors` so reassigning `.options`
        (which can itself reset `.value`) doesn't trigger a redundant render
        via `_on_dotplot_selection_change` — mirrors `_SpatialCellTypeWidget.
        _populate_region`'s `_updating` guard in `plotting/_spatial_widget.py`.

        The blank option (displayed as "None") is always a valid, preserved
        selection here, not just a fallback for "not found" -- this runs on
        every highlight-picker change too (via `_make_cluster_plots`), so
        treating "" as stale would silently revert an explicit "no
        highlight" choice back to some cluster name the moment anything else
        re-renders.
        """
        self._updating_dotplot_selectors = True
        try:
            clusters = _sort_clusters(self.adata.obs[key].astype(str).unique())
            prev_cluster = self.dotplot_cluster_dd.value
            self.dotplot_cluster_dd.options = [("None", "")] + [
                (c, c) for c in clusters
            ]
            self.dotplot_cluster_dd.value = (
                prev_cluster
                if prev_cluster == "" or prev_cluster in clusters
                else (clusters[0] if clusters else "")
            )

            region_col = self.spatial_unique_region.value
            regions = (
                sorted(self.adata.obs[region_col].astype(str).unique())
                if region_col and region_col in self.adata.obs
                else []
            )
            prev_region = self.dotplot_region_dd.value
            self.dotplot_region_dd.options = [("None", "")] + [(r, r) for r in regions]
            self.dotplot_region_dd.value = prev_region if prev_region in regions else ""
        finally:
            self._updating_dotplot_selectors = False

    def _render_dotplot_fig(self):
        """Build the Dotplot figure for the active key and cache it.

        Reads the current cluster/region picks from dotplot_cluster_dd /
        dotplot_region_dd (both optional) to highlight a cluster's row and,
        when a region is also picked, show that cluster's spatial
        distribution there via extended_dotplot's highlight_cluster /
        spatial_region params.

        `clustering_markers` is resolved from `adata.uns["clustering_report"]`
        (the marker panel `sp.tl.clustering` actually used to produce
        `last_key`) whenever that record exists, rather than whatever happens
        to be checked in the Markers widget right now -- those can genuinely
        differ (e.g. nothing checked, or the selection changed after
        clustering already ran). Falls back to the widget's own
        `_last_marker_list` only when no such record is found. `extra_markers`
        is deliberately left empty, and `obs_continuous`/`dendrogram`/
        `show_totals` are deliberately dropped -- only the `obs_flags` QC
        block is shown (auto-detected fresh from `adata.obs` each render,
        with tutorial-style shortened labels via `_qc_flag_label_overrides`)
        -- to keep the figure from growing wider than it has to. What's left
        (markers + QC flags) is still typically too wide for the panel with a
        real marker count, so the figure is left completely uncapped rather
        than squeezed to fit: `plot_out`'s own scroll/overflow handles the
        rest (see `_cap_fig_size` -- capping either dimension shrinks the
        whole figure without rescaling font/dot size, crushing content into
        illegibility instead of just making it smaller). This view is a
        fixed, at-a-glance default, not a fully configurable one.
        """
        if self.adata is None or self.last_key is None:
            self._figs.pop("Dotplot", None)
            return
        clustering_markers = (
            _get_clustering_report_markers(self.adata, self.last_key)
            or self._last_marker_list
        )
        if not clustering_markers:
            self._figs.pop("Dotplot", None)
            return
        highlight_cluster = self.dotplot_cluster_dd.value or None
        spatial_region = (
            (self.dotplot_region_dd.value or None) if highlight_cluster else None
        )
        obs_flags = _qc_default_flags(self.adata)
        style = self._resolve_style()
        with style.context():
            dot_fig = extended_dotplot(
                self.adata,
                self.last_key,
                clustering_markers=clustering_markers,
                extra_markers=[],
                obs_continuous=None,
                obs_flags=obs_flags,
                label_overrides=_qc_flag_label_overrides(obs_flags),
                style=style,
                dendrogram=False,
                show_totals=False,
                highlight_cluster=highlight_cluster,
                spatial_region=spatial_region,
                region_col=self.spatial_unique_region.value,
                x_col=self.spatial_x.value,
                y_col=self.spatial_y.value,
            )
            # Neither dimension is capped: this view's natural size grows with
            # both marker/QC-column count (width) and cluster count (height),
            # and capping either shrinks the whole figure without rescaling
            # font/dot size, crushing content into illegibility rather than
            # making it smaller (see _cap_fig_size docstring). plot_out's own
            # scroll/overflow (see its layout in _build_widgets) handles a
            # figure larger than the panel instead. dpi=_DOTPLOT_SCREEN_DPI
            # (lower than the other views' _SCREEN_DPI) keeps a typical plot
            # mostly scroll-free while still scrolling for a large one.
            self._figs["Dotplot"] = _cap_fig_size(
                dot_fig,
                max_height_in=float("inf"),
                max_width_in=float("inf"),
                dpi=_DOTPLOT_SCREEN_DPI,
            )
            plt.close(dot_fig)

    def _on_dotplot_selection_change(self, change):
        """Re-render UMAP / Cluster counts / Dotplot for the new highlight pick.

        Refreshes whichever view is currently showing, rather than forcing a
        switch to Dotplot, since the cluster picker now also drives the UMAP
        outline and Cluster counts bar color (see `_make_cluster_plots`).
        """
        if self._updating_dotplot_selectors:
            return
        if self.adata is None or self.last_key is None:
            return
        self._make_cluster_plots(self.last_key, self._last_marker_list, show=False)
        self._show(self.view_selector.value)

    def _build_annotation_table(self):
        """Rebuild the per-cluster annotation rows for the active key."""
        self._anno_rows = {}
        if (
            self.adata is None
            or self.last_key is None
            or self.last_key not in self.adata.obs
        ):
            self.annotation_box.children = [widgets.HTML("Run clustering first.")]
            return
        counts = self.adata.obs[self.last_key].value_counts()
        header = widgets.HBox(
            [
                widgets.HTML("<b>Cluster</b>", layout=widgets.Layout(width="90px")),
                widgets.HTML("<b>Cells</b>", layout=widgets.Layout(width="90px")),
                widgets.HTML("<b>Cell type</b>", layout=widgets.Layout(width="220px")),
            ]
        )
        rows = [header]
        for cl in _sort_clusters(counts.index.astype(str)):
            txt = widgets.Text(
                value="", placeholder="cell type", layout=widgets.Layout(width="220px")
            )
            self._anno_rows[str(cl)] = txt
            rows.append(
                widgets.HBox(
                    [
                        widgets.Label(str(cl), layout=widgets.Layout(width="90px")),
                        widgets.Label(
                            str(int(counts[cl])), layout=widgets.Layout(width="90px")
                        ),
                        txt,
                    ]
                )
            )
        self.annotation_box.children = rows

    def _refresh_color_options(self):
        """Keep the Spatial tab's 'Color by' dropdown in sync with adata.obs columns."""
        if self.adata is None:
            self.spatial_color.options = [""]
            return
        cols = list(self.adata.obs.columns)
        current = self.spatial_color.value
        self.spatial_color.options = [""] + cols
        if (
            current in cols
        ):  # reassigning .options can reset .value; restore the selection
            self.spatial_color.value = current

    def _on_view_change(self, change):
        self._show(change["new"])

    def _on_style_change(self, change):
        """Regenerate whichever plots already exist under the new style."""
        if self.adata is None:
            return
        try:
            if self.last_key is not None and self.last_key in self.adata.obs:
                self._make_cluster_plots(
                    self.last_key, self._last_marker_list, show=False
                )
            if "Spatial" in self._figs and self.spatial_color.value:
                self._render_spatial_fig(save=False)
            self._show(self.view_selector.value)
        except Exception as e:
            self._error("applying style", e)

    def _show(self, which):
        """Display a cached figure in the persistent plot area (no leaks)."""
        if self.view_selector.value != which:
            # Triggers _on_view_change which re-enters _show with the new value.
            self.view_selector.value = which
            return
        with self.plot_out:
            clear_output(wait=True)
            fig = self._figs.get(which)
            if fig is None:
                print(f"No '{which}' plot yet. Run the corresponding step.")
            else:
                _display_fig_native_size(fig)

    def _error(self, where, exc):
        tb = traceback.format_exc().strip().splitlines()
        last = tb[-1] if tb else str(exc)
        self._set_status(f"Error during {where}: {last}", "error")


def launch_interactive_clustering(adata=None, output_dir=None, style=None):
    """Launch the interactive clustering app for single-cell data.

    A pure-``ipywidgets`` replacement for the former Panel implementation. It
    wraps :func:`spacec.tools.clustering` and :func:`spacec.plotting.catplot`,
    provides clustering / subclustering / annotation / spatial / save tabs, and
    renders all plots into a single persistent area.

    Parameters
    ----------
    adata : AnnData, optional
        Single-cell data. If given, it is loaded automatically and
        ``output_dir`` is required.
    output_dir : str, optional
        Directory where the annotated AnnData object is saved. Required when
        ``adata`` is provided.
    style : SPACEcStyle, optional
        Plotting style applied to the generated figures. Defaults to the active
        SPACEc style; can also be switched live from the UI.

    Returns
    -------
    _ClusteringGUI
        The GUI controller instance (also displayed in the notebook).

    Raises
    ------
    ValueError
        If ``adata`` is provided without ``output_dir``, or ``output_dir`` is
        not a string.
    """
    gui = _ClusteringGUI(adata=adata, output_dir=output_dir, style=style)
    display(gui._ui())
    return gui
