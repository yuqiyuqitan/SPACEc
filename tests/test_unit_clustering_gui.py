"""Unit tests for spacec.tools._clustering_gui (ipywidgets clustering GUI).

The GUI is exercised headlessly: ``launch_interactive_clustering`` returns a
``_ClusteringGUI`` instance whose callback methods (``_run_clustering``,
``_apply_annotations``, ``_save_adata`` …) are invoked directly without a live
Jupyter frontend. Plot rendering uses the non-interactive ``Agg`` backend.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import os

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spacec.tools._clustering_gui import (
    _MAX_FIG_HEIGHT_IN,
    _MAX_FIG_WIDTH_IN,
    _SCREEN_DPI,
    _cap_fig_size,
    _ClusteringGUI,
    _display_fig_native_size,
    _sort_clusters,
    launch_interactive_clustering,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_adata(n_cells=80, n_markers=6, seed=0):
    """Two well-separated blobs so leiden reliably finds >1 cluster."""
    rng = np.random.default_rng(seed)
    half = n_cells // 2
    block_a = rng.normal(0.0, 0.3, size=(half, n_markers))
    block_b = rng.normal(5.0, 0.3, size=(n_cells - half, n_markers))
    X = np.vstack([block_a, block_b]).astype(np.float32)
    var = pd.DataFrame(index=[f"marker_{i}" for i in range(n_markers)])
    obs = pd.DataFrame(
        {
            "x": rng.uniform(0, 100, size=n_cells),
            "y": rng.uniform(0, 100, size=n_cells),
            "unique_region": ["reg1"] * n_cells,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    return anndata.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def adata():
    return _make_adata()


def _fake_clustering(
    adata, clustering="leiden", marker_list=None, resolution=1, key_added=None, **kwargs
):
    """Lightweight stand-in for spacec.tools.clustering.

    Avoids depending on leidenalg / GPU libs at unit-test time — the real
    clustering algorithms are covered by the integration tests. Produces a
    deterministic 3-cluster labelling plus a UMAP embedding, matching the
    side effects the GUI relies on. Also emits the same ordered print()
    checkpoints as the real backend, so tests can verify the GUI's stdout
    capture (_WidgetLogWriter / _with_progress) actually streams them.
    """
    print("Computing neighbors and UMAP")
    print("- neighbors")
    print("- UMAP")
    print("Clustering")
    print("Leiden clustering")
    if key_added is None:
        key_added = f"{clustering}_{resolution}"
    n = adata.n_obs
    labels = (np.arange(n) % 3).astype(str)
    adata.obs[key_added] = pd.Categorical(labels)
    adata.obsm["X_umap"] = np.random.default_rng(0).normal(size=(n, 2))
    return adata


def _fake_clustering_system_exit(*args, **kwargs):
    """Stand-in for clustering() raising SystemExit on invalid params."""
    raise SystemExit("invalid clustering option")


def _fake_subclustering(
    adata, cluster_key, clusters, resolution=0.3, key_added=None, seed=42, **kwargs
):
    """Lightweight stand-in for spacec.tools.subclustering.

    Splits each requested cluster's cells into two "{c},{0|1}"-style labels
    and carries every other cluster's label forward unchanged -- enough
    structure for the GUI wiring tests without depending on real leidenalg
    convergence on tiny synthetic data.
    """
    print("Subclustering")
    if isinstance(clusters, str):
        clusters = [clusters]
    if key_added is None:
        key_added = f"{cluster_key}_subcluster"
    base = adata.obs[cluster_key].astype(str)
    new_col = base.copy()
    for c in clusters:
        idx = np.flatnonzero((base == c).values)
        for j, i in enumerate(idx):
            new_col.iloc[i] = f"{c},{j % 2}"
    adata.obs[key_added] = pd.Categorical(new_col.values)
    return adata


class _FakeChooser:
    """Stand-in for ipyfilechooser.FileChooser exposing only `.selected`,
    so tests don't need to drive FileChooser's real internal file-list
    click-simulation API."""

    def __init__(self, selected=None):
        self.selected = selected


@pytest.fixture
def patch_clustering(monkeypatch):
    """Patch the clustering backend the GUI calls with the fake above."""
    monkeypatch.setattr("spacec.tools._clustering_gui.clustering", _fake_clustering)


@pytest.fixture
def patch_subclustering(monkeypatch):
    """Patch the subclustering backend the GUI calls with the fake above."""
    monkeypatch.setattr(
        "spacec.tools._clustering_gui.subclustering", _fake_subclustering
    )


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


def test_launch_returns_gui_instance(adata, tmp_path):
    gui = launch_interactive_clustering(adata=adata, output_dir=str(tmp_path))
    assert isinstance(gui, _ClusteringGUI)
    # core widgets exist
    assert gui.run_button is not None
    assert gui.plot_out is not None
    # markers auto-populated from var_names on load
    assert list(gui.marker_list.options) == list(adata.var_names)


def test_missing_output_dir_raises(adata):
    with pytest.raises(ValueError):
        launch_interactive_clustering(adata=adata)


def test_non_string_output_dir_raises(adata):
    with pytest.raises(ValueError):
        _ClusteringGUI(adata=adata, output_dir=123)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def test_run_clustering_populates_obs_and_umap(patch_clustering, adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui.method.value = "leiden"
    gui.resolution.value = 1.0
    gui.n_neighbors.value = 10
    gui._run_clustering()

    key = "leiden_1.0"
    assert gui.last_key == key
    assert key in gui.adata.obs
    assert "X_umap" in gui.adata.obsm
    # status should not be an error
    assert "color:#c62828" not in gui.status.value
    # UMAP + cluster-counts figures cached, ready for the persistent plot area
    assert "UMAP" in gui._figs
    assert "Cluster counts" in gui._figs


def test_run_clustering_with_markers_builds_dotplot(patch_clustering, adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui.marker_list.value = tuple(adata.var_names[:3])
    gui._run_clustering()
    assert "Dotplot" in gui._figs
    assert gui._last_marker_list == list(adata.var_names[:3])


def test_run_clustering_without_marker_selection_still_builds_dotplot(
    patch_clustering, adata, tmp_path
):
    """Dotplot must render automatically using all markers when the user
    hasn't explicitly selected any in the Markers widget."""
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    assert gui.marker_list.value == ()  # nothing selected by default
    gui._run_clustering()
    assert "Dotplot" in gui._figs
    assert gui._last_marker_list == list(adata.var_names)


def test_run_clustering_populates_dotplot_selectors(patch_clustering, adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()

    # blank option is displayed as "None" but keeps "" as its underlying value
    # (see _refresh_dotplot_selectors), so no highlight is selected by default.
    assert gui.dotplot_cluster_dd.options == (
        ("None", ""),
        ("0", "0"),
        ("1", "1"),
        ("2", "2"),
    )
    assert gui.dotplot_cluster_dd.value == ""
    # region options come from the Spatial tab's region-column field, default "unique_region"
    assert gui.dotplot_region_dd.options == (("None", ""), ("reg1", "reg1"))


def test_dotplot_selection_change_rerenders_with_highlight(
    patch_clustering, adata, tmp_path
):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()

    dot_axes = gui._figs["Dotplot"].axes
    # extended_dotplot with this fixture (no qc-flag obs columns, so no QC panel) and the
    # GUI's reduced default (no dendrogram, no totals bar -- see _render_dotplot_fig):
    # cluster-color strip, dots, colorbar -- no spatial panel yet since no region is picked.
    # Also guards against the empty-figure regression this replaced (marker_dotplot's
    # plt.show() clearing the figure before plt.gcf() ran).
    assert len(dot_axes) > 0
    assert len(dot_axes) < 6

    gui.dotplot_cluster_dd.value = "1"
    gui.dotplot_region_dd.value = "reg1"

    # the highlight picker is global now (also drives UMAP/Cluster counts, see
    # _make_cluster_plots) -- it must refresh whatever view is active rather than
    # forcing a switch to Dotplot.
    assert gui.view_selector.value == "UMAP"
    # picking both a cluster and a region adds the spatial panel -> more axes
    assert len(gui._figs["Dotplot"].axes) > len(dot_axes)


def test_highlight_picker_colors_cluster_counts_bar_red(
    patch_clustering, adata, tmp_path
):
    """Picking a cluster in the (now-global) highlight dropdown must color that cluster's
    bar with the style's accent_color_negative in Cluster counts, and every other bar with
    accent_color_neutral (see _make_cluster_plots)."""
    import matplotlib.colors as mcolors

    from spacec.plotting._style import get_active_style

    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    style = get_active_style()
    neutral_rgba = mcolors.to_rgba(style.accent_color_neutral)
    negative_rgba = mcolors.to_rgba(style.accent_color_negative)

    # counts.plot(kind="bar") draws patches in counts.index order (sort_index() in
    # _make_cluster_plots), so this is deterministic without depending on tick-label
    # Text objects, which matplotlib doesn't always populate before a canvas draw.
    cluster_order = sorted(gui.adata.obs[gui.last_key].astype(str).unique())

    bar_colors = [
        p.get_facecolor() for p in gui._figs["Cluster counts"].axes[0].patches
    ]
    assert all(c == neutral_rgba for c in bar_colors)

    gui.dotplot_cluster_dd.value = "1"
    bar_colors2 = [
        p.get_facecolor() for p in gui._figs["Cluster counts"].axes[0].patches
    ]
    highlighted_idx = cluster_order.index("1")
    assert bar_colors2[highlighted_idx] == negative_rgba
    assert all(
        c == neutral_rgba for i, c in enumerate(bar_colors2) if i != highlighted_idx
    )


def test_highlight_picker_adds_umap_outline(patch_clustering, adata, tmp_path):
    """Picking a cluster in the highlight dropdown must add an outline scatter on top of
    UMAP's own collection (see spacec.plotting.umap's highlight_cluster param)."""
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    n_collections_before = len(gui._figs["UMAP"].axes[0].collections)

    gui.dotplot_cluster_dd.value = "1"
    assert len(gui._figs["UMAP"].axes[0].collections) == n_collections_before + 1


def test_highlight_picker_none_option_persists_and_clears_dotplot_highlight(
    patch_clustering, adata, tmp_path, monkeypatch
):
    """Selecting the blank ("None") option must stick, and must actually clear the highlight
    passed to extended_dotplot, rather than silently reverting to some cluster name the next
    time _refresh_dotplot_selectors runs (a real bug: it ran on every highlight-picker change
    via _make_cluster_plots, and its old fallback treated "" as an invalid/stale value).
    """
    import spacec.tools._clustering_gui as gui_mod

    captured = {}
    real_extended_dotplot = gui_mod.extended_dotplot

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return real_extended_dotplot(*args, **kwargs)

    monkeypatch.setattr(gui_mod, "extended_dotplot", _spy)

    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    n_collections_baseline = len(gui._figs["UMAP"].axes[0].collections)

    gui.dotplot_cluster_dd.value = "1"
    assert captured["highlight_cluster"] == "1"

    gui.dotplot_cluster_dd.value = ""
    assert gui.dotplot_cluster_dd.value == ""  # must not silently revert to e.g. "0"
    assert captured["highlight_cluster"] is None
    assert len(gui._figs["UMAP"].axes[0].collections) == n_collections_baseline


def test_region_picker_subsets_umap_and_cluster_counts(patch_clustering, tmp_path):
    """Picking a region must actually subset UMAP/Cluster counts to that region's cells --
    not just highlight it -- mirroring what already happens for Dotplot's own spatial panel
    (see _make_cluster_plots)."""
    two_region_adata = _make_adata(n_cells=80)
    two_region_adata.obs["unique_region"] = ["reg1"] * 40 + ["reg2"] * 40

    gui = _ClusteringGUI(adata=two_region_adata, output_dir=str(tmp_path))
    gui._run_clustering()

    n_points_all = len(gui._figs["UMAP"].axes[0].collections[0].get_offsets())
    total_count_all = sum(
        p.get_height() for p in gui._figs["Cluster counts"].axes[0].patches
    )
    assert n_points_all == 80
    assert total_count_all == 80

    gui.dotplot_region_dd.value = "reg1"

    n_points_region = len(gui._figs["UMAP"].axes[0].collections[0].get_offsets())
    total_count_region = sum(
        p.get_height() for p in gui._figs["Cluster counts"].axes[0].patches
    )
    assert n_points_region == 40
    assert total_count_region == 40


def test_dotplot_region_without_cluster_does_not_error(
    patch_clustering, adata, tmp_path
):
    """Picking a region before a cluster must not raise extended_dotplot's
    'spatial_region requires highlight_cluster' ValueError through the GUI."""
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    gui.dotplot_cluster_dd.value = ""
    gui.dotplot_region_dd.value = "reg1"
    assert "Dotplot" in gui._figs  # rendered without error


def test_dotplot_default_wiring(patch_clustering, adata, tmp_path, monkeypatch):
    """The Dotplot tab's default call to extended_dotplot: no extra ("functional") markers,
    only the obs_flags QC block (obs_continuous/dendrogram/totals dropped -- the GUI's
    embedded panel is too small to fit the full multi-section layout legibly; see
    _render_dotplot_fig), with tutorial-style shortened QC-flag labels and neither dimension
    capped (see test_cap_fig_size_never_upscales_small_figure for the underlying mechanism).
    """
    adata.obs["area"] = np.arange(
        adata.n_obs, dtype=float
    )  # would be obs_continuous, if kept
    adata.obs["qc_flag_test"] = [i % 3 == 0 for i in range(adata.n_obs)]  # -> obs_flags

    import spacec.tools._clustering_gui as gui_mod

    captured = {}
    real_extended_dotplot = gui_mod.extended_dotplot

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return real_extended_dotplot(*args, **kwargs)

    monkeypatch.setattr(gui_mod, "extended_dotplot", _spy)

    real_cap_fig_size = gui_mod._cap_fig_size
    cap_calls = []

    def _cap_spy(fig, **kwargs):
        cap_calls.append(kwargs)
        return real_cap_fig_size(fig, **kwargs)

    monkeypatch.setattr(gui_mod, "_cap_fig_size", _cap_spy)

    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui.marker_list.value = tuple(adata.var_names[:3])
    gui._run_clustering()

    assert captured["extra_markers"] == []
    assert captured["obs_continuous"] is None
    assert captured["obs_flags"] == ["qc_flag_test"]
    assert captured["label_overrides"] == {"qc_flag_test": "test"}
    assert captured["dendrogram"] is False
    assert captured["show_totals"] is False

    dotplot_cap_call = cap_calls[-1]
    assert dotplot_cap_call["max_height_in"] == float("inf")
    assert dotplot_cap_call["max_width_in"] == float("inf")


def test_dotplot_prefers_clustering_report_markers_over_widget_selection(
    patch_clustering, adata, tmp_path, monkeypatch
):
    """clustering_markers should come from adata.uns['clustering_report'] (the real markers
    sp.tl.clustering used to build this column) whenever that record exists, not from
    whatever's currently checked in the Markers widget -- these can genuinely differ (e.g.
    the widget selection changed after clustering already ran)."""
    import json

    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui.marker_list.value = tuple(adata.var_names[:2])
    gui._run_clustering()
    key = gui.last_key

    recorded_markers = list(adata.var_names[3:5])
    gui.adata.uns["clustering_report"] = json.dumps(
        [{"action": "clustering", "key_added": key, "marker_list": recorded_markers}]
    )

    import spacec.tools._clustering_gui as gui_mod

    captured = {}
    real_extended_dotplot = gui_mod.extended_dotplot

    def _spy(*args, **kwargs):
        captured["clustering_markers"] = kwargs.get("clustering_markers")
        return real_extended_dotplot(*args, **kwargs)

    monkeypatch.setattr(gui_mod, "extended_dotplot", _spy)
    gui._render_dotplot_fig()

    assert captured["clustering_markers"] == recorded_markers


def test_run_clustering_without_adata_sets_error():
    gui = _ClusteringGUI()  # no adata, no output_dir is allowed here
    gui._run_clustering()
    assert "color:#c62828" in gui.status.value  # error colour


# ---------------------------------------------------------------------------
# Subclustering
# ---------------------------------------------------------------------------


def test_run_subclustering_populates_obs(
    patch_clustering, patch_subclustering, adata, tmp_path
):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    base_key = gui.last_key

    gui.subcluster_key.value = base_key
    gui.subcluster_values.value = "0, 1"
    gui.subcluster_resolution.value = 0.3
    gui._run_subclustering()

    key = gui.last_key
    assert key == f"{base_key}_subcluster"
    assert key in gui.adata.obs
    assert "color:#c62828" not in gui.status.value
    # new subcluster labels follow the "{c},{i}" convention
    labels = gui.adata.obs[key].astype(str)
    assert labels.str.contains(r"^0,\d+$|^1,\d+$|^2$", regex=True).all()
    # UMAP + cluster counts + annotation table rebuilt for the new key, same as clustering
    assert "UMAP" in gui._figs
    assert len(gui._anno_rows) == gui.adata.obs[key].nunique()


def test_run_subclustering_streams_backend_prints_to_log(
    patch_clustering, patch_subclustering, adata, tmp_path
):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    gui.subcluster_key.value = gui.last_key
    gui.subcluster_values.value = "0"
    gui._run_subclustering()
    logged = "".join(o.get("text", "") for o in gui.log_out.outputs)
    assert "Subclustering" in logged


def test_run_subclustering_without_clustering_errors(adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_subclustering()
    assert "color:#c62828" in gui.status.value


def test_run_subclustering_missing_key_or_values_errors(
    patch_clustering, adata, tmp_path
):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    gui._run_subclustering()  # neither subcluster_key nor subcluster_values set
    assert "color:#c62828" in gui.status.value


def test_run_subclustering_unknown_key_errors(patch_clustering, adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    gui.subcluster_key.value = "does_not_exist"
    gui.subcluster_values.value = "0"
    gui._run_subclustering()
    assert "color:#c62828" in gui.status.value


def test_run_subclustering_backend_error_is_caught(patch_clustering, adata, tmp_path):
    """subclustering() raises plain ValueError (e.g. unknown cluster label);
    the GUI must surface it as a status error, not propagate it."""
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    gui.subcluster_key.value = gui.last_key
    gui.subcluster_values.value = "not_a_real_cluster"
    gui._run_subclustering()  # real subclustering() backend, no patch_subclustering here
    assert "color:#c62828" in gui.status.value
    for name in gui._BUSY_BUTTONS:
        assert getattr(gui, name).disabled is False


# ---------------------------------------------------------------------------
# Progress feedback (busy state, live log, SystemExit handling)
# ---------------------------------------------------------------------------


def test_run_clustering_clears_busy_state_on_success(patch_clustering, adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    assert gui.spinner.value == ""
    for name in gui._BUSY_BUTTONS:
        assert getattr(gui, name).disabled is False


def test_run_clustering_streams_backend_prints_to_log(
    patch_clustering, adata, tmp_path
):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    logged = "".join(o.get("text", "") for o in gui.log_out.outputs)
    assert "neighbors" in logged
    assert "Leiden clustering" in logged


def test_run_clustering_system_exit_is_caught(monkeypatch, adata, tmp_path):
    monkeypatch.setattr(
        "spacec.tools._clustering_gui.clustering", _fake_clustering_system_exit
    )
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()  # must not raise SystemExit out of the callback
    assert "color:#c62828" in gui.status.value
    for name in gui._BUSY_BUTTONS:
        assert getattr(gui, name).disabled is False


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def test_annotation_table_and_apply(patch_clustering, adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    # table built with one row per cluster
    clusters = list(gui._anno_rows.keys())
    assert len(clusters) == gui.adata.obs[gui.last_key].nunique()

    # annotate every cluster, then apply
    for i, cl in enumerate(clusters):
        gui._anno_rows[cl].value = f"type_{i}"
    gui._apply_annotations()

    assert "cell_type" in gui.adata.obs
    assert str(gui.adata.obs["cell_type"].dtype) == "category"
    # mapped values present
    assert (
        gui.adata.obs["cell_type"]
        .isin([f"type_{i}" for i in range(len(clusters))])
        .all()
    )


def test_apply_annotations_keeps_unmapped_label(patch_clustering, adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    clusters = list(gui._anno_rows.keys())
    # annotate only the first cluster
    gui._anno_rows[clusters[0]].value = "annotated"
    gui._apply_annotations()
    cats = set(gui.adata.obs["cell_type"].astype(str))
    assert "annotated" in cats
    # unannotated clusters keep their original label (not NaN)
    assert clusters[1] in cats


def test_apply_annotations_before_clustering_errors(adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._apply_annotations()
    assert "color:#c62828" in gui.status.value


def test_apply_annotations_populates_tree(patch_clustering, adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    assert gui.annotation_tree_out.outputs == ()  # nothing rendered yet

    clusters = list(gui._anno_rows.keys())
    for i, cl in enumerate(clusters):
        gui._anno_rows[cl].value = f"type_{i}"
    gui._apply_annotations()

    tree_text = "".join(o.get("text", "") for o in gui.annotation_tree_out.outputs)
    assert "Clustering family tree" in tree_text
    assert "→ type_0" in tree_text
    assert "cell_type" in gui.adata.uns.get("clustering_tree", "")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def test_save_adata_writes_file(patch_clustering, adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    gui._save_adata()
    out = os.path.join(str(tmp_path), "adata_annotated.h5ad")
    assert os.path.isfile(out)
    assert gui.save_button.disabled is False


def test_save_without_adata_errors(tmp_path):
    gui = _ClusteringGUI(output_dir=str(tmp_path))
    gui._save_adata()
    assert "color:#c62828" in gui.status.value


# ---------------------------------------------------------------------------
# File browser (ipyfilechooser)
# ---------------------------------------------------------------------------


def test_load_data_from_filechooser_selection(tmp_path):
    adata_obj = _make_adata()
    h5ad_path = tmp_path / "in.h5ad"
    adata_obj.write(str(h5ad_path))

    gui = _ClusteringGUI()  # no adata -> Load Data section is active
    gui.input_path = _FakeChooser(str(h5ad_path))
    gui._load_data()

    assert gui.adata is not None
    assert "color:#c62828" not in gui.status.value
    assert list(gui.marker_list.options) == list(gui.adata.var_names)


def test_load_data_without_filechooser_selection_errors():
    gui = _ClusteringGUI()
    gui.input_path = _FakeChooser(selected=None)
    gui._load_data()
    assert "color:#c62828" in gui.status.value


def test_save_adata_uses_filechooser_selection_over_constructor_default(
    patch_clustering, adata, tmp_path
):
    other_dir = tmp_path / "chosen"
    other_dir.mkdir()
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    gui.output_dir_widget = _FakeChooser(str(other_dir))
    gui._save_adata()
    assert os.path.isfile(os.path.join(str(other_dir), "adata_annotated.h5ad"))
    # constructor default dir must NOT have received the file
    assert not os.path.isfile(os.path.join(str(tmp_path), "adata_annotated.h5ad"))


def test_save_adata_falls_back_when_filechooser_unselected(
    patch_clustering, adata, tmp_path
):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    gui.output_dir_widget = _FakeChooser(selected=None)  # user hasn't picked yet
    gui._save_adata()
    assert os.path.isfile(os.path.join(str(tmp_path), "adata_annotated.h5ad"))


# ---------------------------------------------------------------------------
# Style dropdown redraw
# ---------------------------------------------------------------------------


def test_style_change_redraws_existing_plot(patch_clustering, adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    before_id = id(gui._figs["UMAP"])
    gui.style_dd.value = "tissue"
    assert id(gui._figs["UMAP"]) != before_id


def test_style_change_with_no_plots_is_noop(adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui.style_dd.value = "publication"  # must not raise
    assert gui._figs == {}


# ---------------------------------------------------------------------------
# Figure size cap
# ---------------------------------------------------------------------------


def test_cap_fig_size_shrinks_oversized_figure():
    fig = plt.figure(figsize=(17, 12))
    try:
        _cap_fig_size(fig)
        w, h = fig.get_size_inches()
        assert w <= _MAX_FIG_WIDTH_IN + 1e-6
        assert h <= _MAX_FIG_HEIGHT_IN + 1e-6
        assert fig.get_dpi() == _SCREEN_DPI
    finally:
        plt.close(fig)


def test_cap_fig_size_never_upscales_small_figure():
    fig = plt.figure(figsize=(3, 2))
    try:
        _cap_fig_size(fig)
        w, h = fig.get_size_inches()
        assert (w, h) == pytest.approx((3, 2))
    finally:
        plt.close(fig)


def test_run_clustering_caches_capped_figures(patch_clustering, adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    w, h = gui._figs["UMAP"].get_size_inches()
    assert w <= _MAX_FIG_WIDTH_IN + 1e-6
    assert h <= _MAX_FIG_HEIGHT_IN + 1e-6
    assert gui._figs["UMAP"].get_dpi() == _SCREEN_DPI


# ---------------------------------------------------------------------------
# Native-size figure display (plot_out scrolling)
# ---------------------------------------------------------------------------


def test_display_fig_native_size_renders_img_with_max_width_none(monkeypatch):
    """Jupyter's own CSS shrinks an <img> to fit its container unless max-width is
    overridden, which would silently defeat plot_out's overflow="auto" for an oversized
    figure (see _display_fig_native_size's docstring)."""
    import spacec.tools._clustering_gui as gui_mod

    captured = {}

    def _fake_display(obj):
        captured["html"] = obj.data

    monkeypatch.setattr(gui_mod, "display", _fake_display)

    fig = plt.figure(figsize=(4, 3))
    try:
        _display_fig_native_size(fig)
    finally:
        plt.close(fig)

    assert "<img" in captured["html"]
    assert "max-width:none" in captured["html"]
    assert "data:image/png;base64," in captured["html"]


def test_display_fig_native_size_uses_bbox_inches_tight(monkeypatch):
    """A right-margin legend (e.g. scanpy's UMAP legend) extends past the axes' own bounding
    box and gets silently clipped by a plain savefig() -- bbox_inches="tight" is required to
    keep the label text (see _display_fig_native_size's docstring; this is exactly what made
    real UMAP legends lose their text labels in the GUI)."""
    import spacec.tools._clustering_gui as gui_mod

    monkeypatch.setattr(gui_mod, "display", lambda obj: None)

    fig = plt.figure(figsize=(4, 3))
    captured = {}
    real_savefig = fig.savefig

    def _spy_savefig(*args, **kwargs):
        captured.update(kwargs)
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(fig, "savefig", _spy_savefig)
    try:
        _display_fig_native_size(fig)
    finally:
        plt.close(fig)

    assert captured.get("bbox_inches") == "tight"


# ---------------------------------------------------------------------------
# Spatial "color by" dropdown
# ---------------------------------------------------------------------------


def test_spatial_color_options_seeded_from_obs_on_construction(adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    assert gui.spatial_color.options == ("",) + tuple(adata.obs.columns)
    assert gui.spatial_color.value == ""


def test_spatial_color_options_refresh_after_clustering(
    patch_clustering, adata, tmp_path
):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    assert gui.last_key in gui.spatial_color.options


def test_spatial_color_options_refresh_after_annotation(
    patch_clustering, adata, tmp_path
):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_clustering()
    for i, cl in enumerate(gui._anno_rows):
        gui._anno_rows[cl].value = f"type_{i}"
    gui._apply_annotations()
    assert "cell_type" in gui.spatial_color.options


def test_spatial_color_selection_preserved_across_refresh(
    patch_clustering, adata, tmp_path
):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui.spatial_color.value = "unique_region"
    gui._run_clustering()  # triggers _refresh_color_options internally
    assert gui.spatial_color.value == "unique_region"


def test_run_spatial_without_color_selection_errors(adata, tmp_path):
    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui._run_spatial()
    assert (
        "color:#c62828" in gui.status.value
    )  # blank "" default still trips validation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_sort_clusters_numeric():
    assert _sort_clusters(["10", "2", "1"]) == ["1", "2", "10"]


def test_sort_clusters_mixed_falls_back_to_str():
    out = _sort_clusters(["b", "a", "c"])
    assert out == ["a", "b", "c"]


def test_resolve_style_presets(adata, tmp_path):
    from spacec.plotting._style import SPACEcStyle

    gui = _ClusteringGUI(adata=adata, output_dir=str(tmp_path))
    gui.style_dd.value = "tissue"
    assert isinstance(gui._resolve_style(), SPACEcStyle)
    gui.style_dd.value = "active default"
    assert isinstance(gui._resolve_style(), SPACEcStyle)
