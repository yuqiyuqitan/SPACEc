"""Unit tests for spacec.plotting._general (catplot and newer publication plots)."""

import warnings

import matplotlib

matplotlib.use("Agg")

import anndata
import numpy as np
import pandas as pd
import pytest

from spacec.plotting._general import (
    BC_projection,
    _get_clustering_report_markers,
    _qc_default_flags,
    _qc_flag_label_overrides,
    catplot,
    classification_report_heatmap,
    cluster_composition_heatmap,
    cluster_qc,
    cn_map,
    cn_upset,
    compensation_qc,
    confusion_matrix_heatmap,
    correlation_heatmap,
    distance_graph,
    expression_on_masks,
    extended_dotplot,
    marker_dotplot,
    pl_swarm_box_ad,
    plot_masks,
    plot_top_n_distances,
    ppa_res_donut,
    swarm_box_plot,
    umap,
)


def _make_adata():
    obs = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "unique_region": ["region1", "region1", "region2", "region2"],
        }
    )
    return anndata.AnnData(X=np.zeros((4, 1)), obs=obs)


def test_catplot_color_same_as_unique_region():
    # Regression test: color and unique_region referencing the same obs
    # column used to raise
    # AttributeError: 'DataFrame' object has no attribute 'unique'
    # because selecting adata.obs[[X, Y, color, unique_region]] with a
    # duplicated column name produced a DataFrame with two identically
    # named columns, breaking downstream Series-only calls like .unique().
    adata = _make_adata()
    catplot(adata, color="unique_region", unique_region="unique_region", X="x", Y="y")


def test_catplot_color_different_from_unique_region():
    adata = _make_adata()
    adata.obs["cell_type"] = ["a", "b", "a", "b"]
    catplot(adata, color="cell_type", unique_region="unique_region", X="x", Y="y")


def test_catplot_passes_scatter_kws():
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    adata = _make_adata()
    adata.obs["cell_type"] = ["a", "b", "a", "b"]
    fig = catplot(
        adata,
        color="cell_type",
        unique_region="unique_region",
        X="x",
        Y="y",
        scatter_kws={"linewidth": 0},
    )
    dots = [
        c for ax in fig.axes for c in ax.collections if isinstance(c, PathCollection)
    ]
    assert dots and all(np.allclose(c.get_linewidths(), 0) for c in dots)
    plt.close("all")


def _make_clustering_adata(n_cells=40, n_markers=5, n_clusters=3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_cells, n_markers))
    var_names = [f"marker{i}" for i in range(n_markers)]
    obs = pd.DataFrame(
        {
            "leiden": rng.integers(0, n_clusters, size=n_cells).astype(str),
            "x": rng.uniform(0, 100, size=n_cells),
            "y": rng.uniform(0, 100, size=n_cells),
            "unique_region": ["reg1"] * (n_cells // 2)
            + ["reg2"] * (n_cells - n_cells // 2),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    adata.obsm["X_umap"] = rng.normal(size=(n_cells, 2))
    return adata


def test_umap_smoke():
    adata = _make_clustering_adata()
    ax = umap(adata, color="leiden")
    assert ax is not None


def test_umap_despines_top_and_right():
    adata = _make_clustering_adata()
    ax = umap(adata, color="leiden")
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False


def test_umap_highlight_cluster_adds_outline_scatter():
    """highlight_cluster draws an extra edge-only scatter on top, without touching the
    original collection (see _outline_umap_cluster)."""
    adata = _make_clustering_adata()
    ax_plain = umap(adata, color="leiden")
    ax_highlighted = umap(adata, color="leiden", highlight_cluster="0")
    assert len(ax_highlighted.collections) == len(ax_plain.collections) + 1
    outline = ax_highlighted.collections[-1]
    assert (
        outline.get_facecolor().size == 0 or (outline.get_facecolor()[:, 3] == 0).all()
    )


def test_umap_highlight_cluster_no_match_is_noop():
    adata = _make_clustering_adata()
    ax_plain = umap(adata, color="leiden")
    ax_no_match = umap(adata, color="leiden", highlight_cluster="does-not-exist")
    assert len(ax_no_match.collections) == len(ax_plain.collections)


def test_marker_dotplot_smoke():
    adata = _make_clustering_adata()
    axes = marker_dotplot(
        adata, list(adata.var_names), groupby="leiden", dendrogram=False
    )
    assert axes is not None
    assert axes["dots"] is not None
    assert axes["dendrogram"] is None


def test_marker_dotplot_dendrogram_reorders_and_draws_axis():
    adata = _make_clustering_adata()
    axes = marker_dotplot(
        adata, list(adata.var_names), groupby="leiden", dendrogram=True
    )
    assert axes["dendrogram"] is not None
    ordered_labels = [t.get_text() for t in axes["dots"].get_yticklabels()]
    assert set(ordered_labels) == set(adata.obs["leiden"].unique())


def test_marker_dotplot_flags_tiny_cluster():
    # 38 cells in cluster "0", 2 cells in cluster "1" -> "1" is 5% of all
    # cells, so a min_cluster_frac of 10% must flag it in the size panel.
    rng = np.random.default_rng(0)
    n_cells, n_markers = 40, 4
    X = rng.normal(size=(n_cells, n_markers))
    var_names = [f"marker{i}" for i in range(n_markers)]
    obs = pd.DataFrame(
        {"leiden": ["0"] * 38 + ["1"] * 2},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))

    axes = marker_dotplot(
        adata, var_names, groupby="leiden", dendrogram=False, min_cluster_frac=0.1
    )
    ax_size = axes["size"]
    assert ax_size is not None

    order = [t.get_text() for t in axes["dots"].get_yticklabels()]
    tiny_idx = order.index("1")
    normal_idx = order.index("0")
    bars = ax_size.patches
    assert bars[tiny_idx].get_facecolor() != bars[normal_idx].get_facecolor()


# ---------------------------------------------------------------------------
# marker_dotplot at high cluster counts: dendrogram/colorbar/legend/size-panel
# layout bugs that only manifest once ax_dots gets tall.
# ---------------------------------------------------------------------------


def _make_many_cluster_adata(n_clusters, n_cells=3000, n_markers=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_cells, n_markers))
    var_names = [f"marker{i}" for i in range(n_markers)]
    obs = pd.DataFrame(
        {"leiden": rng.integers(0, n_clusters, size=n_cells).astype(str)},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    return anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))


def test_marker_dotplot_dendrogram_ylim_matches_scipy_leaf_scale():
    adata = _make_many_cluster_adata(n_clusters=30)
    n_groups = adata.obs["leiden"].nunique()
    axes = marker_dotplot(
        adata, list(adata.var_names), groupby="leiden", dendrogram=True
    )
    assert axes["dendrogram"].get_ylim() == (0.0, 10.0 * n_groups)


def test_marker_dotplot_colorbar_spans_full_row_height():
    adata = _make_many_cluster_adata(n_clusters=30)
    axes = marker_dotplot(
        adata, list(adata.var_names), groupby="leiden", dendrogram=False
    )
    dots_height = axes["dots"].get_position().height
    cbar_height = axes["colorbar"].get_position().height
    assert cbar_height == pytest.approx(dots_height, rel=0.01)


def test_marker_dotplot_legend_row_height_independent_of_cluster_count():
    adata_small = _make_many_cluster_adata(n_clusters=5)
    adata_large = _make_many_cluster_adata(n_clusters=100)

    def _legend_height_in(adata):
        axes = marker_dotplot(
            adata, list(adata.var_names), groupby="leiden", dendrogram=False
        )
        fig = axes["dots"].figure
        fig_h = fig.get_size_inches()[1]
        return axes["legend"].get_position().height * fig_h

    h_small = _legend_height_in(adata_small)
    h_large = _legend_height_in(adata_large)
    assert h_small == pytest.approx(h_large, rel=0.05)


def test_marker_dotplot_size_panel_hides_duplicate_labels():
    adata = _make_many_cluster_adata(n_clusters=30)
    axes = marker_dotplot(
        adata, list(adata.var_names), groupby="leiden", dendrogram=False
    )
    assert axes["size"].yaxis.get_tick_params()["labelleft"] is False


def test_marker_dotplot_highlight_cluster_without_spatial_region():
    adata = _make_clustering_adata()
    axes = marker_dotplot(
        adata,
        list(adata.var_names),
        groupby="leiden",
        dendrogram=False,
        highlight_cluster="1",
    )
    assert axes["spatial"] is None
    # a highlight band (axhspan) was added as a patch on the dots axes
    assert len(axes["dots"].patches) >= 1


def test_marker_dotplot_spatial_region_requires_highlight_cluster():
    adata = _make_clustering_adata()
    with pytest.raises(ValueError, match="highlight_cluster"):
        marker_dotplot(
            adata,
            list(adata.var_names),
            groupby="leiden",
            dendrogram=False,
            spatial_region="reg1",
        )


def test_marker_dotplot_spatial_panel_shows_only_highlighted_cluster_in_region():
    adata = _make_clustering_adata()
    n_expected = int(
        ((adata.obs["leiden"] == "1") & (adata.obs["unique_region"] == "reg1")).sum()
    )
    axes = marker_dotplot(
        adata,
        list(adata.var_names),
        groupby="leiden",
        dendrogram=False,
        highlight_cluster="1",
        spatial_region="reg1",
    )
    ax_spatial = axes["spatial"]
    assert ax_spatial is not None
    # two scatter collections: grey background + highlighted cluster
    collections = ax_spatial.collections
    assert len(collections) == 2
    assert len(collections[1].get_offsets()) == n_expected


def test_marker_dotplot_spatial_panel_switches_with_region():
    adata = _make_clustering_adata()
    axes_reg1 = marker_dotplot(
        adata,
        list(adata.var_names),
        groupby="leiden",
        dendrogram=False,
        highlight_cluster="1",
        spatial_region="reg1",
    )
    axes_reg2 = marker_dotplot(
        adata,
        list(adata.var_names),
        groupby="leiden",
        dendrogram=False,
        highlight_cluster="1",
        spatial_region="reg2",
    )
    n1 = len(axes_reg1["spatial"].collections[1].get_offsets())
    n2 = len(axes_reg2["spatial"].collections[1].get_offsets())
    total = int((adata.obs["leiden"] == "1").sum())
    assert n1 + n2 == total


def test_cluster_composition_heatmap_smoke():
    df = pd.DataFrame(
        {
            "cluster": ["0", "0", "1", "1", "2"],
            "cell_type": ["T cell", "B cell", "T cell", "T cell", "B cell"],
        }
    )
    ax, table = cluster_composition_heatmap(
        df, cluster_column="cluster", celltype_column="cell_type", return_data=True
    )
    assert ax is not None
    assert table.shape == (3, 2)


def test_cluster_composition_heatmap_accepts_adata():
    adata = _make_clustering_adata()
    adata.obs["cell_type"] = ["a", "b"] * (adata.n_obs // 2)
    ax = cluster_composition_heatmap(
        adata, cluster_column="leiden", celltype_column="cell_type"
    )
    assert ax is not None


def test_correlation_heatmap_smoke():
    markers = ["CD3", "CD20", "CD8"]
    cmat = pd.DataFrame(np.eye(3), index=markers, columns=markers)
    ax, data = correlation_heatmap(cmat, return_data=True)
    assert ax is not None
    assert data.shape == (3, 3)


def test_classification_report_heatmap_from_labels():
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "b", "b", "b", "c", "a"]
    ax, report_df = classification_report_heatmap(
        y_true=y_true, y_pred=y_pred, return_data=True
    )
    assert ax is not None
    assert "precision" in report_df.columns


def test_classification_report_heatmap_requires_labels_or_report():
    with pytest.raises(ValueError):
        classification_report_heatmap()


def test_confusion_matrix_heatmap_smoke():
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "b", "b", "b", "c", "a"]
    ax, cm_df = confusion_matrix_heatmap(y_true, y_pred, return_data=True)
    assert ax is not None
    assert cm_df.shape == (3, 3)


def test_compensation_qc_smoke():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "CD20": [0.5, 0.6, 0.7, 0.8],
            "CD20_compensated": [0.4, 0.5, 0.6, 0.7],
        }
    )
    fig = compensation_qc(df, markers="CD20")
    assert fig is not None


def _make_masks_adata():
    # Two single-pixel labeled masks placed asymmetrically (row != col) so
    # that a (x, y)/(y, x) ordering mix-up between the KDTree and the
    # regionprops centroids produces a detectably wrong nearest-neighbor
    # match instead of accidentally still working out. plot_masks still uses
    # centroid/KDTree proximity matching, so this layout remains relevant for
    # its tests; the "label" column (added for catplot's exact-join tests)
    # matches each mask's own pixel value and is simply unused by plot_masks.
    masks = np.zeros((10, 30), dtype=int)
    masks[2, 22] = 1  # centroid (row=2, col=22)
    masks[7, 2] = 2  # centroid (row=7, col=2)

    obs = pd.DataFrame(
        {
            "x": [22.0, 2.0],
            "y": [2.0, 7.0],
            "cell_type": ["A", "B"],
            "label": [1, 2],
            "unique_region": ["region1", "region1"],
        }
    )
    adata = anndata.AnnData(X=np.zeros((2, 1)), obs=obs)
    adata.uns["masks"] = {"region1": masks}
    return adata


def _make_multi_region_masks_adata():
    # Two distinct real regions ("regA", "regB") sharing one coarser panel
    # value ("condition1"), each with its own mask -- exercises catplot's
    # side-by-side tiling when a panel spans multiple real regions, instead
    # of overlaying them.
    masks_a = np.zeros((5, 6), dtype=int)
    masks_a[2, 3] = 1
    masks_b = np.zeros((4, 8), dtype=int)
    masks_b[1, 5] = 1

    obs = pd.DataFrame(
        {
            "x": [3.0, 5.0],
            "y": [2.0, 1.0],
            "cell_type": ["A", "B"],
            "label": [1, 1],
            "unique_region": ["regA", "regB"],
            "condition": ["condition1", "condition1"],
        }
    )
    adata = anndata.AnnData(X=np.zeros((2, 1)), obs=obs)
    adata.uns["masks"] = {"regA": masks_a, "regB": masks_b}
    return adata


def _make_mixed_region_adata():
    # regA has a mask; regB does not -- exercises per-region fallback within
    # one tiled panel (not a fully-fallback panel).
    masks_a = np.zeros((5, 6), dtype=int)
    masks_a[2, 3] = 1

    obs = pd.DataFrame(
        {
            "x": [3.0, 100.0, 105.0],
            "y": [2.0, 10.0, 12.0],
            "cell_type": ["A", "B", "B"],
            "label": [1, -1, -1],
            "unique_region": ["regA", "regB", "regB"],
            "condition": ["condition1", "condition1", "condition1"],
        }
    )
    adata = anndata.AnnData(X=np.zeros((3, 1)), obs=obs)
    adata.uns["masks"] = {"regA": masks_a}
    return adata


def test_plot_masks_smoke():
    # Regression test: plot_masks used regionprops, cKDTree, and
    # find_boundaries without importing them, raising NameError on any
    # real call.
    adata = _make_masks_adata()
    plot_masks(
        adata,
        color="cell_type",
        unique_region="unique_region",
        subset="region1",
        palette={"A": (1.0, 0.0, 0.0), "B": (0.0, 0.0, 1.0)},
    )
    matplotlib.pyplot.close("all")


def test_plot_masks_centroid_coordinate_order():
    # Regression test: the KDTree used to be built from (X, Y) columns but
    # queried with regionprops centroids, which are (row, col) = (Y, X).
    # With asymmetric mask placement that mismatch assigns the wrong
    # cell_type/color to each mask. Verify each mask pixel ends up colored
    # by the cell placed at its own centroid, not the other one.
    adata = _make_masks_adata()
    palette = {"A": (1.0, 0.0, 0.0), "B": (0.0, 0.0, 1.0)}
    plot_masks(
        adata,
        color="cell_type",
        unique_region="unique_region",
        subset="region1",
        palette=palette,
        outline=False,
        scalebar=False,
    )
    fig = matplotlib.pyplot.gcf()
    ax = fig.axes[0]
    rotated = np.asarray(ax.images[0].get_array())
    colored_masks = np.rot90(rotated, -1)  # undo plot_masks' np.rot90()

    np.testing.assert_allclose(colored_masks[2, 22, :3], palette["A"], atol=1e-6)
    np.testing.assert_allclose(colored_masks[7, 2, :3], palette["B"], atol=1e-6)
    matplotlib.pyplot.close("all")


def test_catplot_masks_fill_smoke():
    # catplot should auto-detect adata.uns["masks"] and fill mask shapes
    # instead of plotting dots, mirroring plot_masks.
    adata = _make_masks_adata()
    catplot(
        adata,
        color="cell_type",
        unique_region="unique_region",
        X="x",
        Y="y",
        subset="region1",
        palette={"A": (1.0, 0.0, 0.0), "B": (0.0, 0.0, 1.0)},
    )
    matplotlib.pyplot.close("all")


def test_catplot_masks_string_palette():
    # Regression test: a string palette name (e.g. "tab20") is valid,
    # documented catplot usage -- sns.scatterplot resolves it natively -- but
    # the mask-fill/tiling code assumed palette was always a dict and raised
    # AttributeError: 'str' object has no attribute 'get'.
    adata = _make_masks_adata()
    catplot(
        adata,
        color="cell_type",
        unique_region="unique_region",
        X="x",
        Y="y",
        subset="region1",
        palette="tab20",
    )
    matplotlib.pyplot.close("all")


def test_catplot_subset_does_not_crash():
    # Regression test: passing subset=<region> used to raise
    # AttributeError: 'numpy.ndarray' object has no attribute 'sort_values'
    # because df[unique_region].unique() returns a plain ndarray, which
    # has no .sort_values() (a pandas Series method).
    adata = _make_masks_adata()
    catplot(
        adata,
        color="cell_type",
        unique_region="unique_region",
        X="x",
        Y="y",
        subset="region1",
        palette={"A": (1.0, 0.0, 0.0), "B": (0.0, 0.0, 1.0)},
    )
    matplotlib.pyplot.close("all")


def test_catplot_masks_coordinate_order():
    # Each mask pixel should end up colored by the cell that names its exact
    # label (an exact join via the "label" column, not centroid/proximity
    # matching). catplot's mask fill uses origin="lower" with no rotation, so
    # no undo-rotation is needed to check the raw rendered array.
    adata = _make_masks_adata()
    palette = {"A": (1.0, 0.0, 0.0), "B": (0.0, 0.0, 1.0)}
    catplot(
        adata,
        color="cell_type",
        unique_region="unique_region",
        X="x",
        Y="y",
        subset="region1",
        palette=palette,
        mask_outline=False,
    )
    fig = matplotlib.pyplot.gcf()
    ax = fig.axes[0]
    colored_masks = np.asarray(ax.images[0].get_array())

    np.testing.assert_allclose(colored_masks[2, 22, :3], palette["A"], atol=1e-6)
    np.testing.assert_allclose(colored_masks[7, 2, :3], palette["B"], atol=1e-6)
    matplotlib.pyplot.close("all")


def test_catplot_masks_multi_region_tiling():
    # Two real regions sharing one coarser panel value ("condition") must be
    # tiled side by side, not overlaid -- per explicit requirement: region
    # lookup (region_col) is decoupled from panel grouping (unique_region=).
    adata = _make_multi_region_masks_adata()
    palette = {"A": (1.0, 0.0, 0.0), "B": (0.0, 0.0, 1.0)}
    catplot(
        adata,
        color="cell_type",
        unique_region="condition",
        region_col="unique_region",
        X="x",
        Y="y",
        palette=palette,
        mask_outline=False,
    )
    fig = matplotlib.pyplot.gcf()
    ax = fig.axes[0]
    assert len(ax.images) == 2

    ext0 = ax.images[0].get_extent()  # (left, right, bottom, top)
    ext1 = ax.images[1].get_extent()
    assert ext0[1] <= ext1[0] or ext1[1] <= ext0[0]  # non-overlapping in x

    arr0 = np.asarray(ax.images[0].get_array())
    arr1 = np.asarray(ax.images[1].get_array())
    np.testing.assert_allclose(
        arr0[2, 3, :3], palette["A"], atol=1e-6
    )  # regA's own pixel
    np.testing.assert_allclose(
        arr1[1, 5, :3], palette["B"], atol=1e-6
    )  # regB's own pixel
    matplotlib.pyplot.close("all")


def test_catplot_masks_mixed_region_fallback():
    # regA has a mask and renders filled; regB has none and falls back to a
    # scatter, tiled next to regA rather than left at its own raw (and here
    # far-away, but in general potentially overlapping) coordinates.
    adata = _make_mixed_region_adata()
    palette = {"A": (1.0, 0.0, 0.0), "B": (0.0, 0.0, 1.0)}
    catplot(
        adata,
        color="cell_type",
        unique_region="condition",
        region_col="unique_region",
        X="x",
        Y="y",
        palette=palette,
        mask_outline=False,
    )
    fig = matplotlib.pyplot.gcf()
    ax = fig.axes[0]
    assert len(ax.images) == 1
    assert len(ax.collections) == 1

    ext = ax.images[0].get_extent()
    offsets = ax.collections[0].get_offsets()
    assert offsets[:, 0].min() >= ext[1] - 1e-6
    matplotlib.pyplot.close("all")


def test_catplot_internal_scatter_fallback_invert_y():
    # A region with region_col/label_col present but no matching entry in
    # adata.uns["masks"] falls back to catplot's own internal tiled scatter
    # (not sns.scatterplot) -- invert_y must still flip it correctly there.
    # Tiling re-baselines each tile's y to start at 0 (so mask and scatter
    # tiles share one consistent vertical scale), so the effect to check is
    # a reversal of each point's *relative* vertical order, not its sign.
    def _offsets_for(invert_y):
        obs = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "cell_type": ["a", "b"],
                "label": [1, 2],
                "unique_region": ["regC", "regC"],
            }
        )
        adata = anndata.AnnData(X=np.zeros((2, 1)), obs=obs)
        adata.uns["masks"] = {"some_other_region": np.zeros((3, 3), dtype=int)}
        catplot(
            adata,
            color="cell_type",
            unique_region="unique_region",
            X="x",
            Y="y",
            invert_y=invert_y,
            palette={"a": (1.0, 0.0, 0.0), "b": (0.0, 0.0, 1.0)},
        )
        fig = matplotlib.pyplot.gcf()
        ax = fig.axes[0]
        assert len(ax.images) == 0
        offsets = ax.collections[0].get_offsets()
        matplotlib.pyplot.close("all")
        return offsets

    offsets_normal = _offsets_for(False)
    offsets_inverted = _offsets_for(True)
    # row0 (y=1.0) sits below row1 (y=2.0) normally, above it once inverted.
    assert offsets_normal[0, 1] < offsets_normal[1, 1]
    assert offsets_inverted[0, 1] > offsets_inverted[1, 1]


def test_catplot_masks_fallback_to_scatter():
    # A region with no matching key in adata.uns["masks"] (or no "masks" key
    # in .uns at all) should still render via the existing dot scatter path.
    adata = _make_adata()
    adata.obs["cell_type"] = ["a", "b", "a", "b"]
    catplot(adata, color="cell_type", unique_region="unique_region", X="x", Y="y")
    matplotlib.pyplot.close("all")

    adata2 = _make_masks_adata()
    adata2.uns["masks"] = {"some_other_region": adata2.uns["masks"]["region1"]}
    catplot(
        adata2,
        color="cell_type",
        unique_region="unique_region",
        X="x",
        Y="y",
        palette={"A": (1.0, 0.0, 0.0), "B": (0.0, 0.0, 1.0)},
    )
    matplotlib.pyplot.close("all")


def _make_expression_masks_adata():
    # Three labeled single-pixel cells, but only two of them present in obs --
    # label 3 stands in for a cell dropped upstream by filter_data/remove_noise,
    # whose pixels must fall back to the background color.
    masks = np.zeros((10, 30), dtype=int)
    masks[2, 22] = 1
    masks[7, 2] = 2
    masks[5, 10] = 3

    obs = pd.DataFrame(
        {
            "label": [1, 2],
            "unique_region": ["region1", "region1"],
            "area": [10.0, 40.0],
        },
        index=["cell1", "cell2"],
    )
    adata = anndata.AnnData(X=np.array([[1.5, 0.25], [-2.0, 3.0]]), obs=obs)
    adata.var_names = ["CD21", "CD8"]
    adata.uns["masks"] = {"region1": masks}
    return adata


def _make_expression_masks_two_regions_adata():
    masks_a = np.zeros((6, 8), dtype=int)
    masks_a[2, 3] = 1
    masks_b = np.zeros((6, 8), dtype=int)
    masks_b[4, 5] = 1

    obs = pd.DataFrame(
        {
            "label": [1, 1],
            "unique_region": ["regA", "regB"],
        },
        index=["cell1", "cell2"],
    )
    adata = anndata.AnnData(X=np.array([[1.0], [5.0]]), obs=obs)
    adata.var_names = ["CD21"]
    adata.uns["masks"] = {"regA": masks_a, "regB": masks_b}
    return adata


def test_expression_on_masks_smoke():
    adata = _make_expression_masks_adata()
    fig = expression_on_masks(adata, marker="CD21", subset="region1")
    assert fig is not None
    assert len(fig.axes[0].images) == 1
    matplotlib.pyplot.close("all")


def test_expression_on_masks_paints_each_cells_own_value():
    # Every pixel of a label carries that cell's expression; background and
    # cells missing from obs (label 3 here) stay masked so the colormap's
    # "bad" color shows through.
    adata = _make_expression_masks_adata()
    fig = expression_on_masks(adata, marker="CD21", subset="region1", scalebar=False)
    image = np.ma.masked_invalid(fig.axes[0].images[0].get_array())

    assert image[2, 22] == pytest.approx(1.5)
    assert image[7, 2] == pytest.approx(-2.0)
    assert image[5, 10] is np.ma.masked  # filtered-out cell
    assert image[0, 0] is np.ma.masked  # background
    matplotlib.pyplot.close("all")


def test_expression_on_masks_uses_layer():
    adata = _make_expression_masks_adata()
    adata.layers["scaled"] = adata.X * 10
    fig = expression_on_masks(
        adata, marker="CD21", subset="region1", layer="scaled", scalebar=False
    )
    image = fig.axes[0].images[0].get_array()
    assert image[2, 22] == pytest.approx(15.0)
    matplotlib.pyplot.close("all")


def test_expression_on_masks_accepts_obs_column():
    adata = _make_expression_masks_adata()
    fig = expression_on_masks(adata, marker="area", subset="region1", scalebar=False)
    image = fig.axes[0].images[0].get_array()
    assert image[2, 22] == pytest.approx(10.0)
    assert image[7, 2] == pytest.approx(40.0)
    matplotlib.pyplot.close("all")


def test_expression_on_masks_rejects_categorical_obs_column():
    adata = _make_expression_masks_adata()
    adata.obs["cell_type"] = ["A", "B"]
    with pytest.raises(ValueError, match="plot_masks"):
        expression_on_masks(adata, marker="cell_type", subset="region1")


def test_expression_on_masks_all_regions_share_scale():
    adata = _make_expression_masks_two_regions_adata()
    fig = expression_on_masks(adata, marker="CD21", scalebar=False)
    panels = [ax for ax in fig.axes if ax.images]
    assert len(panels) == 2
    assert panels[0].images[0].get_clim() == panels[1].images[0].get_clim()
    matplotlib.pyplot.close("all")


def test_expression_on_masks_per_region_scale_when_not_shared():
    adata = _make_expression_masks_two_regions_adata()
    fig = expression_on_masks(adata, marker="CD21", shared_scale=False, scalebar=False)
    panels = [ax for ax in fig.axes if ax.images]
    assert panels[0].images[0].get_clim() != panels[1].images[0].get_clim()
    matplotlib.pyplot.close("all")


def test_expression_on_masks_multiple_markers_one_panel_each():
    adata = _make_expression_masks_adata()
    fig = expression_on_masks(
        adata, marker=["CD21", "CD8"], subset="region1", scalebar=False
    )
    panels = [ax for ax in fig.axes if ax.images]
    assert len(panels) == 2
    matplotlib.pyplot.close("all")


def test_expression_on_masks_unknown_marker_raises():
    adata = _make_expression_masks_adata()
    with pytest.raises(ValueError, match="adata.var_names"):
        expression_on_masks(adata, marker="not_a_marker", subset="region1")


def test_expression_on_masks_requires_attached_masks():
    adata = _make_expression_masks_adata()
    del adata.uns["masks"]
    with pytest.raises(ValueError, match="hf_attach_masks"):
        expression_on_masks(adata, marker="CD21")


def test_expression_on_masks_unknown_region_raises():
    adata = _make_expression_masks_adata()
    with pytest.raises(ValueError, match="no mask found"):
        expression_on_masks(adata, marker="CD21", subset="region_nope")


def test_expression_on_masks_label_out_of_range_raises():
    # A label larger than anything in the mask means obs and mask do not come
    # from the same region -- fail loudly instead of indexing past the lookup.
    # Caught by the upfront mask-ownership check now (only 1 of 2 labels found in
    # the mask), before ever reaching the per-panel image-building code.
    adata = _make_expression_masks_adata()
    adata.obs["label"] = [1, 9999]
    with pytest.raises(ValueError, match="do not look like they belong"):
        expression_on_masks(adata, marker="CD21", subset="region1")


# ---------------------------------------------------------------------------
# cluster_qc
# ---------------------------------------------------------------------------


def _make_qc_adata(n_cells=300, n_clusters=3, seed=0):
    """Post-clustering AnnData where cluster "c0" is deliberately made of
    small, dim cells -- the segmentation-fragment signature cluster_qc exists
    to make visible."""
    rng = np.random.default_rng(seed)
    labels = np.array([f"c{i % n_clusters}" for i in range(n_cells)])
    small = labels == "c0"

    area = rng.normal(500, 40, n_cells)
    dapi = rng.normal(900, 60, n_cells)
    area[small] = rng.normal(200, 20, small.sum())
    dapi[small] = rng.normal(300, 30, small.sum())

    obs = pd.DataFrame(
        {
            "leiden": pd.Categorical(labels),
            "area": area,
            "DAPI": dapi,
            "x": rng.uniform(0, 100, n_cells),
            "y": rng.uniform(0, 100, n_cells),
            "label": np.arange(n_cells),
            "unique_region": rng.choice(["R0", "R1"], n_cells),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    return anndata.AnnData(
        X=rng.normal(size=(n_cells, 4)),
        obs=obs,
        var=pd.DataFrame(index=[f"marker{i}" for i in range(4)]),
    )


def test_cluster_qc_smoke():
    adata = _make_qc_adata()
    fig = cluster_qc(adata, "leiden")
    assert fig.__class__.__name__ == "Figure"
    assert len(fig.axes) == 2


def test_cluster_qc_default_features_are_measured_obs_columns():
    # Coordinates, the per-cell `label` ID and the categorical region column
    # must not become panels.
    adata = _make_qc_adata()
    fig = cluster_qc(adata, "leiden")
    ylabels = [ax.get_ylabel() for ax in fig.axes]
    assert ylabels == ["area", "DAPI"]


def test_cluster_qc_one_box_per_cluster():
    adata = _make_qc_adata(n_clusters=5)
    fig = cluster_qc(adata, "leiden")
    for ax in fig.axes:
        assert len([p for p in ax.patches]) == 5


def test_cluster_qc_highlight_colors_only_that_cluster():
    adata = _make_qc_adata()
    fig = cluster_qc(adata, "leiden", highlight_cluster="c0")
    colors = [p.get_facecolor() for p in fig.axes[0].patches]
    assert colors[0] != colors[1]
    assert len({c for c in colors[1:]}) == 1


def test_cluster_qc_highlight_accepts_non_string_label():
    adata = _make_qc_adata()
    adata.obs["leiden"] = pd.Categorical(
        [lbl.replace("c", "") for lbl in adata.obs["leiden"].astype(str)]
    )
    fig = cluster_qc(adata, "leiden", highlight_cluster=0)
    colors = [p.get_facecolor() for p in fig.axes[0].patches]
    assert colors[0] != colors[1]


def test_cluster_qc_reference_line_is_overall_median():
    adata = _make_qc_adata()
    fig = cluster_qc(adata, "leiden")
    dashed = [
        line
        for line in fig.axes[0].get_lines()
        if line.get_linestyle() == "--" and len(set(line.get_ydata())) == 1
    ]
    assert len(dashed) == 1
    assert dashed[0].get_ydata()[0] == pytest.approx(adata.obs["area"].median())


def test_cluster_qc_return_data():
    adata = _make_qc_adata()
    fig, summary = cluster_qc(adata, "leiden", return_data=True)
    assert fig.__class__.__name__ == "Figure"
    assert list(summary.index) == ["c0", "c1", "c2", "all cells"]
    assert summary.loc["all cells", "n_cells"] == adata.n_obs
    # the deliberately-small cluster must read below the overall median
    assert summary.loc["c0", "area"] < summary.loc["all cells", "area"]
    assert summary.loc["c0", "DAPI"] < summary.loc["all cells", "DAPI"]


def test_cluster_qc_explicit_features_respected():
    adata = _make_qc_adata()
    fig = cluster_qc(adata, "leiden", features=["area"])
    assert len(fig.axes) == 1
    assert fig.axes[0].get_ylabel() == "area"


def test_cluster_qc_unknown_feature_raises():
    adata = _make_qc_adata()
    with pytest.raises(ValueError, match="Not in adata.obs"):
        cluster_qc(adata, "leiden", features=["nope"])


def test_cluster_qc_unknown_cluster_key_raises():
    adata = _make_qc_adata()
    with pytest.raises(ValueError, match="not a column of adata.obs"):
        cluster_qc(adata, "nope")


def test_cluster_qc_no_numeric_features_raises():
    adata = _make_qc_adata()
    adata.obs = adata.obs[["leiden", "x", "y", "label", "unique_region"]]
    with pytest.raises(ValueError, match="No numeric adata.obs column"):
        cluster_qc(adata, "leiden")


def test_cluster_qc_savefig_writes_file(tmp_path):
    adata = _make_qc_adata()
    cluster_qc(
        adata,
        "leiden",
        savefig=True,
        output_dir=str(tmp_path),
        output_fname="qc",
    )
    assert len(list(tmp_path.glob("qc_cluster_qc.*"))) == 1


# ---------------------------------------------------------------------------
# _qc_default_flags
# ---------------------------------------------------------------------------


def _make_flags_adata():
    obs = pd.DataFrame(
        {
            "leiden": pd.Categorical([f"c{i % 3}" for i in range(30)]),
            "area": np.arange(30, dtype=float),  # numeric, not boolean
            "qc_flag_noise": [i % 5 == 0 for i in range(30)],
            "qc_flag_low_dapi": [i % 7 == 0 for i in range(30)],
            "x": np.arange(30, dtype=float),  # excluded coordinate
            "label": np.arange(30),  # excluded identifier
        }
    )
    return anndata.AnnData(X=np.zeros((30, 1)), obs=obs)


def test_qc_default_flags_picks_only_boolean_columns():
    adata = _make_flags_adata()
    assert _qc_default_flags(adata) == ["qc_flag_noise", "qc_flag_low_dapi"]


def test_qc_default_flags_excludes_qc_excluded_obs():
    adata = _make_flags_adata()
    adata.obs["label"] = adata.obs["label"].astype(
        bool
    )  # would match if not excluded by name
    assert "label" not in _qc_default_flags(adata)


# ---------------------------------------------------------------------------
# _qc_flag_label_overrides
# ---------------------------------------------------------------------------


def test_qc_flag_label_overrides_strips_prefix():
    assert _qc_flag_label_overrides(["qc_flag_low_dapi", "qc_flag_large_size"]) == {
        "qc_flag_low_dapi": "low_dapi",
        "qc_flag_large_size": "large_size",
    }


def test_qc_flag_label_overrides_renames_noise_to_noisy():
    assert _qc_flag_label_overrides(["qc_flag_noise"]) == {"qc_flag_noise": "noisy"}


def test_qc_flag_label_overrides_skips_names_without_prefix():
    assert _qc_flag_label_overrides(["custom_flag", "qc_flag_any"]) == {
        "qc_flag_any": "any"
    }


# ---------------------------------------------------------------------------
# _get_clustering_report_markers
# ---------------------------------------------------------------------------


def _make_report_adata(report_steps=None, markers=("M0", "M1", "M2")):
    obs = pd.DataFrame({"leiden_1": pd.Categorical(["0", "1"] * 5)})
    a = anndata.AnnData(
        X=np.zeros((10, len(markers))), obs=obs, var=pd.DataFrame(index=list(markers))
    )
    if report_steps is not None:
        import json

        a.uns["clustering_report"] = json.dumps(report_steps)
    return a


def test_get_clustering_report_markers_returns_recorded_list():
    report = [
        {"action": "clustering", "key_added": "leiden_1", "marker_list": ["M0", "M1"]}
    ]
    adata = _make_report_adata(report)
    assert _get_clustering_report_markers(adata, "leiden_1") == ["M0", "M1"]


def test_get_clustering_report_markers_none_recorded_falls_back_to_var_names():
    report = [{"action": "clustering", "key_added": "leiden_1", "marker_list": None}]
    adata = _make_report_adata(report)
    assert _get_clustering_report_markers(adata, "leiden_1") == ["M0", "M1", "M2"]


def test_get_clustering_report_markers_no_report_returns_none():
    adata = _make_report_adata(None)
    assert _get_clustering_report_markers(adata, "leiden_1") is None


def test_get_clustering_report_markers_no_matching_key_added_returns_none():
    report = [{"action": "clustering", "key_added": "other_col", "marker_list": ["M0"]}]
    adata = _make_report_adata(report)
    assert _get_clustering_report_markers(adata, "leiden_1") is None


def test_get_clustering_report_markers_subclustering_action_returns_none():
    report = [{"action": "subclustering", "key_added": "leiden_1"}]
    adata = _make_report_adata(report)
    assert _get_clustering_report_markers(adata, "leiden_1") is None


def test_get_clustering_report_markers_uses_most_recent_step():
    report = [
        {"action": "clustering", "key_added": "leiden_1", "marker_list": ["M0"]},
        {"action": "clustering", "key_added": "leiden_1", "marker_list": ["M0", "M1"]},
    ]
    adata = _make_report_adata(report)
    assert _get_clustering_report_markers(adata, "leiden_1") == ["M0", "M1"]


def test_get_clustering_report_markers_malformed_json_returns_none_never_raises():
    adata = _make_report_adata(None)
    adata.uns["clustering_report"] = "not valid json{{{"
    assert _get_clustering_report_markers(adata, "leiden_1") is None


def test_extended_dotplot_warns_on_marker_subset_via_refactored_lookup():
    # End-to-end check that _warn_on_clustering_marker_mismatch (which now delegates to
    # _get_clustering_report_markers) still fires correctly after the refactor.
    report = [
        {
            "action": "clustering",
            "key_added": "leiden_1",
            "marker_list": ["M0", "M1", "M2"],
        }
    ]
    adata = _make_report_adata(report)
    with pytest.warns(UserWarning, match="subset"):
        extended_dotplot(adata, "leiden_1", clustering_markers=["M0"])


def test_extended_dotplot_no_warning_on_exact_marker_match():
    report = [
        {
            "action": "clustering",
            "key_added": "leiden_1",
            "marker_list": ["M0", "M1", "M2"],
        }
    ]
    adata = _make_report_adata(report)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        extended_dotplot(adata, "leiden_1", clustering_markers=["M0", "M1", "M2"])


def _dots_ax(fig, title="Clustering markers"):
    for ax in fig.axes:
        if ax.get_title() == title:
            return ax
    raise AssertionError(f"no {title!r} axis found")


def _make_dotplot_scale_adata():
    """3 clusters with deliberately distinct per-cluster means for one marker (0, 1, 10) --
    "2" is a single-cell, ~0.7%-of-total outlier cluster (below the default 1% min_cluster_frac)
    so flag_small_clusters has something real to flag."""
    rng = np.random.default_rng(0)

    def block(n, mean):
        return rng.normal(mean, 0.01, size=(n, 1))

    X = np.vstack([block(74, 0.0), block(74, 1.0), block(1, 10.0)]).astype(np.float32)
    n = 149
    obs = pd.DataFrame(
        {
            "leiden": ["0"] * 74 + ["1"] * 74 + ["2"] * 1,
            "x": rng.uniform(0, 100, n),
            "y": rng.uniform(0, 100, n),
            "unique_region": ["reg1"] * n,
        }
    )
    var = pd.DataFrame(index=["marker0"])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.obs["leiden"] = adata.obs["leiden"].astype("category")
    return adata


def test_extended_dotplot_invalid_color_scale_raises():
    adata = _make_dotplot_scale_adata()
    with pytest.raises(ValueError, match="color_scale"):
        extended_dotplot(
            adata, "leiden", clustering_markers=["marker0"], color_scale="bogus"
        )


def test_extended_dotplot_minmax_is_bounded_0_1():
    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=False,
        color_scale="minmax",
    )
    colors = _dots_ax(fig).collections[0].get_array()
    assert colors.min() == pytest.approx(0.0, abs=1e-6)
    assert colors.max() == pytest.approx(1.0, abs=1e-6)
    assert _dots_ax(fig).collections[0].get_clim() == (0.0, 1.0)


def test_extended_dotplot_defaults_to_zscore():
    """color_scale defaults to "zscore" (see its docstring: sp.pp.format's own default already
    per-marker z-scores adata.X, so re-z-scoring cluster means preserves that comparability
    across markers, unlike "minmax" which would discard it)."""
    adata = _make_dotplot_scale_adata()
    fig_default = extended_dotplot(
        adata, "leiden", clustering_markers=["marker0"], show_totals=False
    )
    fig_zscore = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=False,
        color_scale="zscore",
    )
    np.testing.assert_array_equal(
        _dots_ax(fig_default).collections[0].get_array(),
        _dots_ax(fig_zscore).collections[0].get_array(),
    )
    assert _dots_ax(fig_default).collections[0].get_clim() == (-3.0, 3.0)


def test_extended_dotplot_zscore_uses_diverging_cmap_and_symmetric_range():
    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=False,
        color_scale="zscore",
    )
    coll = _dots_ax(fig).collections[0]
    assert coll.get_cmap().name == "RdBu_r"  # SPACEcStyle's default diverging_cmap
    assert coll.get_clim() == (-3.0, 3.0)  # SPACEcStyle's default heatmap_vmin/vmax
    # standardized values, not the minmax path's forced [0, 1] bounds
    colors = coll.get_array()
    assert not np.isclose(colors.max(), 1.0)
    assert colors.min() < 0  # the below-average cluster reads negative, not 0


def test_extended_dotplot_flag_small_clusters_colors_tiny_cluster_bar_red():
    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=True,
        flag_small_clusters=True,
    )
    bar_colors = [p.get_facecolor() for p in _dots_ax(fig, "Cluster size").patches]
    # clusters "0"/"1" are ~50% each; "2" is a single cell (~0.7%, below the 1% default)
    assert bar_colors[0] == bar_colors[1]
    assert bar_colors[2] != bar_colors[0]


def test_extended_dotplot_flag_small_clusters_off_by_default():
    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata, "leiden", clustering_markers=["marker0"], show_totals=True
    )
    bar_colors = [p.get_facecolor() for p in _dots_ax(fig, "Cluster size").patches]
    assert bar_colors[0] == bar_colors[1] == bar_colors[2]


def test_extended_dotplot_highlighted_bar_uses_own_color_not_flag_red():
    """A highlighted cluster's totals-bar row shows its own established color (from `palette`),
    not the small-cluster-flag red -- the two signals are independent and must not collide even
    when the highlighted cluster IS the flagged small one (cluster "2" here is both)."""
    import matplotlib.colors as mcolors

    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=True,
        flag_small_clusters=True,
        highlight_cluster="2",
        palette={"0": "#000000", "1": "#000000", "2": "#123456"},
    )
    bar_colors = [p.get_facecolor() for p in _dots_ax(fig, "Cluster size").patches]
    assert bar_colors[2] == mcolors.to_rgba("#123456")


def test_extended_dotplot_highlighted_bar_gets_black_outline_when_light():
    """A pale highlighted-cluster color gets a black outline so it stays visible against the
    white background; a dark one doesn't need it (see _is_light_color)."""
    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=True,
        highlight_cluster=["0", "1"],
        palette={"0": "#ffffff", "1": "#000000", "2": "#888888"},
    )
    light_bar, dark_bar = _dots_ax(fig, "Cluster size").patches[0:2]
    assert light_bar.get_edgecolor()[:3] == (0.0, 0.0, 0.0)
    assert light_bar.get_linewidth() > 0
    assert dark_bar.get_linewidth() == 0


def test_extended_dotplot_highlight_cluster_accepts_list():
    adata = _make_dotplot_scale_adata()
    fig_single = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=False,
        highlight_cluster="0",
    )
    fig_multi = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=False,
        highlight_cluster=["0", "1"],
    )
    # one row band per highlighted cluster on the dots axis
    assert len(_dots_ax(fig_single).patches) == 1
    assert len(_dots_ax(fig_multi).patches) == 2


def test_extended_dotplot_highlight_cluster_list_spatial_panel_per_cluster_color():
    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=False,
        highlight_cluster=["0", "1"],
        spatial_region="reg1",
    )
    spatial_ax = next(ax for ax in fig.axes if "reg1" in ax.get_title())
    assert spatial_ax.get_title() == "0, 1 -- reg1 (0: n=74, 1: n=74)"
    # grey background + one scatter per highlighted cluster
    assert len(spatial_ax.collections) == 3


# ---------------------------------------------------------------------------
# extended_dotplot: transpose=True
# ---------------------------------------------------------------------------


def test_extended_dotplot_transpose_swaps_which_axis_shows_what():
    """Clusters move to the x-axis (as tick labels, not just data), features to the y-axis --
    the opposite of the default layout."""
    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=False,
        transpose=True,
    )
    ax = _dots_ax(fig)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1", "2"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["marker0"]


def test_extended_dotplot_transpose_dot_values_match_non_transpose():
    """transpose only changes layout, not what's computed -- same color/size values either way,
    just read off a different axis."""
    adata = _make_many_cluster_adata(n_clusters=4, n_markers=3, seed=1)
    markers = list(adata.var_names)
    fig_plain = extended_dotplot(
        adata, "leiden", clustering_markers=markers, show_totals=False
    )
    fig_t = extended_dotplot(
        adata, "leiden", clustering_markers=markers, show_totals=False, transpose=True
    )
    colors_plain = np.sort(_dots_ax(fig_plain).collections[0].get_array())
    colors_t = np.sort(_dots_ax(fig_t).collections[0].get_array())
    np.testing.assert_allclose(colors_plain, colors_t)


def test_extended_dotplot_transpose_highlight_cluster_draws_column_bands():
    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=False,
        highlight_cluster=["0", "1"],
        transpose=True,
    )
    # one axvspan patch per highlighted cluster, same convention the non-transpose axhspan test
    # above checks for row bands
    assert len(_dots_ax(fig).patches) == 2


def test_extended_dotplot_transpose_totals_bar_is_vertical():
    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=True,
        transpose=True,
    )
    totals_ax = _dots_ax(fig, "Cluster size")
    bars = totals_ax.patches
    assert len(bars) == 3
    # a vertical bar is taller than it is wide; a horizontal one (the non-transpose convention)
    # is the other way round
    assert all(b.get_height() > b.get_width() for b in bars)


def test_extended_dotplot_transpose_spatial_panel_present_and_unrotated():
    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=False,
        highlight_cluster=["0", "1"],
        spatial_region="reg1",
        transpose=True,
    )
    spatial_ax = next(ax for ax in fig.axes if "reg1" in ax.get_title())
    assert spatial_ax.get_title() == "0, 1 -- reg1 (0: n=74, 1: n=74)"
    assert spatial_ax.get_xlabel() == "x"
    assert spatial_ax.get_ylabel() == "y"
    assert len(spatial_ax.collections) == 3


def test_extended_dotplot_transpose_dendrogram_adds_axis_with_no_extra_error():
    adata = _make_many_cluster_adata(n_clusters=5, n_markers=4, seed=2)
    markers = list(adata.var_names)
    fig_no_dendro = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=markers,
        show_totals=False,
        transpose=True,
    )
    fig_dendro = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=markers,
        show_totals=False,
        dendrogram=True,
        transpose=True,
    )
    assert len(fig_dendro.axes) > len(fig_no_dendro.axes)
    assert any(len(ax.lines) > 0 for ax in fig_dendro.axes)


def test_extended_dotplot_transpose_legend_present_and_titled():
    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        show_totals=True,
        transpose=True,
    )
    assert any(leg.get_title().get_text() == "frac > thr" for leg in fig.legends)


def test_extended_dotplot_transpose_legend_present_when_sections_too_small_for_it():
    """Regression test: a dot section too short to host the colorbar label + legend must still
    get the legend somewhere (falling back to the totals band) rather than silently dropping it
    or overlapping a neighboring section's colorbar label."""
    adata = _make_dotplot_scale_adata()
    fig = extended_dotplot(
        adata,
        "leiden",
        clustering_markers=["marker0"],
        obs_continuous=None,
        show_totals=True,
        transpose=True,
    )
    assert any(leg.get_title().get_text() == "frac > thr" for leg in fig.legends)


# ---------------------------------------------------------------------------
# plot_top_n_distances
# ---------------------------------------------------------------------------


def _make_dist_data(condition_col="condition"):
    dist_table_filt = pd.DataFrame(
        {"GroupA": [1.0, 5.0], "GroupB": [3.0, 2.0]},
        index=pd.Index(["X_Y", "A_B"], name="pairs"),
    )
    dist_data_filt = pd.DataFrame(
        {
            "pairs": ["X_Y", "X_Y", "A_B", "A_B"],
            condition_col: ["GroupA", "GroupB", "GroupA", "GroupB"],
            "interaction": ["X --> Y", "X --> Y", "A --> B", "A --> B"],
            "observed": [[1.0, 1.2], [3.0, 2.9], [5.0, 5.1], [2.0, 2.1]],
        }
    )
    return dist_table_filt, dist_data_filt


def test_plot_top_n_distances_custom_condition_col():
    # Regression test: x="condition" used to be hardcoded with no way to override it,
    # so any dist_data_filt built with a differently-named comparison column (e.g.
    # "Gut_Segment", whatever name identify_interactions' own `comparison` argument
    # was) raised a KeyError from inside seaborn.
    dist_table_filt, dist_data_filt = _make_dist_data(condition_col="Gut_Segment")
    plot_top_n_distances(
        dist_table_filt, dist_data_filt, n=2, condition_col="Gut_Segment"
    )


def test_plot_top_n_distances_default_condition_col_still_works():
    dist_table_filt, dist_data_filt = _make_dist_data(condition_col="condition")
    plot_top_n_distances(dist_table_filt, dist_data_filt, n=2)


def test_plot_top_n_distances_does_not_modify_input():
    # Regression test: an "abs_dist" column was added to the caller's table and
    # then leaked into dumbbell().
    dist_table_filt, dist_data_filt = _make_dist_data()
    plot_top_n_distances(dist_table_filt, dist_data_filt, n=2)
    assert list(dist_table_filt.columns) == ["GroupA", "GroupB"]


def test_plot_top_n_distances_ranking_labels_and_overlay():
    dist_table_filt, dist_data_filt = _make_dist_data(condition_col="Gut_Segment")
    dist_data_filt["expected_mean"] = [2.0, 2.5, 4.0, 3.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        top = plot_top_n_distances(
            dist_table_filt,
            dist_data_filt,
            n=2,
            condition_col="Gut_Segment",
            errorbars=False,
        )
    # errorbars=False used to pass seaborn's deprecated ci=None
    assert not [w for w in caught if issubclass(w.category, FutureWarning)]
    assert not [w for w in caught if "expected_mean" in str(w.message)]
    # A_B: 2 - 5 = -3 ranks above X_Y: 3 - 1 = +2
    assert list(top.index) == ["A_B", "X_Y"]
    assert list(top["delta_log2fc"]) == [-3.0, 2.0]
    import matplotlib.pyplot as plt

    ax = plt.gcf().axes[0]
    assert ax.get_xlabel() == "Gut_Segment"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["GroupA", "GroupB"]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["A --> B (-3.00)", "X --> Y (+2.00)", "expected (permutation)"]
    assert any(line.get_linestyle() == "--" for line in ax.lines)


def test_plot_top_n_distances_empty_table_raises():
    dist_table_filt, dist_data_filt = _make_dist_data()
    with pytest.raises(ValueError, match="empty"):
        plot_top_n_distances(dist_table_filt.iloc[:0], dist_data_filt)


# ---------------------------------------------------------------------------
# distance_graph
# ---------------------------------------------------------------------------


def _make_distance_pvals():
    return pd.DataFrame(
        {
            "celltype1": ["X", "X", "A", "A"],
            "celltype2": ["Y", "Y", "B", "B"],
            "Gut_Segment": ["GroupA", "GroupB", "GroupA", "GroupB"],
            "interaction": ["X --> Y", "X --> Y", "A --> B", "A --> B"],
            "logfold_group": [0.5, -0.2, -0.3, 0.4],
        }
    )


def test_distance_graph_accepts_pairs_as_index():
    # Regression test: sp.tl.filter_interactions returns dist_table with "pairs" as
    # the index (its own .set_index("pairs") + .pivot()), but distance_graph used to
    # immediately do dist_table[pair_col] expecting a column, raising KeyError: 'pairs'
    # on the exact output the sibling function produces.
    dist_table = pd.DataFrame(
        {"GroupA": [1.0], "GroupB": [3.0]}, index=pd.Index(["X_Y"], name="pairs")
    )
    distance_graph(dist_table, _make_distance_pvals(), condition_col="Gut_Segment")


def test_distance_graph_accepts_pairs_as_column():
    dist_table = pd.DataFrame({"pairs": ["X_Y"], "GroupA": [1.0], "GroupB": [3.0]})
    distance_graph(dist_table, _make_distance_pvals(), condition_col="Gut_Segment")


# ---------------------------------------------------------------------------
# BC_projection
# ---------------------------------------------------------------------------


def _make_bc_adata_and_cnmap():
    rng = np.random.default_rng(0)
    n = 50
    cn_labels = rng.choice(["A", "B", "C"], size=n)
    adata = anndata.AnnData(X=np.zeros((n, 2)), obs=pd.DataFrame({"CN": cn_labels}))
    w = pd.DataFrame(
        {
            "A": rng.integers(0, 76, n),
            "B": rng.integers(0, 76, n),
            "C": rng.integers(0, 76, n),
        }
    )
    w.loc[0, ["A", "B", "C"]] = [70, 3, 2]  # one window at ~100% coverage
    w["CN"] = cn_labels
    cnmap_dict = {"w": w, "l": ["A", "B", "C"], "k": 75, "threshold": 0.85}
    return adata, cnmap_dict


def test_bc_projection_cn_col_annt_none_does_not_crash():
    # Regression test: with palette=None and cn_col_annt=None (the natural call for a
    # single, unannotated cn_col), building the default palette used to do
    # `cn_col_annt + "_colors"`, raising TypeError: unsupported operand type(s) for +:
    # 'NoneType' and 'str'.
    adata, cnmap_dict = _make_bc_adata_and_cnmap()
    BC_projection(
        adata, cnmap_dict, cn_col="CN", plot_list=["A", "B", "C"], threshold=0.9
    )


def test_bc_projection_explicit_cn_col_annt_still_works():
    adata, cnmap_dict = _make_bc_adata_and_cnmap()
    adata.obs["CN_annot"] = adata.obs["CN"]
    BC_projection(
        adata,
        cnmap_dict,
        cn_col="CN",
        cn_col_annt="CN_annot",
        plot_list=["A", "B", "C"],
        threshold=0.9,
    )


def test_bc_projection_threshold_filters_by_coverage_fraction():
    # Regression test: the threshold filter used to compare a raw window count
    # (capped at k, e.g. 0-75) against threshold*100 without normalizing, so any
    # realistic coverage threshold (e.g. 0.9 -> 90) was unreachable for k<90 and
    # silently produced an empty plot. Reproduce the internal computation directly to
    # confirm it now filters by coverage fraction, not raw count.
    w = pd.DataFrame({"A": [70, 10, 5], "B": [3, 10, 5], "C": [2, 10, 5]})
    n_num = 75
    threshold = 0.9 * 100
    wgc = w.loc[(w.loc[:, ["A", "B", "C"]].sum(axis=1) / n_num * 100) > threshold, :]
    assert list(wgc.index) == [
        0
    ], "only the ~100%-coverage row should pass a 90% threshold"


def test_bc_projection_legend_below_triangle_and_rcparams_untouched():
    # Regression test: the legend used to grow upward over the triangle, and the
    # function changed global rcParams.
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    adata, cnmap_dict = _make_bc_adata_and_cnmap()
    before = dict(mpl.rcParams)
    BC_projection(
        adata, cnmap_dict, cn_col="CN", plot_list=["A", "B", "C"], threshold=0.1
    )
    fig = plt.gcf()
    ax = fig.axes[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert (
        ax.get_legend().get_window_extent(renderer).y1
        <= ax.get_window_extent(renderer).y0
    )
    assert ax.get_aspect() == 1.0
    # limits cover the whole triangle even if the windows do not
    assert ax.get_xlim()[0] < 0 and ax.get_xlim()[1] > 1
    assert dict(mpl.rcParams) == before
    plt.close("all")


# ---------------------------------------------------------------------------
# cn_map
# ---------------------------------------------------------------------------


def test_cn_map_colours_follow_cnmap_order():
    # Regression test: node squares were coloured through the key order of the
    # palette passed to cn_map instead of cnmap_dict["l"], the order used by
    # build_cn_map, so squares showed the wrong neighborhood.
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx

    names = ["B", "A", "C"]  # build_cn_map order, differs from the palette order below
    graph = nx.DiGraph([((0,), (0, 1)), ((1,), (0, 1))])
    simp_freqs = pd.Series(
        [0.5, 0.3, 0.2], index=pd.Index([(0,), (1,), (0, 1)], tupleize_cols=False)
    )
    cnmap_dict = {"g": graph, "tops": [(0,)], "simp_freqs": simp_freqs, "l": names}
    palette = {"A": "red", "B": "blue", "C": "green"}
    adata = anndata.AnnData(X=np.zeros((3, 1)), obs=pd.DataFrame({"cn": names}))
    cn_map(adata, cnmap_dict, cn_col="cn", palette=palette, figsize=(4, 4))
    squares = [p for p in plt.gca().patches if isinstance(p, mpatches.Rectangle)]
    drawn = {mcolors.to_hex(p.get_facecolor()) for p in squares}
    assert drawn == {mcolors.to_hex("blue"), mcolors.to_hex("red")}  # B and A, never C
    assert len(squares) == 4  # (0,), (1,) and the two squares of (0, 1)
    plt.close("all")


def test_cn_map_top_marker_beside_circle_and_in_legend():
    # The top-combination "*" was a text glyph drawn above the circle center and missing
    # from the legend; it is now an asterisk marker right of the circle (never covering
    # it) at the circle's height, with a legend entry.
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx

    names = ["B", "A", "C"]
    graph = nx.DiGraph([((0,), (0, 1)), ((1,), (0, 1))])
    simp_freqs = pd.Series(
        [0.5, 0.3, 0.2], index=pd.Index([(0,), (1,), (0, 1)], tupleize_cols=False)
    )
    cnmap_dict = {"g": graph, "tops": [(0,)], "simp_freqs": simp_freqs, "l": names}
    adata = anndata.AnnData(X=np.zeros((3, 1)), obs=pd.DataFrame({"cn": names}))
    cn_map(adata, cnmap_dict, cn_col="cn", figsize=(4, 4))
    ax = plt.gca()
    marks = [l for l in ax.lines if l.get_marker() == (6, 2, 0)]
    assert len(marks) == 1 and len(marks[0].get_xdata()) == 1
    assert not any(t.get_text() == "*" for t in ax.texts)
    # the largest circle belongs to (0,), the most frequent combination
    circle = max(
        (p for p in ax.patches if isinstance(p, mpatches.Circle)),
        key=lambda p: p.radius,
    )
    mx, my = marks[0].get_xdata()[0], marks[0].get_ydata()[0]
    assert my == pytest.approx(circle.center[1])
    assert mx > circle.center[0] + circle.radius
    labels = [t.get_text() for leg in plt.gcf().legends for t in leg.get_texts()]
    assert "Top 1 by frequency" in labels
    plt.close("all")


def _make_upset_cnmap():
    import networkx as nx

    names = ["B", "A", "C"]  # build_cn_map order, differs from the palette order below
    graph = nx.DiGraph([((0,), (0, 1)), ((1,), (0, 1))])
    # (2,) has no edge, so it is missing from graph; (1, 2) is below the 0.001 cutoff
    simp_freqs = pd.Series(
        [0.4, 0.3, 0.2, 0.0995, 0.0005],
        index=pd.Index([(0,), (1,), (0, 1), (2,), (1, 2)], tupleize_cols=False),
    )
    cnmap_dict = {"g": graph, "tops": [(0,)], "simp_freqs": simp_freqs, "l": names}
    adata = anndata.AnnData(X=np.zeros((3, 1)), obs=pd.DataFrame({"cn": names}))
    return adata, cnmap_dict


def test_cn_upset_colours_follow_cnmap_order():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    adata, cnmap_dict = _make_upset_cnmap()
    palette = {"A": "red", "B": "blue", "C": "green"}
    cn_upset(adata, cnmap_dict, cn_col="cn", palette=palette, min_freq=0.25)
    ax_dot = plt.gcf().axes[1]
    drawn = {
        mcolors.to_hex(fc)
        for coll in ax_dot.collections[1:]  # [0] is the grey background grid
        for fc in coll.get_facecolors()
    }
    assert drawn == {mcolors.to_hex("blue"), mcolors.to_hex("red")}  # B and A, never C
    absent = ax_dot.collections[0].get_facecolors()
    assert {mcolors.to_hex(fc) for fc in absent} == {
        "#d0d0d0"
    }  # network_node_default_color
    plt.close("all")


def test_cn_upset_cutoff_includes_combinations_missing_from_graph():
    import matplotlib.pyplot as plt

    adata, cnmap_dict = _make_upset_cnmap()
    cn_upset(adata, cnmap_dict, cn_col="cn")
    assert len(plt.gcf().axes[0].patches) == 4  # all but (1, 2), including (2,)
    assert [t.get_text() for t in plt.gcf().axes[3].texts] == [
        "B",
        "A",
        "C",
    ]  # label boxes
    plt.close("all")
    cn_upset(adata, cnmap_dict, cn_col="cn", min_freq=0.25)
    assert len(plt.gcf().axes[0].patches) == 2  # (0,) and (1,)
    plt.close("all")


def test_cn_upset_long_names_and_many_columns_do_not_overlap():
    # Regression test: wrapped long CN names spilled out of their label boxes, percentage
    # labels ran into each other with many columns, and the left x label was clipped.
    import itertools

    import matplotlib.pyplot as plt

    names = [f"Long neighborhood name {i}" for i in range(12)]
    combos = [(i,) for i in range(12)] + list(itertools.combinations(range(12), 2))[:22]
    freqs = np.linspace(0.1, 0.006, len(combos))
    simp_freqs = pd.Series(
        freqs / freqs.sum(), index=pd.Index(combos, tupleize_cols=False)
    )
    cnmap_dict = {"simp_freqs": simp_freqs, "l": names}
    adata = anndata.AnnData(X=np.zeros((12, 1)), obs=pd.DataFrame({"cn": names}))
    cn_upset(adata, cnmap_dict, cn_col="cn")

    fig = plt.gcf()
    ax_bar, _, ax_cn, ax_lab = fig.axes
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lab_box = ax_lab.get_window_extent(r)
    labels = sorted((t.get_window_extent(r) for t in ax_lab.texts), key=lambda b: -b.y0)
    assert all(lab_box.x0 <= b.x0 and b.x1 <= lab_box.x1 for b in labels)
    assert all(lo.y1 <= hi.y0 for hi, lo in zip(labels, labels[1:]))
    pcts = sorted((t.get_window_extent(r) for t in ax_bar.texts), key=lambda b: b.x0)
    assert all(a.x1 <= b.x0 for a, b in zip(pcts, pcts[1:]))
    xlabel = ax_cn.xaxis.label.get_window_extent(r)
    assert xlabel.x0 >= 0 and xlabel.x1 <= fig.bbox.x1
    plt.close("all")


def test_cn_upset_show_values():
    # show_values=True prints every percentage even when "auto" would leave them out
    # (narrow columns), shrinking the font so neighbours still don't overlap.
    import matplotlib.pyplot as plt

    from spacec.plotting._style import SPACEcStyle

    adata, cnmap_dict = _make_upset_cnmap()
    cn_upset(adata, cnmap_dict, cn_col="cn", show_values=False)
    assert len(plt.gcf().axes[0].texts) == 0
    plt.close("all")

    n = 80
    simp_freqs = pd.Series(
        np.full(n, 1 / n),
        index=pd.Index([(i % 3,) for i in range(n)], tupleize_cols=False),
    )
    cnmap_dict = {"simp_freqs": simp_freqs, "l": ["B", "A", "C"]}
    style = SPACEcStyle(font_size_tiny=14)
    cn_upset(adata, cnmap_dict, cn_col="cn", figsize=(4, 3), style=style)
    assert len(plt.gcf().axes[0].texts) == 0  # "auto": too narrow even when vertical
    plt.close("all")
    cn_upset(
        adata,
        cnmap_dict,
        cn_col="cn",
        figsize=(4, 3),
        style=style,
        show_values=True,
        value_decimals=2,
    )
    fig = plt.gcf()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = fig.axes[0].texts
    assert [t.get_text() for t in texts] == ["1.25"] * n
    boxes = sorted((t.get_window_extent(r) for t in texts), key=lambda b: b.x0)
    assert all(a.x1 <= b.x0 for a, b in zip(boxes, boxes[1:]))
    plt.close("all")

    # each number sits above its own bar, never on it (horizontal and vertical labels)
    for kw in ({"figsize": (8, 4)}, {"figsize": (4, 3), "show_values": True}):
        adata, cnmap_dict = _make_upset_cnmap()
        cn_upset(adata, cnmap_dict, cn_col="cn", style=style, **kw)
        fig = plt.gcf()
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        bars = fig.axes[0].patches
        texts = fig.axes[0].texts
        assert len(texts) == len(bars)
        gap_px = 1.5 * fig.dpi / 72  # visibly clear of the bar, not just touching it
        for bar, t in zip(bars, texts):
            assert t.get_window_extent(r).y0 >= bar.get_window_extent(r).y1 + gap_px
        plt.close("all")


# ---------------------------------------------------------------------------
# PPA result donut plot (ppa_res_donut)
# ---------------------------------------------------------------------------


def test_ppa_res_donut_ring_labels_are_centred_and_boxed():
    # Regression test: ring labels used baseline alignment without a background and
    # overlapped the wedges.
    import matplotlib.pyplot as plt

    res = pd.DataFrame(
        {
            "cell_type": ["a", "b", "a", "b", "a"],
            "distance_from_patch": [10, 10, 20, 20, 20],
        }
    )
    adata = anndata.AnnData(
        X=np.zeros((2, 1)), obs=pd.DataFrame({"cell_type": ["a", "b"]})
    )
    adata.uns["ppa"] = res
    ppa_res_donut(
        adata, cat_col="cell_type", key_name="ppa", palette={"a": "red", "b": "blue"}
    )
    labels = [t for t in plt.gca().texts if t.get_text().endswith("µm")]
    assert len(labels) == 2
    assert all(
        t.get_verticalalignment() == "center" and t.get_bbox_patch() is not None
        for t in labels
    )
    plt.close("all")


def test_catplot_single_figure_legend_with_patches():
    # Regression test: per-panel seaborn legends inherited the scatter size (invisible
    # swatches) and ran into neighbouring panels.
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    adata = _make_adata()
    adata.obs["cell_type"] = pd.Categorical(["t1", "t2", "t1", "t2"])
    palette = {"t1": "red", "t2": "blue"}
    catplot(
        adata, color="cell_type", unique_region="unique_region", palette=palette, size=1
    )
    fig = plt.gcf()
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert len(fig.legends) == 1
    handles = fig.legends[0].legend_handles
    assert all(isinstance(h, mpatches.Patch) for h in handles)
    assert {mcolors.to_hex(h.get_facecolor()) for h in handles} == {
        mcolors.to_hex(c) for c in palette.values()
    }
    plt.close("all")


# ---------------------------------------------------------------------------
# swarm_box_plot
# ---------------------------------------------------------------------------


def _make_swarm_adata():
    """2 conditions x 3 patients each; per patient 10 cells: 5 B, 3 T, 2 NK.

    Patient p3 (cond1) has no NK at all -> must give an explicit 0, not a missing row.
    A coarse column ("lineage") lumps B and T as "lymph" and NK as "innate".
    """
    rows = []
    for cond, pats in (("ctrl", ["p1", "p2", "p3"]), ("treat", ["p4", "p5", "p6"])):
        for p in pats:
            ct = ["B"] * 5 + ["T"] * 3 + (["NK"] * 2 if p != "p3" else ["B"] * 2)
            rows += [(cond, p, c) for c in ct]
    obs = pd.DataFrame(rows, columns=["condition", "patient", "celltype"])
    obs["lineage"] = obs["celltype"].map({"B": "lymph", "T": "lymph", "NK": "innate"})
    obs.index = obs.index.astype(str)
    return anndata.AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def _swarm(adata, **kw):
    import matplotlib.pyplot as plt

    kw.setdefault("color", "celltype")
    kw.setdefault("grouping", "condition")
    kw.setdefault("replicate", "patient")
    kw.setdefault("cell_list", ["B", "T", "NK"])
    kw.setdefault("output_dir", None)
    out = swarm_box_plot(adata, **kw)
    plt.close("all")
    return out


def test_swarm_box_plot_returns_long_table_and_order():
    df, order = _swarm(_make_swarm_adata())
    assert list(df.columns) == ["condition", "patient", "celltype", "percent"]
    assert order == ["B", "T", "NK"]
    assert len(df) == 6 * 3  # patients x categories
    assert swarm_box_plot is pl_swarm_box_ad


def test_swarm_box_plot_one_panel_per_category():
    import matplotlib.pyplot as plt

    swarm_box_plot(
        _make_swarm_adata(),
        color="celltype",
        grouping="condition",
        replicate="patient",
        cell_list=["B", "T", "NK"],
        output_dir=None,
    )
    fig = plt.gcf()
    assert [ax.get_title() for ax in fig.axes if ax.get_visible()] == ["B", "T", "NK"]
    plt.close("all")


def test_swarm_box_plot_point_size_sets_dot_size():
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    for point_size in (5, 2.5):
        swarm_box_plot(
            _make_swarm_adata(),
            color="celltype",
            grouping="condition",
            replicate="patient",
            cell_list=["B"],
            output_dir=None,
            point_size=point_size,
        )
        dots = [
            c
            for ax in plt.gcf().axes
            for c in ax.collections
            if isinstance(c, PathCollection)
        ]
        assert dots
        assert all(np.allclose(c.get_sizes(), point_size**2) for c in dots)
        plt.close("all")


def test_swarm_box_plot_norm_true_sums_to_100_and_values_correct():
    df, _ = _swarm(_make_swarm_adata())
    assert np.allclose(df.groupby("patient")["percent"].sum(), 100)
    p1 = df[df["patient"] == "p1"].set_index("celltype")["percent"]
    assert p1["B"] == 50 and p1["T"] == 30 and p1["NK"] == 20


def test_swarm_box_plot_norm_false_uses_all_cells_as_denominator():
    df, _ = _swarm(_make_swarm_adata(), cell_list=["B", "T"], norm=False)
    # NK is not plotted but still counts in the denominator
    p1 = df[df["patient"] == "p1"].set_index("celltype")["percent"]
    assert p1["B"] == 50 and p1["T"] == 30
    assert (df.groupby("patient")["percent"].sum() < 100).any()


def test_swarm_box_plot_absent_category_is_zero_not_missing():
    df, _ = _swarm(_make_swarm_adata())
    row = df[(df["patient"] == "p3") & (df["celltype"] == "NK")]
    assert len(row) == 1 and row["percent"].iloc[0] == 0


def test_swarm_box_plot_sub_col_selects_categories_inside_subset():
    df, order = _swarm(_make_swarm_adata(), sub_col="lineage", cell_list=["lymph"])
    assert set(order) == {"B", "T"}
    assert np.allclose(df.groupby("patient")["percent"].sum(), 100)


def test_swarm_box_plot_does_not_mutate_adata():
    adata = _make_swarm_adata()
    before = adata.obs.copy()
    _swarm(adata)
    pd.testing.assert_frame_equal(adata.obs, before)


def test_swarm_box_plot_col_order_and_plot_order_respected():
    import matplotlib.pyplot as plt

    df, order = swarm_box_plot(
        _make_swarm_adata(),
        color="celltype",
        grouping="condition",
        replicate="patient",
        cell_list=["B", "T", "NK"],
        output_dir=None,
        col_order=["treat", "ctrl"],
        plot_order=["NK", "B"],
    )
    ax = plt.gcf().axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["treat", "ctrl"]
    assert order == ["NK", "B"] and ax.get_title() == "NK"
    plt.close("all")


def test_swarm_box_plot_palette_dict_colors_panel():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    swarm_box_plot(
        _make_swarm_adata(),
        color="celltype",
        grouping="condition",
        replicate="patient",
        cell_list=["B"],
        output_dir=None,
        palette={"B": "#ff0000"},
    )
    ax = plt.gcf().axes[0]
    assert to_rgba("#ff0000")[:3] == ax.patches[0].get_facecolor()[:3]
    plt.close("all")


@pytest.mark.parametrize(
    "kw",
    [
        {"color": "nope"},
        {"grouping": "nope"},
        {"replicate": "nope"},
        {"replicate": "condition"},
        {"cell_list": "B"},
        {"cell_list": ["nothing"]},
        {"col_order": ["nothing"]},
        {"plot_order": ["nothing"]},
    ],
)
def test_swarm_box_plot_misuse_explains_usage(kw):
    with pytest.raises(ValueError, match="How to use swarm_box_plot"):
        _swarm(_make_swarm_adata(), **kw)


def test_swarm_box_plot_rejects_non_adata():
    with pytest.raises(ValueError, match="AnnData"):
        swarm_box_plot(pd.DataFrame(), "a", "b", "c", ["x"], None)


def test_swarm_box_plot_warns_on_single_replicate_per_group():
    adata = _make_swarm_adata()
    adata.obs["patient"] = adata.obs["condition"]  # one replicate per group
    with pytest.warns(UserWarning, match="only one replicate"):
        _swarm(adata)


def test_swarm_box_plot_warns_on_replicate_spanning_groups():
    adata = _make_swarm_adata()
    adata.obs["patient"] = "same"
    with pytest.warns(UserWarning, match="more than one"):
        _swarm(adata)


def test_swarm_box_plot_savefig_writes_file(tmp_path):
    _swarm(
        _make_swarm_adata(), savefig=True, output_dir=str(tmp_path), output_fname="sb"
    )
    assert len(list(tmp_path.glob("sb_swarm_boxplot.*"))) == 1


def test_classification_report_heatmap_unsorted_target_names():
    # Rows must follow target_names, not sklearn's sorted label order renamed
    # positionally ("c" is perfect, "a" is not).
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["b", "a", "b", "b", "c", "c"]
    _, report_df = classification_report_heatmap(
        y_true=y_true, y_pred=y_pred, target_names=["c", "a", "b"], return_data=True
    )
    assert list(report_df.index[:3]) == ["c", "a", "b"]
    assert report_df.loc["c", "recall"] == 1.0
    assert report_df.loc["a", "recall"] == 0.5


def test_extended_dotplot_layout_keywords_shrink_figure():
    import matplotlib.pyplot as plt

    adata = _make_report_adata(None)
    wide = extended_dotplot(adata, "leiden_1", clustering_markers=["M0", "M1", "M2"])
    narrow = extended_dotplot(
        adata,
        "leiden_1",
        clustering_markers=["M0", "M1", "M2"],
        margin_left_inches=0.5,
        section_gap_inches=0.1,
        totals_width_inches=0.5,
        label_block_min_inches=0.5,
    )
    exact = extended_dotplot(
        adata, "leiden_1", clustering_markers=["M0", "M1", "M2"], label_block_inches=0.2
    )
    assert exact.get_size_inches()[1] < narrow.get_size_inches()[1]
    assert narrow.get_size_inches()[0] < wide.get_size_inches()[0] - 2
    assert narrow.get_size_inches()[1] < wide.get_size_inches()[1]
    plt.close("all")


def test_patch_qc_size_and_font_keywords():
    import matplotlib.pyplot as plt

    from spacec.plotting._general import patch_QC

    rng = np.random.default_rng(0)
    mask = np.zeros((200, 200), dtype=np.int32)
    xs, ys = [], []
    for i in range(16):
        r, c = divmod(i, 4)
        mask[r * 50 + 10 : r * 50 + 40, c * 50 + 10 : c * 50 + 40] = i + 1
        ys.append(r * 50 + 25)
        xs.append(c * 50 + 25)
    obs = pd.DataFrame(
        {
            "x": xs,
            "y": ys,
            "unique_region": "reg1",
            "label": np.arange(1, 17),
            "cl": pd.Categorical(["a", "b"] * 8),
        },
        index=[str(i) for i in range(16)],
    )
    ad = anndata.AnnData(X=rng.random((16, 2)), obs=obs)
    ad.var_names = ["DAPI", "CD3"]
    images = {"reg1": {"DAPI": rng.random((200, 200)), "CD3": rng.random((200, 200))}}
    big = patch_QC(
        ad,
        "cl",
        "a",
        ["CD3"],
        {"reg1": mask},
        images,
        region="reg1",
        n_patches=4,
        patch_size=(40, 40),
    )
    small = patch_QC(
        ad,
        "cl",
        "a",
        ["CD3"],
        {"reg1": mask},
        images,
        region="reg1",
        n_patches=4,
        patch_size=(40, 40),
        cell_in=0.5,
        gap_in=0.25,
        legend_width_in=1.0,
        fontsize=7,
    )
    assert small.get_size_inches()[0] < big.get_size_inches()[0] / 2
    sizes = {t.get_fontsize() for t in small.findobj(plt.Text) if t.get_text()}
    assert min(sizes) >= 7 and max(sizes) <= 9
    plt.close("all")


def test_add_scalebar_label_text_and_position():
    import dataclasses

    import matplotlib.pyplot as plt

    from spacec.plotting._style import SPACEcStyle, _add_scalebar

    for top in (False, True):
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((100, 400)))
        s = dataclasses.replace(SPACEcStyle(), scalebar_label_top=top)
        _add_scalebar(ax, 400, s, scalebar_length_px=200 / 0.377442, um_per_px=0.377442)
        bar = ax.artists[-1]
        texts = [t.get_text() for t in bar.findobj(lambda o: hasattr(o, "get_text"))]
        assert "200 µm" in texts  # float noise must not print as "200.0 µm"
        children = (
            bar.get_child().get_children()
        )  # VPacker: [label, bar] when label_top, else [bar, label]
        assert (type(children[0]).__name__ == "TextArea") == top
        plt.close(fig)


def test_downscale_tissue_uses_shared_scalebar(tmp_path):
    import matplotlib.pyplot as plt
    import tifffile
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    import spacec as sp

    img = np.zeros((1, 512, 512), dtype=np.uint16)
    img[0, 100:400, 100:400] = 1000
    path = tmp_path / "t.tif"
    tifffile.imwrite(path, img)
    figs = []
    real_show = plt.show
    plt.show = lambda *a, **k: figs.extend(map(plt.figure, plt.get_fignums()))
    try:
        sp.hf.downscale_tissue(
            file_path=str(path),
            DNAslice=0,
            downscale_factor=8,
            sigma=1,
            padding=0,
            showfig=True,
            verbose=False,
            um_per_px=0.5,
            scalebar_length_px=200,
        )
    finally:
        plt.show = real_show
    labels = [
        a.txt_label.get_text()
        for ax in figs[0].axes[:2]
        for a in ax.artists
        if isinstance(a, AnchoredSizeBar)
    ]
    assert labels == [
        "100 µm",
        "100 µm",
    ]  # raw and downscaled panel: same bar, same label
    plt.close("all")


def test_extended_dotplot_layout_keywords_shrink_figure():
    import matplotlib.pyplot as plt

    adata = _make_report_adata(None)
    wide = extended_dotplot(adata, "leiden_1", clustering_markers=["M0", "M1", "M2"])
    narrow = extended_dotplot(
        adata,
        "leiden_1",
        clustering_markers=["M0", "M1", "M2"],
        margin_left_inches=0.5,
        section_gap_inches=0.1,
        totals_width_inches=0.5,
        label_block_min_inches=0.5,
    )
    exact = extended_dotplot(
        adata, "leiden_1", clustering_markers=["M0", "M1", "M2"], label_block_inches=0.2
    )
    assert exact.get_size_inches()[1] < narrow.get_size_inches()[1]
    assert narrow.get_size_inches()[0] < wide.get_size_inches()[0] - 2
    assert narrow.get_size_inches()[1] < wide.get_size_inches()[1]
    plt.close("all")


def test_patch_qc_size_and_font_keywords():
    import matplotlib.pyplot as plt

    from spacec.plotting._general import patch_QC

    rng = np.random.default_rng(0)
    mask = np.zeros((200, 200), dtype=np.int32)
    xs, ys = [], []
    for i in range(16):
        r, c = divmod(i, 4)
        mask[r * 50 + 10 : r * 50 + 40, c * 50 + 10 : c * 50 + 40] = i + 1
        ys.append(r * 50 + 25)
        xs.append(c * 50 + 25)
    obs = pd.DataFrame(
        {
            "x": xs,
            "y": ys,
            "unique_region": "reg1",
            "label": np.arange(1, 17),
            "cl": pd.Categorical(["a", "b"] * 8),
        },
        index=[str(i) for i in range(16)],
    )
    ad = anndata.AnnData(X=rng.random((16, 2)), obs=obs)
    ad.var_names = ["DAPI", "CD3"]
    images = {"reg1": {"DAPI": rng.random((200, 200)), "CD3": rng.random((200, 200))}}
    big = patch_QC(
        ad,
        "cl",
        "a",
        ["CD3"],
        {"reg1": mask},
        images,
        region="reg1",
        n_patches=4,
        patch_size=(40, 40),
    )
    small = patch_QC(
        ad,
        "cl",
        "a",
        ["CD3"],
        {"reg1": mask},
        images,
        region="reg1",
        n_patches=4,
        patch_size=(40, 40),
        cell_in=0.5,
        gap_in=0.25,
        legend_width_in=1.0,
        fontsize=7,
    )
    assert small.get_size_inches()[0] < big.get_size_inches()[0] / 2
    sizes = {t.get_fontsize() for t in small.findobj(plt.Text) if t.get_text()}
    assert min(sizes) >= 7 and max(sizes) <= 9
    plt.close("all")


def test_add_scalebar_label_text_and_position():
    import dataclasses

    import matplotlib.pyplot as plt

    from spacec.plotting._style import SPACEcStyle, _add_scalebar

    for top in (False, True):
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((100, 400)))
        s = dataclasses.replace(SPACEcStyle(), scalebar_label_top=top)
        _add_scalebar(ax, 400, s, scalebar_length_px=200 / 0.377442, um_per_px=0.377442)
        bar = ax.artists[-1]
        texts = [t.get_text() for t in bar.findobj(lambda o: hasattr(o, "get_text"))]
        assert "200 µm" in texts  # float noise must not print as "200.0 µm"
        children = (
            bar.get_child().get_children()
        )  # VPacker: [label, bar] when label_top, else [bar, label]
        assert (type(children[0]).__name__ == "TextArea") == top
        plt.close(fig)


def test_downscale_tissue_uses_shared_scalebar(tmp_path):
    import matplotlib.pyplot as plt
    import tifffile
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    import spacec as sp

    img = np.zeros((1, 512, 512), dtype=np.uint16)
    img[0, 100:400, 100:400] = 1000
    path = tmp_path / "t.tif"
    tifffile.imwrite(path, img)
    figs = []
    real_show = plt.show
    plt.show = lambda *a, **k: figs.extend(map(plt.figure, plt.get_fignums()))
    try:
        sp.hf.downscale_tissue(
            file_path=str(path),
            DNAslice=0,
            downscale_factor=8,
            sigma=1,
            padding=0,
            showfig=True,
            verbose=False,
            um_per_px=0.5,
            scalebar_length_px=200,
        )
    finally:
        plt.show = real_show
    labels = [
        a.txt_label.get_text()
        for ax in figs[0].axes[:2]
        for a in ax.artists
        if isinstance(a, AnchoredSizeBar)
    ]
    assert labels == [
        "100 µm",
        "100 µm",
    ]  # raw and downscaled panel: same bar, same label
    plt.close("all")


def test_bundled_mplstyles_registered():
    import matplotlib.pyplot as plt

    import spacec  # noqa: F401  (registers the bundled styles)

    expected = {"spacec_publication", "spacec_presentation", "spacec_tissue"}
    assert expected <= set(plt.style.available)
    with plt.style.context("spacec_publication"):
        pass
