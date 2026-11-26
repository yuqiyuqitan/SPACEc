"""Unit tests for spacec.plotting._spatial_widget.

The widget is exercised headlessly: ``spatial_cell_type_widget`` returns a
``_SpatialCellTypeWidget`` instance whose methods are invoked directly without
a live Jupyter frontend. Plot rendering uses the non-interactive Agg backend.
"""

import matplotlib

matplotlib.use("Agg")

import anndata
import numpy as np
import pandas as pd
import pytest

from spacec.plotting._spatial_widget import (
    _build_color_map,
    _SpatialCellTypeWidget,
    _validate,
    spatial_cell_type_widget,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adata(n_cells: int = 120, seed: int = 0) -> anndata.AnnData:
    """Synthetic AnnData with two regions and three cell types."""
    rng = np.random.default_rng(seed)
    cell_types = ["T cell", "B cell", "Macrophage"]
    regions = ["region_A", "region_B"]

    obs = pd.DataFrame(
        {
            "x": rng.uniform(0, 500, size=n_cells),
            "y": rng.uniform(0, 500, size=n_cells),
            "cell_type": pd.Categorical(
                [cell_types[i % len(cell_types)] for i in range(n_cells)]
            ),
            "unique_region": pd.Categorical(
                [regions[i % len(regions)] for i in range(n_cells)]
            ),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    X = rng.normal(size=(n_cells, 4)).astype(np.float32)
    var = pd.DataFrame(index=[f"g{i}" for i in range(4)])
    return anndata.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def adata():
    return _make_adata()


def _make_widget(adata) -> _SpatialCellTypeWidget:
    """Build a widget instance without displaying it."""
    return _SpatialCellTypeWidget(
        adata=adata,
        cell_type_col="cell_type",
        region_col="unique_region",
        x_col="x",
        y_col="y",
        palette="tab20",
        point_size=5.0,
        grey_alpha=0.2,
        color_alpha=0.8,
        figsize=(4, 4),
    )


# ---------------------------------------------------------------------------
# _build_color_map
# ---------------------------------------------------------------------------


def test_build_color_map_length():
    types = ["A", "B", "C"]
    cmap = _build_color_map(types, "tab20")
    assert set(cmap.keys()) == set(types)


def test_build_color_map_stable_order():
    types = ["Z", "A", "M"]
    cmap = _build_color_map(types, "tab10")
    # sorted order: A, M, Z → A gets index 0 color
    cmap2 = _build_color_map(["A", "M", "Z"], "tab10")
    assert cmap == cmap2


def test_build_color_map_continuous_palette():
    types = ["X", "Y"]
    cmap = _build_color_map(types, "viridis")
    assert len(cmap) == 2
    for v in cmap.values():
        assert len(v) >= 3  # RGB or RGBA from continuous cmap


# ---------------------------------------------------------------------------
# _validate
# ---------------------------------------------------------------------------


def test_validate_passes(adata):
    _validate(adata, "cell_type", "unique_region", "x", "y")  # no exception


def test_validate_missing_column(adata):
    with pytest.raises(ValueError, match="missing from adata.obs"):
        _validate(adata, "nonexistent_col", "unique_region", "x", "y")


# ---------------------------------------------------------------------------
# Widget initialisation
# ---------------------------------------------------------------------------


def test_widget_initializes(adata):
    w = _make_widget(adata)
    expected_regions = sorted(adata.obs["unique_region"].astype(str).unique())
    assert list(w.region_dd.options) == expected_regions


def test_cell_type_options_populated_on_init(adata):
    w = _make_widget(adata)
    # After init the first region is selected; its cell types must appear.
    first_region = w.region_dd.value
    expected = sorted(
        adata.obs.loc[
            adata.obs["unique_region"].astype(str) == str(first_region), "cell_type"
        ]
        .astype(str)
        .unique()
    )
    assert list(w.cell_type_sm.options) == expected


def test_all_cell_types_selected_by_default(adata):
    w = _make_widget(adata)
    assert set(w.cell_type_sm.value) == set(w.cell_type_sm.options)


# ---------------------------------------------------------------------------
# Region change
# ---------------------------------------------------------------------------


def test_cell_type_options_update_on_region_change(adata):
    w = _make_widget(adata)
    regions = list(w.region_dd.options)
    second_region = regions[1] if len(regions) > 1 else regions[0]
    w.region_dd.value = second_region

    expected = sorted(
        adata.obs.loc[
            adata.obs["unique_region"].astype(str) == str(second_region), "cell_type"
        ]
        .astype(str)
        .unique()
    )
    assert list(w.cell_type_sm.options) == expected


# ---------------------------------------------------------------------------
# Select All / Clear
# ---------------------------------------------------------------------------


def test_select_all(adata):
    w = _make_widget(adata)
    w._clear_selection()
    assert w.cell_type_sm.value == ()
    w._select_all()
    assert set(w.cell_type_sm.value) == set(w.cell_type_sm.options)


def test_clear_selection(adata):
    w = _make_widget(adata)
    w._clear_selection()
    assert w.cell_type_sm.value == ()


# ---------------------------------------------------------------------------
# Plot rendering
# ---------------------------------------------------------------------------


def test_render_plot_non_empty_selection(adata):
    w = _make_widget(adata)
    w._render_plot()  # should not raise


def test_render_plot_empty_selection(adata):
    w = _make_widget(adata)
    w._clear_selection()
    w._render_plot()  # grey-only path — should not raise


def test_render_plot_single_type(adata):
    w = _make_widget(adata)
    w.cell_type_sm.value = [w.cell_type_sm.options[0]]
    w._render_plot()  # single highlighted type — should not raise


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def test_spatial_cell_type_widget_returns_instance(adata):
    w = spatial_cell_type_widget(adata)
    assert isinstance(w, _SpatialCellTypeWidget)


def test_spatial_cell_type_widget_custom_cols():
    """Verify non-default column names are accepted."""
    rng = np.random.default_rng(1)
    n = 40
    obs = pd.DataFrame(
        {
            "cx": rng.uniform(0, 100, n),
            "cy": rng.uniform(0, 100, n),
            "label": pd.Categorical(["A", "B"] * (n // 2)),
            "sample": pd.Categorical(["s1"] * n),
        },
        index=[f"c{i}" for i in range(n)],
    )
    ad = anndata.AnnData(X=rng.normal(size=(n, 2)).astype(np.float32), obs=obs)
    w = spatial_cell_type_widget(
        ad,
        cell_type_col="label",
        region_col="sample",
        x_col="cx",
        y_col="cy",
    )
    assert isinstance(w, _SpatialCellTypeWidget)


def test_spatial_cell_type_widget_missing_col_raises(adata):
    with pytest.raises(ValueError, match="missing from adata.obs"):
        spatial_cell_type_widget(adata, cell_type_col="bad_col")
