"""Unit tests for spacec.helperfunctions._general utility functions.

NOTE: hf_replace_names() is NOT tested here because it calls hf_invert_dictionary()
which does not exist in the module. The actual function is invert_dictionary() (no
hf_ prefix). This is a known bug that must be fixed before hf_replace_names can be
tested.
"""

import os
import pathlib
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile

from spacec.helperfunctions._general import (
    hf_alloc_cells,
    hf_attach_masks,
    hf_extract_filename,
    hf_generate_random_colors,
    hf_get_redundant_pairs,
    hf_get_thresh_simps,
    hf_get_tif_filepaths,
    hf_get_top_abs_correlations,
    hf_index_rank,
    hf_list_folders,
    hf_load_masks,
    hf_normalize,
    hf_per_only,
    hf_prepare_neighborhood_df,
    invert_dictionary,
    is_dark,
    make_anndata,
)

# ---------------------------------------------------------------------------
# invert_dictionary
# ---------------------------------------------------------------------------


def test_invert_dictionary_basic():
    d = {"a": 1, "b": 2, "c": 3}
    result = invert_dictionary(d)
    assert result == {1: "a", 2: "b", 3: "c"}
    assert len(result) == 3


def test_invert_dictionary_empty():
    assert invert_dictionary({}) == {}


def test_invert_dictionary_string_values():
    d = {"x": "foo", "y": "bar"}
    result = invert_dictionary(d)
    assert result == {"foo": "x", "bar": "y"}


# ---------------------------------------------------------------------------
# is_dark
# ---------------------------------------------------------------------------


def test_is_dark_black_is_dark():
    assert is_dark("black") is True


def test_is_dark_white_is_not_dark():
    assert is_dark("white") is False


def test_is_dark_red_is_dark():
    # red: r=1, g=0, b=0 → brightness=(299+0+0)/1000=0.299 < 0.5
    assert is_dark("red") is True


def test_is_dark_yellow_is_not_dark():
    # yellow: r=1, g=1, b=0 → brightness=(299+587+0)/1000=0.886
    assert is_dark("yellow") is False


def test_is_dark_hex_color():
    assert is_dark("#000000") is True  # pure black in hex
    assert is_dark("#ffffff") is False  # pure white in hex


# ---------------------------------------------------------------------------
# hf_extract_filename
# ---------------------------------------------------------------------------


def test_hf_extract_filename_with_suffix():
    path = "/some/path/mysample_plot.png_cut.png"
    result = hf_extract_filename(path)
    assert result == "mysample"


def test_hf_extract_filename_without_suffix():
    path = "/path/to/plain_name.tif"
    result = hf_extract_filename(path)
    assert result == "plain_name.tif"


def test_hf_extract_filename_nested_path():
    path = "a/b/c/sample_plot.png_cut.png"
    result = hf_extract_filename(path)
    assert result == "sample"


# ---------------------------------------------------------------------------
# hf_generate_random_colors
# ---------------------------------------------------------------------------


def test_hf_generate_random_colors_count():
    colors = hf_generate_random_colors(5)
    assert len(colors) == 5
    for c in colors:
        assert c.startswith("#")
        assert len(c) == 7
        # each char after '#' must be hex digit
        assert all(ch in "0123456789ABCDEFabcdef" for ch in c[1:])


def test_hf_generate_random_colors_reproducible():
    c1 = hf_generate_random_colors(10, rand_seed=42)
    c2 = hf_generate_random_colors(10, rand_seed=42)
    assert c1 == c2


def test_hf_generate_random_colors_different_seeds():
    c1 = hf_generate_random_colors(5, rand_seed=0)
    c2 = hf_generate_random_colors(5, rand_seed=99)
    assert c1 != c2


def test_hf_generate_random_colors_zero():
    assert hf_generate_random_colors(0) == []


# ---------------------------------------------------------------------------
# hf_normalize
# ---------------------------------------------------------------------------


def test_hf_normalize_returns_dataframe():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = hf_normalize(df)
    assert isinstance(result, pd.DataFrame)


def test_hf_normalize_shape_preserved():
    df = pd.DataFrame(np.ones((5, 4)))
    result = hf_normalize(df)
    assert result.shape == (5, 4)


def test_hf_normalize_no_nan():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = hf_normalize(df)
    assert not result.isnull().any().any(), "normalized result should contain no NaN"


def test_hf_normalize_handles_nan_input():
    df = pd.DataFrame({"a": [np.nan, 1.0], "b": [2.0, 3.0]})
    result = hf_normalize(df)
    assert isinstance(result, pd.DataFrame)
    assert (
        not result.isnull().any().any()
    ), "NaN input should be filled before normalizing"


# ---------------------------------------------------------------------------
# hf_index_rank
# ---------------------------------------------------------------------------


def test_hf_index_rank_basic():
    # row [10, 30, 20]: rank-ascending → 10=0, 30=2, 20=1
    a = np.array([[10, 30, 20]])
    result = hf_index_rank(a, axis=1)
    assert result.shape == (1, 3)
    np.testing.assert_array_equal(result[0], [0, 2, 1])


def test_hf_index_rank_ties_produce_valid_indices():
    a = np.array([[5, 5, 1]])
    result = hf_index_rank(a, axis=1)
    assert result.shape == (1, 3)
    assert set(result[0]) == {0, 1, 2}, "all rank values should be in {0, 1, 2}"


# ---------------------------------------------------------------------------
# hf_get_thresh_simps
# ---------------------------------------------------------------------------


def test_hf_get_thresh_simps_single_col_exceeds_thresh():
    x = np.array([[0.6, 0.3, 0.1]])
    result = hf_get_thresh_simps(x, thresh=0.5)
    assert isinstance(result, pd.Series)
    assert len(result) == 1
    assert result[0] == (0,)


def test_hf_get_thresh_simps_multiple_cols_needed():
    x = np.array([[0.4, 0.35, 0.25]])
    result = hf_get_thresh_simps(x, thresh=0.7)
    assert result[0] == (0, 1)


def test_hf_get_thresh_simps_multiple_rows():
    x = np.array([[0.9, 0.1], [0.4, 0.6]])
    result = hf_get_thresh_simps(x, thresh=0.5)
    assert len(result) == 2
    assert result[0] == (0,)  # 0.9 alone exceeds 0.5
    assert result[1] == (1,)  # 0.6 (col1) is the largest and exceeds 0.5


def test_hf_get_thresh_simps_returns_series():
    x = np.array([[0.5, 0.3, 0.2]])
    result = hf_get_thresh_simps(x, thresh=0.4)
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# hf_get_redundant_pairs
# ---------------------------------------------------------------------------


def test_hf_get_redundant_pairs_three_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    pairs = hf_get_redundant_pairs(df)
    assert isinstance(pairs, set)
    # diagonal + lower triangle for 3 cols = 3+2+1 = 6
    assert len(pairs) == 6
    assert ("a", "a") in pairs  # diagonal
    assert ("b", "a") in pairs  # lower triangle
    assert ("a", "b") not in pairs  # upper triangle excluded


def test_hf_get_redundant_pairs_two_columns():
    df = pd.DataFrame({"x": [1], "y": [2]})
    pairs = hf_get_redundant_pairs(df)
    assert len(pairs) == 3  # (x,x), (y,x), (y,y)


# ---------------------------------------------------------------------------
# hf_get_top_abs_correlations
# ---------------------------------------------------------------------------


def test_hf_get_top_abs_correlations_structure():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [1.0, 2.0, 3.0],  # perfect correlation with a
            "c": [-1.0, -2.0, -3.0],  # perfect anti-correlation with a
        }
    )
    result = hf_get_top_abs_correlations(df, thresh=0.5)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) >= {"col1", "col2", "value"}
    # no self-correlations (diagonal dropped)
    assert not any(result["col1"] == result["col2"])
    # all returned values exceed threshold in absolute value
    assert all(result["value"].abs() >= 0.5)


def test_hf_get_top_abs_correlations_high_thresh():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [1.0, 2.0, 3.0],
        }
    )
    result = hf_get_top_abs_correlations(df, thresh=0.99)
    assert all(result["value"].abs() >= 0.99)


def test_hf_get_top_abs_correlations_no_results():
    # orthogonal columns → near-zero correlation
    df = pd.DataFrame(
        {
            "a": [1.0, 0.0, -1.0, 0.0],
            "b": [0.0, 1.0, 0.0, -1.0],
        }
    )
    result = hf_get_top_abs_correlations(df, thresh=0.99)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# hf_alloc_cells
# ---------------------------------------------------------------------------


def test_hf_alloc_cells_cosine():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = hf_alloc_cells(X, centroids, metric="cosine")
    assert result[0] == 0
    assert result[1] == 1
    assert result[2] == 0  # [0.9, 0.1] closer to [1, 0]


def test_hf_alloc_cells_euclidean():
    X = np.array([[0.0, 0.0], [10.0, 0.0]])
    centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
    result = hf_alloc_cells(X, centroids, metric="euclidean")
    assert result[0] == 0
    assert result[1] == 1


def test_hf_alloc_cells_returns_ndarray():
    X = np.array([[1.0, 0.0]])
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = hf_alloc_cells(X, centroids, metric="cosine")
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# hf_prepare_neighborhood_df
# ---------------------------------------------------------------------------


def test_hf_prepare_neighborhood_df_creates_patients_column():
    df = pd.DataFrame(
        {
            "tissue": ["T", "T", "N"],
            "sample": ["s1", "s2", "s1"],
            "cell": [1, 2, 3],
        }
    )
    result = hf_prepare_neighborhood_df(df, "tissue", "sample")
    assert "patients" in result.columns
    assert result["patients"].iloc[0] == "T_s1"
    assert result["patients"].dtype.name == "category"


def test_hf_prepare_neighborhood_df_with_neighborhood_column():
    df = pd.DataFrame(
        {
            "tissue": ["T", "T", "N"],
            "sample": ["s1", "s2", "s1"],
            "neighborhood": ["A", "B", "A"],
        }
    )
    result = hf_prepare_neighborhood_df(
        df, "tissue", "sample", neighborhood_column="neighborhood"
    )
    assert "neigh_num" in result.columns
    assert result["neigh_num"].dtype.name == "category"
    assert set(result["neigh_num"].cat.categories) == {0, 1}


def test_hf_prepare_neighborhood_df_without_neighborhood_column():
    df = pd.DataFrame(
        {
            "tissue": ["T"],
            "sample": ["s1"],
        }
    )
    result = hf_prepare_neighborhood_df(df, "tissue", "sample")
    assert "neigh_num" not in result.columns


# ---------------------------------------------------------------------------
# hf_per_only
# ---------------------------------------------------------------------------


def test_hf_per_only_norm_true_returns_percentage_dataframe():
    data = pd.DataFrame(
        {
            "condition": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "replicate": ["r1", "r1", "r2", "r2", "r1", "r1", "r2", "r2"],
            "tissue": ["X"] * 8,
            "cell_type": ["CD4", "CD8", "CD4", "CD8", "CD4", "CD8", "CD4", "CD8"],
        }
    )
    result = hf_per_only(
        data,
        grouping="condition",
        replicate="replicate",
        sub_col="tissue",
        sub_list=["X"],
        per_cat="cell_type",
        norm=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert "percentage" in result.columns
    assert "condition" in result.columns
    assert "replicate" in result.columns
    assert pd.api.types.is_numeric_dtype(result["percentage"])


def test_hf_per_only_norm_false_returns_same_structure():
    data = pd.DataFrame(
        {
            "condition": ["A", "A", "B", "B"],
            "replicate": ["r1", "r2", "r1", "r2"],
            "tissue": ["X"] * 4,
            "cell_type": ["CD4", "CD8", "CD4", "CD8"],
        }
    )
    result = hf_per_only(
        data,
        grouping="condition",
        replicate="replicate",
        sub_col="tissue",
        sub_list=["X"],
        per_cat="cell_type",
        norm=False,
    )
    assert isinstance(result, pd.DataFrame)
    assert "percentage" in result.columns


# ---------------------------------------------------------------------------
# hf_list_folders
# ---------------------------------------------------------------------------


def test_hf_list_folders_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "subdir_a"))
        os.makedirs(os.path.join(tmpdir, "subdir_b"))
        open(os.path.join(tmpdir, "file.txt"), "w").close()
        result = hf_list_folders(tmpdir)
        assert isinstance(result, list)
        assert "subdir_a" in result
        assert "subdir_b" in result
        assert "file.txt" not in result
        assert len(result) == 2


def test_hf_list_folders_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = hf_list_folders(tmpdir)
        assert result == []


def test_hf_list_folders_returns_bare_names():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "sub"))
        result = hf_list_folders(tmpdir)
        for name in result:
            assert os.sep not in name, "should return bare folder names, not full paths"


# ---------------------------------------------------------------------------
# hf_get_tif_filepaths
# ---------------------------------------------------------------------------


def test_hf_get_tif_filepaths_recursive():
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "sub")
        os.makedirs(subdir)
        top_tif = os.path.join(tmpdir, "top.tif")
        nested_tif = os.path.join(subdir, "nested.tif")
        png_file = os.path.join(subdir, "not_tif.png")
        for p in [top_tif, nested_tif, png_file]:
            open(p, "w").close()
        result = hf_get_tif_filepaths(tmpdir)
        assert any("top.tif" in p for p in result)
        assert any("nested.tif" in p for p in result)
        assert not any("not_tif.png" in p for p in result)
        assert len(result) == 2


def test_hf_get_tif_filepaths_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = hf_get_tif_filepaths(tmpdir)
        assert result == []


def test_hf_get_tif_filepaths_returns_full_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        tif = os.path.join(tmpdir, "sample.tif")
        open(tif, "w").close()
        result = hf_get_tif_filepaths(tmpdir)
        assert len(result) == 1
        assert os.path.isabs(result[0]), "should return absolute paths"
        assert result[0].endswith(".tif")


# ---------------------------------------------------------------------------
# make_anndata: obs_names_col
# ---------------------------------------------------------------------------


def _make_df_nn(unique_cell_id_col=True):
    # 2 protein markers (col_sum=1 -> X = first two columns), followed by
    # obs metadata including a duplicate-across-regions 'label' and a
    # (usually) unique 'unique_cell_id'.
    data = {
        "marker1": [1.0, 2.0, 3.0, 4.0],
        "marker2": [5.0, 6.0, 7.0, 8.0],
        "x": [0.0, 1.0, 0.0, 1.0],
        "y": [0.0, 1.0, 0.0, 1.0],
        "label": [1, 2, 1, 2],
        "unique_region": ["reg1", "reg1", "reg2", "reg2"],
    }
    if unique_cell_id_col:
        data["unique_cell_id"] = ["reg1_1", "reg1_2", "reg2_1", "reg2_2"]
    return pd.DataFrame(data)


def test_make_anndata_sets_obs_names_from_unique_cell_id():
    df_nn = _make_df_nn(unique_cell_id_col=True)
    adata = make_anndata(df_nn=df_nn, col_sum=1, nonFuncAb_list=[])
    assert adata.obs_names.is_unique
    assert list(adata.obs_names) == ["reg1_1", "reg1_2", "reg2_1", "reg2_2"]
    # label/unique_region survive as ordinary obs columns for mask lookup
    assert list(adata.obs["label"]) == [1, 2, 1, 2]
    assert list(adata.obs["unique_region"]) == ["reg1", "reg1", "reg2", "reg2"]


def test_make_anndata_degrades_gracefully_without_id_column():
    df_nn = _make_df_nn(unique_cell_id_col=False)
    # should not raise even though 'unique_cell_id' is absent
    adata = make_anndata(df_nn=df_nn, col_sum=1, nonFuncAb_list=[])
    assert adata.n_obs == 4
    # falls back to AnnData's default index-based obs_names
    assert list(adata.obs_names) == list(
        df_nn.index.astype(str)
    )  # AnnData stringifies obs_names


def test_make_anndata_warns_on_non_unique_obs_names_col():
    df_nn = _make_df_nn(unique_cell_id_col=True)
    df_nn["unique_cell_id"] = ["dup", "dup", "reg2_1", "reg2_2"]
    with pytest.warns(UserWarning, match="not unique"):
        make_anndata(df_nn=df_nn, col_sum=1, nonFuncAb_list=[])


def test_make_anndata_obs_names_col_none_uses_default_index():
    df_nn = _make_df_nn(unique_cell_id_col=True)
    adata = make_anndata(df_nn=df_nn, col_sum=1, nonFuncAb_list=[], obs_names_col=None)
    assert list(adata.obs_names) == list(
        df_nn.index.astype(str)
    )  # AnnData stringifies obs_names


# ---------------------------------------------------------------------------
# hf_attach_masks / hf_load_masks
# ---------------------------------------------------------------------------


def _tiny_adata(regions):
    n = len(regions)
    return ad.AnnData(
        X=np.zeros((n, 2)),
        obs=pd.DataFrame({"unique_region": regions}, index=[str(i) for i in range(n)]),
    )


def test_hf_attach_masks_basic():
    adata = _tiny_adata(["reg1", "reg1", "reg2"])
    mask1 = np.array([[1, 1], [0, 2]], dtype=np.int32)
    mask2 = np.array([[1, 0], [0, 0]], dtype=np.int32)
    result = hf_attach_masks(adata, {"reg1": mask1, "reg2": mask2})
    assert result is adata
    assert set(adata.uns["masks"].keys()) == {"reg1", "reg2"}
    np.testing.assert_array_equal(adata.uns["masks"]["reg1"], mask1)
    np.testing.assert_array_equal(adata.uns["masks"]["reg2"], mask2)


def test_hf_attach_masks_merges_incrementally():
    adata = _tiny_adata(["reg1", "reg2"])
    mask1 = np.array([[1, 0]], dtype=np.int32)
    mask2 = np.array([[2, 0]], dtype=np.int32)
    hf_attach_masks(adata, {"reg1": mask1})
    hf_attach_masks(adata, {"reg2": mask2})
    assert set(adata.uns["masks"].keys()) == {"reg1", "reg2"}


def test_hf_attach_masks_warns_on_unknown_region_key():
    adata = _tiny_adata(["reg1"])
    mask_unknown = np.array([[1, 0]], dtype=np.int32)
    with pytest.warns(UserWarning, match="not found"):
        hf_attach_masks(adata, {"reg_missing": mask_unknown})


def test_hf_load_masks_and_attach_roundtrip(tmp_path):
    mask = np.array([[1, 1, 2], [0, 2, 2], [3, 0, 0]], dtype=np.int32)
    mask_path = pathlib.Path(tmp_path) / "reg1_mask.tif"
    tifffile.imwrite(mask_path, mask)

    loaded = hf_load_masks({"reg1": mask_path})
    np.testing.assert_array_equal(loaded["reg1"], mask)

    adata = _tiny_adata(["reg1"])
    hf_attach_masks(adata, loaded)
    attached = adata.uns["masks"]["reg1"]
    np.testing.assert_array_equal(attached, mask)
    # pixel round-trip: a label isolates exactly its own pixels
    assert (attached == 1).sum() == 2
    assert (attached == 2).sum() == 3
    assert (attached == 3).sum() == 1
