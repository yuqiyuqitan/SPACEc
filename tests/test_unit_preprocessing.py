"""Unit tests for spacec.preprocessing._general.read_segdf and
compensate_cell_matrix, focused on the unique_cell_id / label plumbing that
keeps cells traceable back to their segmentation mask across multiple
concatenated regions -- and, for compensate_cell_matrix, on joining
compensated intensities back onto that mask by label rather than by row
position (see test_compensation_is_invariant_to_dataframe_row_order below
for the corruption this guards against).
"""

import pathlib
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacec.preprocessing._general import (
    compensate_cell_matrix,
    filter_data,
    format,
    plot_filter_dashboard,
    read_segdf,
)

TEST_DIR = pathlib.Path(__file__).parent
CELLSEG_DIR = TEST_DIR / "data" / "processed" / "cellseg"


def _write_cellpose_csv(path, labels, extra_cols=None):
    """Mimic the CSV shape written by extract_features(): 'label' first
    (from markers_df.reset_index()), followed by feature columns."""
    data = {"label": labels}
    data.update(extra_cols or {})
    pd.DataFrame(data).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# unique_cell_id uniqueness across regions
# ---------------------------------------------------------------------------


def test_read_segdf_unique_cell_id_default(tmp_path):
    csv1 = tmp_path / "reg1_features.csv"
    csv2 = tmp_path / "reg2_features.csv"
    _write_cellpose_csv(csv1, labels=[1, 2, 3], extra_cols={"DAPI": [10, 20, 30]})
    _write_cellpose_csv(csv2, labels=[1, 2, 3], extra_cols={"DAPI": [40, 50, 60]})

    df = read_segdf(
        segfile_list=[csv1, csv2],
        seg_method="cellpose",
        region_list=["reg1", "reg2"],
    )

    assert "unique_cell_id" in df.columns
    assert df["unique_cell_id"].is_unique
    # raw labels collide across regions even though unique_cell_id doesn't
    assert not df["label"].is_unique
    assert df["unique_cell_id"].tolist() == [
        "reg1_1",
        "reg1_2",
        "reg1_3",
        "reg2_1",
        "reg2_2",
        "reg2_3",
    ]


def test_read_segdf_label_column_present_for_cellpose(tmp_path):
    csv1 = tmp_path / "reg1_features.csv"
    _write_cellpose_csv(csv1, labels=[5, 6, 7], extra_cols={"DAPI": [1, 2, 3]})

    df = read_segdf(segfile_list=[csv1], seg_method="cellpose", region_list=["reg1"])

    assert "label" in df.columns
    assert df["label"].tolist() == [5, 6, 7]
    assert (df["unique_cell_id"] == "reg1_" + df["label"].astype(str)).all()


def test_read_segdf_add_unique_id_false_preserves_legacy_output(tmp_path):
    csv1 = tmp_path / "reg1_features.csv"
    csv2 = tmp_path / "reg2_features.csv"
    _write_cellpose_csv(csv1, labels=[1, 2], extra_cols={"DAPI": [10, 20]})
    _write_cellpose_csv(csv2, labels=[1, 2], extra_cols={"DAPI": [30, 40]})

    df = read_segdf(
        segfile_list=[csv1, csv2],
        seg_method="cellpose",
        region_list=["reg1", "reg2"],
        add_unique_id=False,
    )

    assert "unique_cell_id" not in df.columns
    assert "label" not in df.columns  # label stays as the (colliding) index
    assert df.index.name == "label"
    assert list(df.index) == [1, 2, 1, 2]


# ---------------------------------------------------------------------------
# legacy 'cellseg' branch unaffected
# ---------------------------------------------------------------------------


def test_read_segdf_cellseg_branch_unaffected():
    df = read_segdf(
        segfile_list=[
            CELLSEG_DIR / "reg010_X01_Y01_Z01_compensated.csv",
            CELLSEG_DIR / "reg001_X01_Y01_Z01_compensated.csv",
        ],
        seg_method="cellseg",
        region_list=["reg010", "reg001"],
        meta_list=["tonsillitis", "tonsil"],
    )

    # legacy behavior: 'cell_id' produced by the existing branch-specific logic
    assert "cell_id" in df.columns
    assert not df["cell_id"].is_unique  # restarts near 1 per region

    # new behavior: unique_cell_id is unique; no redundant 'label' column is
    # added for this branch (cell_id already serves that role)
    assert "unique_cell_id" in df.columns
    assert df["unique_cell_id"].is_unique
    assert "label" not in df.columns


# ---------------------------------------------------------------------------
# input validation (pre-existing behavior, unaffected by this change)
# ---------------------------------------------------------------------------


def test_read_segdf_region_list_length_mismatch_exits(tmp_path):
    csv1 = tmp_path / "reg1_features.csv"
    _write_cellpose_csv(csv1, labels=[1], extra_cols={"DAPI": [10]})

    with pytest.raises(SystemExit):
        read_segdf(
            segfile_list=[csv1],
            seg_method="cellpose",
            region_list=["reg1", "reg2"],
        )


# ---------------------------------------------------------------------------
# filtered-out labels: present in mask, absent from features/AnnData
# ---------------------------------------------------------------------------


def test_filtered_label_absent_from_features_but_present_in_mask(tmp_path):
    """Simulate cell_segmentation(size_cutoff=...) dropping a small object:
    its label has no row in the features CSV, but the raw mask array (as
    returned/saved by cell_segmentation) still contains its pixels. Confirm
    the full read_segdf -> make_anndata -> hf.attach_masks pipeline reflects
    this correctly."""
    import spacec as sp

    # A tiny synthetic mask with 3 labeled objects; label 3 will be treated
    # as "filtered out by size_cutoff" (absent from the features CSV).
    mask_reg1 = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 2, 2, 0],
            [0, 0, 2, 2, 0],
            [0, 0, 0, 0, 3],
        ],
        dtype=np.int32,
    )

    csv1 = tmp_path / "reg1_features.csv"
    # Columns mirror extract_features() output ordering: label first, then a
    # marker column, then spatial columns -- label 3 has no row (filtered).
    pd.DataFrame(
        {
            "label": [1, 2],
            "DAPI": [100.0, 200.0],
            "x": [0.5, 2.5],
            "y": [0.5, 2.5],
        }
    ).to_csv(csv1, index=False)

    df = read_segdf(segfile_list=[csv1], seg_method="cellpose", region_list=["reg1"])
    adata = sp.hf.make_anndata(df_nn=df, col_sum=0, nonFuncAb_list=[])
    adata = sp.hf.hf_attach_masks(adata, {"reg1": mask_reg1})

    assert adata.obs_names.is_unique
    assert set(adata.obs["label"]) == {1, 2}
    assert 3 not in set(adata.obs["label"])

    attached_mask = adata.uns["masks"]["reg1"]
    # the filtered-out label's pixels are still present in the raw mask
    assert (attached_mask == 3).sum() == 1
    # a surviving cell's label still isolates exactly its own pixels
    assert (attached_mask == 1).sum() == 4
    assert (attached_mask == 2).sum() == 4


# ---------------------------------------------------------------------------
# compensate_cell_matrix: known-answer geometry
# ---------------------------------------------------------------------------
#
# Two fixtures with a closed-form (or at least closed-direction) expected
# result, used throughout the tests below:
#   - _isolated_cells_fixture: two non-touching cells. The adjacency matrix
#     is the identity, so lstsq(I, means) == means exactly -- compensation
#     is a no-op. This is an exact answer, not a regression snapshot.
#   - _touching_cells_fixture: two cells sharing a border, one brighter than
#     the other. Compensation must move both intensities down (each cell
#     "gives" some signal to its neighbor), and the dimmer cell must lose
#     more than the brighter one, since a larger fraction of its measured
#     signal near the shared border came from bleed. Exact resulting values
#     are geometry-dependent, so only direction/ordering is asserted.


def _isolated_cells_fixture():
    mask = np.zeros((12, 12), dtype=np.int32)
    mask[1:3, 1:3] = 1
    mask[1:3, 6:8] = 2
    img = np.zeros((12, 12), dtype=np.float32)
    img[mask == 1] = 100.0
    img[mask == 2] = 50.0
    df = pd.DataFrame({"label": [1, 2], "MarkerA": [100.0, 50.0]})
    return df, {"MarkerA": img}, mask


def _touching_cells_fixture():
    mask = np.zeros((12, 12), dtype=np.int32)
    mask[2:6, 2:4] = 1  # bright cell
    mask[2:6, 4:6] = 2  # dim cell, shares a border with cell 1
    img = np.zeros((12, 12), dtype=np.float32)
    img[mask == 1] = 100.0
    img[mask == 2] = 50.0
    df = pd.DataFrame({"label": [1, 2], "MarkerA": [100.0, 50.0]})
    return df, {"MarkerA": img}, mask


def test_compensation_is_identity_for_isolated_cells():
    df, image_dict, mask = _isolated_cells_fixture()

    out = compensate_cell_matrix(
        df,
        image_dict=image_dict,
        masks=mask,
        overwrite=False,
        verbose=False,
    )

    # non-touching cells have an identity adjacency matrix, so
    # lstsq(I, means) == means exactly: this is a closed-form answer.
    assert np.allclose(
        out["MarkerA_compensated"], [100.0, 50.0]
    ), "isolated (non-touching) cells should be unaffected by compensation"


def test_compensation_removes_bleed_from_dimmer_neighbor():
    df, image_dict, mask = _touching_cells_fixture()

    out = compensate_cell_matrix(
        df,
        image_dict=image_dict,
        masks=mask,
        overwrite=False,
        verbose=False,
    )
    raw = df.set_index("label")["MarkerA"]
    comp = out.set_index("label")["MarkerA_compensated"]

    assert (
        comp[1] < raw[1]
    ), "the bright cell should lose signal borrowed by its dim neighbor"
    assert comp[2] < raw[2], "the dim cell should lose some of its own bled-in signal"
    assert (raw[2] - comp[2]) > (raw[1] - comp[1]), (
        "the dimmer neighbor should lose more absolute signal than the "
        "brighter one: a larger share of its raw intensity near the shared "
        "border came from the brighter cell bleeding in"
    )


# ---------------------------------------------------------------------------
# compensate_cell_matrix: the label-join fix (regression coverage)
# ---------------------------------------------------------------------------


def test_compensation_is_invariant_to_dataframe_row_order():
    """Regression test for the corruption this function used to have: it
    joined compensated values onto df by row *position*, not by mask label,
    so any row reordering silently swapped cells' intensities."""
    mask = np.zeros((12, 12), dtype=np.int32)
    mask[1:3, 1:3] = 1
    mask[1:3, 6:8] = 2
    mask[6:8, 1:3] = 3
    mask[6:8, 6:8] = 4
    img = np.zeros((12, 12), dtype=np.float32)
    label_values = {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}
    for label, value in label_values.items():
        img[mask == label] = value

    # rows deliberately NOT in mask-label order
    df = pd.DataFrame({"label": [4, 3, 2, 1], "MarkerA": [40.0, 30.0, 20.0, 10.0]})

    out = compensate_cell_matrix(
        df,
        image_dict={"MarkerA": img},
        masks=mask,
        overwrite=True,
        verbose=False,
    )

    for _, row in out.iterrows():
        assert row["MarkerA"] == pytest.approx(label_values[row["label"]]), (
            f"row for label {row['label']} got the wrong cell's intensity -- "
            "the join must follow the 'label' column, not row position"
        )


def test_compensation_is_invariant_to_mask_label_renumbering():
    """Non-consecutive mask labels (e.g. after remove_border_objects, which
    zeroes labels without relabeling) used to crash compute_channel_means_
    sums_compensated. Renumbering labels must not change per-cell results."""
    df_a, image_dict_a, mask_a = _touching_cells_fixture()
    out_a = compensate_cell_matrix(
        df_a,
        image_dict=image_dict_a,
        masks=mask_a,
        overwrite=False,
        verbose=False,
    )

    # same geometry, but labels 1 -> 41 and 2 -> 7 (non-consecutive, out of order)
    mask_b = np.where(mask_a == 1, 41, np.where(mask_a == 2, 7, 0)).astype(np.int32)
    df_b = pd.DataFrame({"label": [41, 7], "MarkerA": [100.0, 50.0]})
    out_b = compensate_cell_matrix(
        df_b,
        image_dict=image_dict_a,
        masks=mask_b,
        overwrite=False,
        verbose=False,
    )

    comp_a = out_a.set_index("label")["MarkerA_compensated"]
    comp_b = out_b.set_index("label")["MarkerA_compensated"]
    assert comp_b[41] == pytest.approx(
        comp_a[1]
    ), "relabeling must not change the bright cell's result"
    assert comp_b[7] == pytest.approx(
        comp_a[2]
    ), "relabeling must not change the dim cell's result"


def test_compensation_ignores_mask_labels_filtered_by_size_cutoff(tmp_path):
    """A mask with more labels than df has rows is the normal case (a
    size_cutoff removes small objects from the features CSV but the mask
    keeps every pixel). This must join correctly on surviving labels, not
    raise a shape-mismatch error."""
    mask = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 2, 2, 0],
            [0, 0, 2, 2, 0],
            [0, 0, 0, 0, 3],
        ],
        dtype=np.int32,
    )
    img = np.zeros((5, 5), dtype=np.float32)
    img[mask == 1] = 100.0
    img[mask == 2] = 200.0
    img[mask == 3] = 300.0  # label 3 was filtered out of the features CSV

    csv1 = tmp_path / "reg1_features.csv"
    pd.DataFrame({"label": [1, 2], "DAPI": [100.0, 200.0]}).to_csv(csv1, index=False)
    df = read_segdf(segfile_list=[csv1], seg_method="cellpose", region_list=["reg1"])

    out = compensate_cell_matrix(
        df,
        image_dict={"DAPI": img},
        masks=mask,
        overwrite=False,
        verbose=False,
    )

    assert list(out["label"]) == [
        1,
        2,
    ], "only surviving labels should appear in the output"
    assert not out["DAPI_compensated"].isna().any()


def test_compensation_raises_when_df_has_label_absent_from_mask():
    """A df label with no corresponding mask pixels means df and masks were
    mismatched (e.g. paired from different regions) -- this must always be
    an error, never a silent join."""
    df, image_dict, mask = _isolated_cells_fixture()
    df.loc[len(df)] = {"label": 999, "MarkerA": 1.0}

    with pytest.raises(ValueError, match="not present in masks"):
        compensate_cell_matrix(
            df,
            image_dict=image_dict,
            masks=mask,
            overwrite=False,
            verbose=False,
        )


def test_compensation_raises_on_mask_image_shape_mismatch():
    df, image_dict, mask = _isolated_cells_fixture()
    image_dict = {"MarkerA": np.zeros((5, 5), dtype=np.float32)}

    with pytest.raises(ValueError, match="does not match mask shape"):
        compensate_cell_matrix(
            df,
            image_dict=image_dict,
            masks=mask,
            overwrite=False,
            verbose=False,
        )


def test_compensation_raises_on_multi_region_df_without_region_argument():
    df, image_dict, mask = _isolated_cells_fixture()
    df["unique_region"] = ["reg001", "reg002"]

    with pytest.raises(ValueError, match="spans multiple regions"):
        compensate_cell_matrix(
            df,
            image_dict=image_dict,
            masks=mask,
            overwrite=False,
            verbose=False,
        )

    # scoping to one region resolves the ambiguity; all rows are kept, but
    # only the given region's rows get compensated ('_compensated' stays
    # NaN for rows outside it)
    out = compensate_cell_matrix(
        df,
        image_dict=image_dict,
        masks=mask,
        overwrite=False,
        verbose=False,
        region="reg001",
    )
    assert list(out["label"]) == [1, 2]
    reg001_row = out[out["unique_region"] == "reg001"].iloc[0]
    reg002_row = out[out["unique_region"] == "reg002"].iloc[0]
    assert not pd.isna(
        reg001_row["MarkerA_compensated"]
    ), "reg001 was requested and should be compensated"
    assert pd.isna(
        reg002_row["MarkerA_compensated"]
    ), "reg002 was not requested and should be left uncompensated"


def test_compensation_warns_and_joins_positionally_without_label_column():
    df, image_dict, mask = _isolated_cells_fixture()
    df = df.drop(columns=["label"])

    with pytest.warns(UserWarning, match="row position"):
        out = compensate_cell_matrix(
            df,
            image_dict=image_dict,
            masks=mask,
            overwrite=False,
            verbose=False,
        )
    assert np.allclose(out["MarkerA_compensated"], [100.0, 50.0])

    # a row-count mismatch can't be joined positionally and must raise
    df_extra = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="cannot join positionally"):
        compensate_cell_matrix(
            df_extra,
            image_dict=image_dict,
            masks=mask,
            overwrite=False,
            verbose=False,
        )


def test_compensation_does_not_mutate_input_dataframe():
    for overwrite in (True, False):
        df, image_dict, mask = _isolated_cells_fixture()
        original = df.copy()

        compensate_cell_matrix(
            df,
            image_dict=image_dict,
            masks=mask,
            overwrite=overwrite,
            verbose=False,
        )

        assert df.equals(
            original
        ), f"compensate_cell_matrix(overwrite={overwrite}) must not mutate its input df"


def test_compensation_raises_when_overwrite_true_and_channel_missing_from_df():
    df, image_dict, mask = _isolated_cells_fixture()
    image_dict = {**image_dict, "MarkerB": image_dict["MarkerA"]}  # not a column in df

    with pytest.raises(ValueError, match="not present in df.columns"):
        compensate_cell_matrix(
            df,
            image_dict=image_dict,
            masks=mask,
            overwrite=True,
            verbose=False,
        )


# ---------------------------------------------------------------------------
# compensate_cell_matrix: exclude_channels
# ---------------------------------------------------------------------------


def _two_channel_touching_fixture():
    """_touching_cells_fixture plus a nuclei channel, so exclusion has
    something to skip and something to still compensate."""
    df, image_dict, mask = _touching_cells_fixture()
    dapi = np.zeros_like(image_dict["MarkerA"])
    dapi[mask == 1] = 80.0
    dapi[mask == 2] = 30.0
    df["DAPI"] = [80.0, 30.0]
    return df, {**image_dict, "DAPI": dapi}, mask


def test_excluded_channel_keeps_raw_values_when_overwriting():
    df, image_dict, mask = _two_channel_touching_fixture()

    out = compensate_cell_matrix(
        df,
        image_dict=image_dict,
        masks=mask,
        overwrite=True,
        exclude_channels=["DAPI"],
        verbose=False,
    )

    assert list(out["DAPI"]) == [
        80.0,
        30.0,
    ], "an excluded channel must keep its raw values under overwrite=True"
    assert not np.allclose(
        out["MarkerA"], df["MarkerA"]
    ), "the non-excluded channel should still have been compensated"


def test_excluded_channel_gets_no_compensated_column():
    df, image_dict, mask = _two_channel_touching_fixture()

    out = compensate_cell_matrix(
        df,
        image_dict=image_dict,
        masks=mask,
        overwrite=False,
        exclude_channels="DAPI",  # a bare string behaves like a 1-element list
        verbose=False,
    )

    assert "MarkerA_compensated" in out.columns
    assert "DAPI_compensated" not in out.columns


def test_excluding_a_channel_does_not_change_the_others():
    """Compensation unmixes neighbouring cells within one channel; channels
    are solved independently, so dropping one must not move another."""
    df, image_dict, mask = _two_channel_touching_fixture()

    kwargs = dict(masks=mask, overwrite=False, verbose=False)
    with_dapi = compensate_cell_matrix(df, image_dict=image_dict, **kwargs)
    without_dapi = compensate_cell_matrix(
        df, image_dict=image_dict, exclude_channels=["DAPI"], **kwargs
    )

    assert np.allclose(
        with_dapi["MarkerA_compensated"], without_dapi["MarkerA_compensated"]
    )


def test_excluded_channel_need_not_be_a_column_in_df():
    """image_dict normally holds the nuclei channel even when df doesn't --
    excluding it must satisfy the overwrite=True column check."""
    df, image_dict, mask = _two_channel_touching_fixture()
    df = df.drop(columns=["DAPI"])

    out = compensate_cell_matrix(
        df,
        image_dict=image_dict,
        masks=mask,
        overwrite=True,
        exclude_channels=["DAPI"],
        verbose=False,
    )
    assert "DAPI" not in out.columns

    # ... and without the exclusion it is still an error, as before
    with pytest.raises(ValueError, match="not present in df.columns"):
        compensate_cell_matrix(
            df,
            image_dict=image_dict,
            masks=mask,
            overwrite=True,
            verbose=False,
        )


def test_compensation_raises_on_unknown_excluded_channel():
    df, image_dict, mask = _two_channel_touching_fixture()

    with pytest.raises(ValueError, match="are not channels in image_dict"):
        compensate_cell_matrix(
            df,
            image_dict=image_dict,
            masks=mask,
            overwrite=False,
            exclude_channels=["DAPl"],  # lowercase L, not a 1 -- a typo
            verbose=False,
        )


def test_compensation_raises_when_all_channels_are_excluded():
    df, image_dict, mask = _two_channel_touching_fixture()

    with pytest.raises(ValueError, match="nothing left to compensate"):
        compensate_cell_matrix(
            df,
            image_dict=image_dict,
            masks=mask,
            overwrite=False,
            exclude_channels=["MarkerA", "DAPI"],
            verbose=False,
        )


# ---------------------------------------------------------------------------
# The compensation solve itself: sparse, exact, and with a safe fallback
# ---------------------------------------------------------------------------


def _blobby_mask(n_seeds=400, size=200, seed=3):
    """A Voronoi-like field of touching cells, with background carved
    between some pairs -- the shape a real segmentation mask has, and dense
    enough that most cells actually have neighbours to bleed into."""
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(seed)
    pts = np.zeros((size, size), bool)
    pts[rng.integers(0, size, n_seeds), rng.integers(0, size, n_seeds)] = True
    markers, _ = ndimage.label(pts)
    _, (iy, ix) = ndimage.distance_transform_edt(markers == 0, return_indices=True)
    lab = markers[iy, ix].astype(np.int32)
    eroded = ndimage.grey_erosion(lab, size=3)
    return np.where((eroded == lab) | (rng.random((size, size)) > 0.5), lab, 0).astype(
        np.int32
    )


def test_compensation_solve_is_exact():
    """The adjacency matrix is square, so its least-squares solution is its
    exact solution. Solving it directly (and in float64) must therefore
    satisfy A x = m to machine precision -- the dense float32 path this
    replaced left a residual around 1e-3.

    Cells whose solution is clipped at zero are excluded: that clip is part
    of the published CellSeg method and deliberately breaks the equality.
    """
    from scipy.sparse.linalg import spsolve

    from spacec.preprocessing._general import ImageProcessor

    mask = _blobby_mask()
    rng = np.random.default_rng(11)
    channels = [(rng.random(mask.shape) * 500).astype(np.float32) for _ in range(4)]

    processor = ImageProcessor(mask)
    comp, means, _ = processor.compute_channel_means_sums_compensated(channels)

    adj = processor._build_adjacency()
    unclipped = np.asarray(spsolve(adj, means)).reshape(means.shape)
    assert np.abs(adj @ unclipped - means).max() < 1e-9, "solve is not exact"
    assert np.array_equal(comp, np.maximum(unclipped, 0))
    assert (comp >= 0).all()


def test_compensation_adjacency_matrix_is_sparse():
    """A cell touches only a handful of neighbours, so the matrix must stay
    sparse -- this is what keeps memory linear in cell count instead of
    quadratic (57 GB dense at 120,000 cells)."""
    from spacec.preprocessing._general import ImageProcessor

    processor = ImageProcessor(_blobby_mask())
    adj = processor._build_adjacency()
    n_masks = len(processor.labels)

    assert adj.shape == (n_masks, n_masks)
    assert adj.nnz < 20 * n_masks, "adjacency is denser than a segmentation should be"
    # Unit diagonal, as CellSeg sets it after normalizing each row.
    assert np.allclose(adj.diagonal(), 1.0)


def test_compensation_falls_back_to_dense_lstsq_when_singular(monkeypatch):
    """A singular adjacency matrix must not silently yield NaNs. It should
    warn and fall back to the dense least-squares solver CellSeg used,
    which returns the minimum-norm solution."""
    from scipy.sparse.linalg import MatrixRankWarning

    from spacec.preprocessing import _general
    from spacec.preprocessing._general import ImageProcessor

    _, image_dict, mask = _isolated_cells_fixture()
    processor = ImageProcessor(mask)
    channels = [image_dict["MarkerA"]]
    expected, _, _ = processor.compute_channel_means_sums_compensated(channels)

    def singular(a, b, *args, **kwargs):
        warnings.warn("matrix is exactly singular", MatrixRankWarning)
        return np.full(b.shape, np.nan)

    monkeypatch.setattr(_general, "spsolve", singular)

    with pytest.warns(UserWarning, match="singular"):
        fallback, _, _ = processor.compute_channel_means_sums_compensated(channels)

    assert np.all(np.isfinite(fallback))
    assert np.allclose(fallback, expected)


def test_compensation_handles_non_consecutive_mask_labels():
    """Mask labels need not be consecutive (a size_cutoff during
    segmentation leaves gaps). Row i of the result must correspond to
    processor.labels[i], not to label i + 1."""
    from spacec.preprocessing._general import ImageProcessor

    mask = np.zeros((12, 12), dtype=np.int32)
    mask[1:3, 1:3] = 7  # gaps on both sides of the label space
    mask[1:3, 6:8] = 42
    channel = np.zeros((12, 12), dtype=np.float32)
    channel[mask == 7] = 100.0
    channel[mask == 42] = 50.0

    processor = ImageProcessor(mask)
    assert np.array_equal(processor.labels, [7, 42])

    comp, means, counts = processor.compute_channel_means_sums_compensated([channel])
    assert np.allclose(means.ravel(), [100.0, 50.0])
    # These two cells don't touch, so compensation is a no-op.
    assert np.allclose(comp.ravel(), [100.0, 50.0])
    assert np.array_equal(counts, [4, 4])


# ---------------------------------------------------------------------------
# format(): invalid method raises ValueError, not SystemExit
# ---------------------------------------------------------------------------


def test_format_invalid_method_raises_value_error():
    """format()'s docstring documents 'Raises ValueError', but it used to
    call bare exit() instead, which raises SystemExit and kills the kernel
    in a notebook rather than being catchable."""
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="not supported"):
        format(df, list_out=[], list_keep=[], method="not_a_real_method")


# ---------------------------------------------------------------------------
# filter_data() / plot_filter_dashboard(): QC-flag mechanism, dashboard
# layout, and the Plotting-API-contract fixes made alongside them
# (no plt.show(), fig returned, return shape backward compatible).
# ---------------------------------------------------------------------------


def _make_filter_df(n=600, seed=0, n_regions=2):
    """Synthetic segmentation-like DataFrame with an area/DAPI distribution
    wide enough that 1%/5%/99% percentile thresholds all land somewhere
    non-trivial, plus a couple of marker columns for zcount_thres-adjacent
    use elsewhere. Index is intentionally non-unique across regions, matching
    what real concatenated multi-region data looks like (see
    test_filter_data_handles_non_unique_index below for why that matters)."""
    rng = np.random.default_rng(seed)
    region_ids = np.arange(n) % n_regions
    df = pd.DataFrame(
        {
            "DAPI": rng.normal(100, 30, n).clip(min=1),
            "area": rng.normal(200, 60, n).clip(min=1),
            "region_num": region_ids,
        }
    )
    # non-unique index: each region's own local 0..k-1 numbering, like a concat of
    # several per-region CSVs without ignore_index=True
    df.index = pd.Index(
        np.concatenate([np.arange((region_ids == r).sum()) for r in range(n_regions)])
    )
    return df


def _thresholds_for(df, drop_pct=1, flag_pct=5, upper_flag_pct=99):
    nuc_thres = np.percentile(df["DAPI"], drop_pct)
    size_thres = np.percentile(df["area"], drop_pct)
    nuc_flag = np.percentile(df["DAPI"], flag_pct)
    size_flag = np.percentile(df["area"], flag_pct)
    size_upper_flag = np.percentile(df["area"], upper_flag_pct)
    return nuc_thres, size_thres, nuc_flag, size_flag, size_upper_flag


def test_filter_data_plain_call_is_backward_compatible():
    """No qc_dashboard, no flag thresholds -> the original behaviour:
    a single DataFrame, no qc_flag_* columns, nothing printed."""
    df = _make_filter_df()
    nuc_thres, size_thres, *_ = _thresholds_for(df)

    result = filter_data(
        df, nuc_thres=nuc_thres, size_thres=size_thres, region_column="region_num"
    )

    assert isinstance(result, pd.DataFrame)
    assert not any(c.startswith("qc_flag") for c in result.columns)
    assert (result["DAPI"] > nuc_thres).all()
    assert (result["area"] > size_thres).all()


def test_filter_data_silent_without_dashboard_or_flags(capsys):
    """Nothing should print when there's neither a dashboard nor any flags
    -- matches the pre-flag function's own silent default."""
    df = _make_filter_df()
    nuc_thres, size_thres, *_ = _thresholds_for(df)

    filter_data(
        df, nuc_thres=nuc_thres, size_thres=size_thres, region_column="region_num"
    )

    assert capsys.readouterr().out == ""


def test_filter_data_flags_print_even_without_dashboard(capsys):
    """Flag rates used to print unconditionally regardless of `plot`/
    `qc_dashboard` -- folding them into the "Filter summary" block must not
    silently drop that behaviour for the flags-but-no-dashboard case."""
    df = _make_filter_df()
    nuc_thres, size_thres, nuc_flag, size_flag, size_upper_flag = _thresholds_for(df)

    filter_data(
        df,
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        region_column="region_num",
        nuc_flag_thres=nuc_flag,
        size_flag_thres=size_flag,
        size_upper_flag_thres=size_upper_flag,
    )

    out = capsys.readouterr().out
    assert "Filter summary" in out
    assert "Flagged (low DAPI" in out


def test_filter_data_qc_flag_columns_match_manual_computation():
    df = _make_filter_df()
    nuc_thres, size_thres, nuc_flag, size_flag, size_upper_flag = _thresholds_for(df)

    result = filter_data(
        df,
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        region_column="region_num",
        nuc_flag_thres=nuc_flag,
        size_flag_thres=size_flag,
        size_upper_flag_thres=size_upper_flag,
    )

    assert (result["qc_flag_low_dapi"] == (result["DAPI"] <= nuc_flag)).all()
    assert (result["qc_flag_low_size"] == (result["area"] <= size_flag)).all()
    assert (result["qc_flag_large_size"] == (result["area"] >= size_upper_flag)).all()
    expected_any = result[
        ["qc_flag_low_dapi", "qc_flag_low_size", "qc_flag_large_size"]
    ].any(axis=1)
    assert (result["qc_flag_any"] == expected_any).all()
    # flag thresholds are looser than the drop thresholds -- every surviving cell already
    # passed nuc_thres/size_thres, so nothing here should be flagged AND already-dropped
    assert (result["DAPI"] > nuc_thres).all() and (result["area"] > size_thres).all()


@pytest.mark.parametrize(
    "which_flags",
    [
        [],
        ["nuc"],
        ["size"],
        ["size_upper"],
        ["nuc", "size"],
        ["nuc", "size", "size_upper"],
    ],
    ids=[
        "none",
        "nuc_only",
        "size_only",
        "size_upper_only",
        "nuc_and_size",
        "all_three",
    ],
)
def test_filter_data_qc_dashboard_survives_partial_flag_combinations(which_flags):
    """The dashboard (and its panel B2 combination bars specifically) must
    not crash for any subset of the three flag thresholds being set --
    exercises 0, 1, 2, and 3 simultaneous flags."""
    df = _make_filter_df(n=800)
    nuc_thres, size_thres, nuc_flag, size_flag, size_upper_flag = _thresholds_for(df)
    kwargs = {}
    if "nuc" in which_flags:
        kwargs["nuc_flag_thres"] = nuc_flag
    if "size" in which_flags:
        kwargs["size_flag_thres"] = size_flag
    if "size_upper" in which_flags:
        kwargs["size_upper_flag_thres"] = size_upper_flag

    df_out, fig = filter_data(
        df,
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        region_column="region_num",
        qc_dashboard=True,
        **kwargs,
    )
    try:
        n_flags_set = len(which_flags)
        expected_panels = 6 if n_flags_set > 0 else 5
        assert len(fig.get_axes()) == expected_panels
        expected_flag_cols = {
            "nuc": "qc_flag_low_dapi",
            "size": "qc_flag_low_size",
            "size_upper": "qc_flag_large_size",
        }
        for key in which_flags:
            assert expected_flag_cols[key] in df_out.columns
    finally:
        plt.close(fig)


def test_filter_data_qc_dashboard_never_calls_plt_show(monkeypatch):
    """Plotting-API-contract: no plt.show()/fig.show() inside a plotting
    function -- display is 100% caller-controlled."""
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))

    df = _make_filter_df()
    nuc_thres, size_thres, nuc_flag, size_flag, size_upper_flag = _thresholds_for(df)
    df_out, fig = filter_data(
        df,
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        region_column="region_num",
        qc_dashboard=True,
        nuc_flag_thres=nuc_flag,
        size_flag_thres=size_flag,
        size_upper_flag_thres=size_upper_flag,
    )
    try:
        assert shown == [], "plot_filter_dashboard (via filter_data) called plt.show()"
        assert isinstance(fig, plt.Figure)
    finally:
        plt.close(fig)


def test_filter_data_return_shape_is_positional_and_additive():
    """qc_dashboard_fig and summary are appended in a fixed order regardless
    of which combination is requested -- df_nuc always first."""
    df = _make_filter_df()
    nuc_thres, size_thres, *_ = _thresholds_for(df)

    only_df = filter_data(
        df, nuc_thres=nuc_thres, size_thres=size_thres, region_column="region_num"
    )
    assert isinstance(only_df, pd.DataFrame)

    df_out, summary = filter_data(
        df,
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        region_column="region_num",
        return_summary=True,
    )
    assert isinstance(df_out, pd.DataFrame) and isinstance(summary, pd.DataFrame)

    df_out, fig = filter_data(
        df,
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        region_column="region_num",
        qc_dashboard=True,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    df_out, fig, summary = filter_data(
        df,
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        region_column="region_num",
        qc_dashboard=True,
        return_summary=True,
    )
    assert isinstance(fig, plt.Figure) and isinstance(summary, pd.DataFrame)
    plt.close(fig)


def test_filter_data_summary_matches_or_exceeds_printed_detail():
    """The returned summary table must carry (at least) the same detail as
    the printed text: per-region counts for every removal reason and every
    flag actually present, summable back up to n_total/n_removed."""
    df = _make_filter_df(n=1000, n_regions=3)
    nuc_thres, size_thres, nuc_flag, size_flag, size_upper_flag = _thresholds_for(df)

    df_out, summary = filter_data(
        df,
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        region_column="region_num",
        nuc_flag_thres=nuc_flag,
        size_flag_thres=size_flag,
        size_upper_flag_thres=size_upper_flag,
        return_summary=True,
    )

    assert {"n_total", "n_kept", "n_removed", "pct_kept", "pct_flagged"} <= set(
        summary.columns
    )
    assert (summary["n_total"] == summary["n_kept"] + summary["n_removed"]).all()

    removed_cols = [c for c in summary.columns if c.startswith("removed: ")]
    assert removed_cols, "expected at least one 'removed: <reason>' column"
    assert (summary[removed_cols].sum(axis=1) == summary["n_removed"]).all()

    flagged_cols = [
        c
        for c in summary.columns
        if c.startswith("flagged: ") and c != "flagged: any flag"
    ]
    assert flagged_cols
    # every individual flagged count is <= the combined "any flag" count, per region
    assert (summary[flagged_cols].max(axis=1) <= summary["flagged: any flag"]).all()
    # column names are human-readable, not literal qc_flag_* identifiers
    assert not any("qc_flag" in c for c in summary.columns)


def test_filter_data_return_summary_no_flags_has_no_flag_columns():
    df = _make_filter_df()
    nuc_thres, size_thres, *_ = _thresholds_for(df)

    _, summary = filter_data(
        df,
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        region_column="region_num",
        return_summary=True,
    )

    assert "pct_flagged" not in summary.columns
    assert not any(c.startswith("flagged: ") for c in summary.columns)


def test_filter_data_handles_non_unique_index():
    """Real bug found on real (multi-region-concatenated) data: propagating
    the new qc_flag_* columns from df_nuc back onto df_flagged via
    `.loc[df_nuc.index]` breaks with a non-unique index (each duplicate
    label matches every row sharing it) -- ValueError: Must have equal len
    keys and value. `_make_filter_df()`'s index is deliberately non-unique
    across regions for exactly this reason."""
    df = _make_filter_df(n=500, n_regions=3)
    assert not df.index.is_unique
    nuc_thres, size_thres, nuc_flag, size_flag, size_upper_flag = _thresholds_for(df)

    df_out, fig = filter_data(
        df,
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        region_column="region_num",
        qc_dashboard=True,
        nuc_flag_thres=nuc_flag,
        size_flag_thres=size_flag,
        size_upper_flag_thres=size_upper_flag,
    )
    try:
        assert len(df_out) == df_out["qc_flag_any"].notna().sum()
    finally:
        plt.close(fig)


def test_filter_data_combo_bar_denominator_regression():
    """Real bug: panel B2's multi-flag combination bars were computed with
    value_counts(normalize=True) on the already-filtered-to-multi-flag
    subset, silently normalising against that subset's own total instead of
    all kept cells -- a combo affecting every multi-flagged cell read as
    ~100% even though it was a small fraction of all kept cells. Build data
    where the true fraction is known exactly and check the drawn bar value
    against it, not just that the panel doesn't crash."""
    n = 1000
    df = pd.DataFrame(
        {
            "DAPI": np.full(n, 100.0),
            "area": np.full(n, 200.0),
            "region_num": np.zeros(n, dtype=int),
        }
    )
    # exactly 20 cells (2% of all cells) get both flags; nothing else is ever flagged
    df.loc[:19, "DAPI"] = 1.0  # below nuc_flag_thres
    df.loc[:19, "area"] = (
        1.0  # below size_flag_thres, but still above the (very low) drop thresholds
    )
    nuc_thres, size_thres = 0.5, 0.5
    nuc_flag, size_flag = 50.0, 50.0

    df_out, fig = filter_data(
        df,
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        region_column="region_num",
        qc_dashboard=True,
        nuc_flag_thres=nuc_flag,
        size_flag_thres=size_flag,
    )
    try:
        n_kept = len(df_out)
        expected_pct = 20 / n_kept * 100
        b2_ax = fig.get_axes()[2]  # A, B, B2 in that creation order
        assert b2_ax.get_title() == "Why Kept Cells Were Flagged"
        # find the bar whose height/width corresponds to the combo (labelled on the y-axis)
        ytick_labels = [t.get_text() for t in b2_ax.get_yticklabels()]
        combo_idx = next(i for i, lab in enumerate(ytick_labels) if "&" in lab)
        combo_width = b2_ax.patches[combo_idx].get_width()
        assert combo_width == pytest.approx(expected_pct, abs=0.05), (
            f"combo bar shows {combo_width:.2f}%, expected {expected_pct:.2f}% of kept cells "
            "(not % of the multi-flagged subset)"
        )
    finally:
        plt.close(fig)


def test_filter_data_dashboard_respects_active_style():
    """plot_filter_dashboard used to hardcode SPACEcStyle.publication()
    regardless of what the caller had actually set via sp.pl.set_style(...)
    -- verify the active style's font size is what actually renders."""
    from spacec.plotting._style import SPACEcStyle, set_style

    df = _make_filter_df()
    nuc_thres, size_thres, *_ = _thresholds_for(df)

    set_style(SPACEcStyle.presentation())
    try:
        _, fig = filter_data(
            df,
            nuc_thres=nuc_thres,
            size_thres=size_thres,
            region_column="region_num",
            qc_dashboard=True,
        )
        try:
            title_fs = fig.get_axes()[0].title.get_fontsize()
            assert title_fs == pytest.approx(
                SPACEcStyle.presentation().font_size_medium, abs=0.5
            )
        finally:
            plt.close(fig)
    finally:
        set_style(SPACEcStyle.publication())


def test_plot_filter_dashboard_direct_call_five_vs_six_panels():
    """plot_filter_dashboard is itself callable directly (not just via
    filter_data) -- panel count should depend only on whether flag_cols is
    given, independent of anything filter_data-specific."""
    df = _make_filter_df(n=400)
    df_flagged = df.assign(keep=df["area"] > 50, removal_reason="kept")
    df_flagged.loc[~df_flagged["keep"], "removal_reason"] = "small area"
    summary = (
        df_flagged.groupby("region_num")
        .agg(n_total=("area", "count"), n_kept=("keep", "sum"))
        .reset_index()
    )
    summary["pct_kept"] = summary["n_kept"] / summary["n_total"] * 100
    thresholds = {
        "area_low_global": 50,
        "nuc_low_global": None,
        "area_scope": "global",
        "nuc_scope": "global",
    }

    fig5 = plot_filter_dashboard(
        df_flagged,
        summary,
        thresholds,
        area_col="area",
        nuc_col="DAPI",
        region_col="region_num",
    )
    try:
        assert len(fig5.get_axes()) == 5
    finally:
        plt.close(fig5)

    df_flagged["qc_flag_low_size"] = df_flagged["area"] < 80
    df_flagged["qc_flag_any"] = df_flagged["qc_flag_low_size"]
    summary["pct_flagged"] = 10.0
    fig6 = plot_filter_dashboard(
        df_flagged,
        summary,
        thresholds,
        area_col="area",
        nuc_col="DAPI",
        region_col="region_num",
        flag_cols=["qc_flag_low_size"],
    )
    try:
        assert len(fig6.get_axes()) == 6
    finally:
        plt.close(fig6)


# ---------------------------------------------------------------------------
# cell_segmentation must raise, not return None / an empty mask, on failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tiled", [False, True])
def test_cell_segmentation_raises_on_model_error(tmp_path, monkeypatch, tiled):
    import tifffile

    import spacec.tools._segmentation as seg

    # 5 planes: 3-plane stacks get read back as RGB (H, W, 3).
    channels = ["DAPI", "CD45", "betaCatenin", "CD3", "CD20"]
    rng = np.random.default_rng(0)
    tifffile.imwrite(
        tmp_path / "img.tif",
        rng.integers(0, 255, (5, 64, 64), dtype=np.uint8),
    )
    (tmp_path / "channelnames.txt").write_text("\n".join(channels))

    def missing_model(*args, **kwargs):
        raise FileNotFoundError("Custom Cellpose model not found at: /nope")

    monkeypatch.setattr(seg, "run_cellpose", missing_model)

    tiling = {"tile_size": 32, "tiling_threshold": 32} if tiled else {}
    with pytest.raises(FileNotFoundError):
        seg.cell_segmentation(
            file_name=str(tmp_path / "img.tif"),
            channel_file=str(tmp_path / "channelnames.txt"),
            output_dir=str(tmp_path / "out"),
            seg_method="cellpose",
            model="/nope",
            custom_model=True,
            nuclei_channel="DAPI",
            membrane_channel_list=["CD45", "betaCatenin"],
            input_format="Multichannel",
            use_gpu=False,
            **tiling,
        )


def test_stitch_masks_raises_when_every_tile_failed():
    from spacec.tools._segmentation import stitch_masks

    tiles = [(0, 32, 0, 32), (0, 32, 32, 64)]
    with pytest.raises(RuntimeError, match="every tile"):
        stitch_masks(tiles, [None, None], (32, 64))
