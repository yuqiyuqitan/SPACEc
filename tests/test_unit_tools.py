"""Unit tests for utility functions in spacec.tools._general."""

import matplotlib

matplotlib.use("Agg")

import anndata
import numpy as np
import pandas as pd
import pytest

from spacec.tools._general import (
    adata_cell_percentages,
    add_missing_columns,
    annotate_clusters,
    build_cn_map,
    calculate_pvalue,
    filter_interactions,
    get_triangulation_distances,
    masks_to_outlines_scikit_image,
    ml_train,
    neighborhood_analysis,
    remove_rare_cell_types,
    suggest_annotation,
    tl_calculate_neigh_combs,
)

# ---------------------------------------------------------------------------
# calculate_pvalue
# ---------------------------------------------------------------------------


def test_calculate_pvalue_separated_distributions():
    row = pd.Series(
        {
            "expected": [1.0, 2.0, 3.0, 4.0, 5.0],
            "observed": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )
    result = calculate_pvalue(row)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    assert result < 0.05, "clearly separated distributions should yield small p-value"


def test_calculate_pvalue_empty_lists_returns_nan():
    row = pd.Series({"expected": [], "observed": []})
    result = calculate_pvalue(row)
    assert np.isnan(result), "empty lists should return NaN (ValueError caught)"


def test_calculate_pvalue_result_is_float_or_nan():
    row = pd.Series(
        {
            "expected": [1.0, 2.0, 3.0],
            "observed": [4.0, 5.0, 6.0],
        }
    )
    result = calculate_pvalue(row)
    assert isinstance(result, float) or np.isnan(result)


def test_calculate_pvalue_identical_distributions_returns_nan_or_one():
    # All-identical data may raise ValueError (NaN returned) or return p=1.0 —
    # both are acceptable outcomes.
    row = pd.Series(
        {
            "expected": [5.0, 5.0, 5.0, 5.0],
            "observed": [5.0, 5.0, 5.0, 5.0],
        }
    )
    result = calculate_pvalue(row)
    assert np.isnan(result) or result == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# masks_to_outlines_scikit_image
# ---------------------------------------------------------------------------


def test_masks_to_outlines_2d_basic():
    masks = np.zeros((8, 8), dtype=np.int32)
    masks[2:6, 2:6] = 1  # filled 4×4 square not touching image edge
    result = masks_to_outlines_scikit_image(masks)
    assert result.shape == (8, 8)
    assert result.dtype == bool
    # edge pixel of square is on the inner boundary (mode="inner")
    assert result[2, 2] is np.bool_(True), "edge pixel of mask should be boundary"
    # interior pixel away from boundary should be False
    assert result[3, 3] is np.bool_(False) or result[4, 4] is np.bool_(False)
    # background pixel is never a boundary
    assert result[0, 0] is np.bool_(False)


def test_masks_to_outlines_2d_all_zeros():
    masks = np.zeros((6, 6), dtype=np.int32)
    result = masks_to_outlines_scikit_image(masks)
    assert result.shape == (6, 6)
    assert not np.any(result), "no masks → no boundaries"


def test_masks_to_outlines_3d_basic():
    masks = np.zeros((3, 8, 8), dtype=np.int32)
    masks[1, 2:6, 2:6] = 1  # filled square only in frame 1
    result = masks_to_outlines_scikit_image(masks)
    assert result.shape == (3, 8, 8)
    assert result.dtype == bool
    assert not np.any(result[0]), "frame 0 is empty"
    assert np.any(result[1]), "frame 1 has a mask → has boundaries"
    assert not np.any(result[2]), "frame 2 is empty"


def test_masks_to_outlines_1d_raises():
    with pytest.raises(ValueError):
        masks_to_outlines_scikit_image(np.zeros(8, dtype=np.int32))


def test_masks_to_outlines_4d_raises():
    with pytest.raises(ValueError):
        masks_to_outlines_scikit_image(np.zeros((2, 4, 4, 4), dtype=np.int32))


def test_masks_to_outlines_error_message_contains_ndim():
    with pytest.raises(ValueError, match="4D"):
        masks_to_outlines_scikit_image(np.zeros((1, 2, 3, 4), dtype=np.int32))


# ---------------------------------------------------------------------------
# adata_cell_percentages
# ---------------------------------------------------------------------------


def test_adata_cell_percentages_basic():
    adata = anndata.AnnData(X=np.zeros((6, 2)))
    adata.obs["cell_type"] = ["CD4", "CD4", "CD8", "CD8", "NK", "CD4"]
    result = adata_cell_percentages(adata, column_percentage="cell_type")
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) >= {"cell_type", "percentage"}
    assert len(result) == 3  # three unique cell types
    cd4_row = result.loc[result["cell_type"] == "CD4", "percentage"]
    assert cd4_row.values[0] == pytest.approx(50.0)
    assert result["percentage"].sum() == pytest.approx(100.0)


def test_adata_cell_percentages_custom_column():
    adata = anndata.AnnData(X=np.zeros((4, 2)))
    adata.obs["neighborhood"] = ["N1", "N1", "N2", "N2"]
    result = adata_cell_percentages(adata, column_percentage="neighborhood")
    assert set(result.columns) >= {"neighborhood", "percentage"}
    assert len(result) == 2
    assert result["percentage"].sum() == pytest.approx(100.0)


def test_adata_cell_percentages_single_type():
    adata = anndata.AnnData(X=np.zeros((5, 2)))
    adata.obs["cell_type"] = ["CD4"] * 5
    result = adata_cell_percentages(adata, column_percentage="cell_type")
    assert len(result) == 1
    assert result["percentage"].values[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# add_missing_columns
# ---------------------------------------------------------------------------


def test_add_missing_columns_basic():
    tri_dist = pd.DataFrame(
        {
            "unique_region": ["R1", "R2", "R1"],
            "distance": [1.0, 2.0, 3.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "unique_region": ["R1", "R2"],
            "condition": ["treated", "control"],
        }
    )
    result = add_missing_columns(tri_dist, metadata, shared_column="unique_region")
    assert "condition" in result.columns
    r1_cond = result.loc[result["unique_region"] == "R1", "condition"].iloc[0]
    r2_cond = result.loc[result["unique_region"] == "R2", "condition"].iloc[0]
    assert r1_cond == "treated"
    assert r2_cond == "control"
    assert "distance" in result.columns


def test_add_missing_columns_missing_region_fills_unknown():
    tri_dist = pd.DataFrame(
        {
            "unique_region": ["R1", "R99"],  # R99 not in metadata
            "distance": [1.0, 2.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "unique_region": ["R1"],
            "condition": ["treated"],
        }
    )
    result = add_missing_columns(tri_dist, metadata)
    r99_cond = result.loc[result["unique_region"] == "R99", "condition"].iloc[0]
    assert r99_cond == "Unknown"


def test_add_missing_columns_already_present_not_overwritten():
    tri_dist = pd.DataFrame(
        {
            "unique_region": ["R1"],
            "condition": ["original_value"],
        }
    )
    metadata = pd.DataFrame(
        {
            "unique_region": ["R1"],
            "condition": ["would_overwrite"],
        }
    )
    result = add_missing_columns(tri_dist, metadata)
    # "condition" is already in tri_dist so it's not in missing_columns
    assert result["condition"].iloc[0] == "original_value"


def test_add_missing_columns_custom_shared_column():
    tri_dist = pd.DataFrame({"section": ["S1", "S2"], "distance": [1.0, 2.0]})
    metadata = pd.DataFrame({"section": ["S1", "S2"], "condition": ["a", "b"]})
    result = add_missing_columns(tri_dist, metadata, shared_column="section")
    assert result["condition"].tolist() == ["a", "b"]


def test_add_missing_columns_returns_dataframe():
    tri_dist = pd.DataFrame({"unique_region": ["R1"], "x": [1.0]})
    metadata = pd.DataFrame({"unique_region": ["R1"], "meta": ["m"]})
    result = add_missing_columns(tri_dist, metadata)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# remove_rare_cell_types
# ---------------------------------------------------------------------------


def test_remove_rare_cell_types_removes_rare():
    adata = anndata.AnnData(X=np.zeros((100, 2)))
    # 90 CD4 (90%), 9 CD8 (9%), 1 NK (1%)
    adata.obs["cell_type"] = ["CD4"] * 90 + ["CD8"] * 9 + ["NK"] * 1
    distance_pvals = pd.DataFrame(
        {
            "celltype1": ["CD4", "CD4", "NK"],
            "celltype2": ["CD8", "NK", "CD8"],
            "pvalue": [0.01, 0.04, 0.02],
        }
    )
    result = remove_rare_cell_types(
        adata,
        distance_pvals,
        cell_type_column="cell_type",
        min_cell_type_percentage=2,
    )
    assert len(result) == 1, "only the CD4-CD8 row should survive"
    assert "NK" not in result["celltype1"].values
    assert "NK" not in result["celltype2"].values


def test_remove_rare_cell_types_no_rare():
    adata = anndata.AnnData(X=np.zeros((10, 2)))
    adata.obs["cell_type"] = ["CD4"] * 5 + ["CD8"] * 5  # 50% each
    distance_pvals = pd.DataFrame(
        {
            "celltype1": ["CD4"],
            "celltype2": ["CD8"],
            "pvalue": [0.01],
        }
    )
    result = remove_rare_cell_types(adata, distance_pvals, min_cell_type_percentage=1)
    assert len(result) == 1


def test_remove_rare_cell_types_returns_dataframe():
    adata = anndata.AnnData(X=np.zeros((4, 2)))
    adata.obs["cell_type"] = ["CD4"] * 4
    distance_pvals = pd.DataFrame(
        {
            "celltype1": ["CD4"],
            "celltype2": ["CD4"],
            "pvalue": [0.5],
        }
    )
    result = remove_rare_cell_types(adata, distance_pvals)
    assert isinstance(result, pd.DataFrame)


def test_remove_rare_cell_types_filters_both_columns():
    adata = anndata.AnnData(X=np.zeros((100, 2)))
    adata.obs["cell_type"] = ["CD4"] * 99 + ["NK"] * 1
    distance_pvals = pd.DataFrame(
        {
            "celltype1": ["CD4"],  # CD4 is common
            "celltype2": ["NK"],  # NK is rare (1%)
            "pvalue": [0.01],
        }
    )
    result = remove_rare_cell_types(
        adata,
        distance_pvals,
        cell_type_column="cell_type",
        min_cell_type_percentage=2,
    )
    assert len(result) == 0, "row with rare NK in celltype2 should be removed"


# ---------------------------------------------------------------------------
# annotate_clusters
# ---------------------------------------------------------------------------


def test_annotate_clusters_int_mapping_keys():
    # Regression test: adata.obs[cluster_key] is matched as str internally, so a
    # mapping keyed by the column's original int dtype used to match nothing at all
    # and silently leave every cell unmapped.
    adata = anndata.AnnData(X=np.zeros((6, 2)))
    adata.obs["cl"] = [0, 0, 1, 1, 2, 2]
    adata = annotate_clusters(
        adata, cluster_key="cl", mapping={0: "A", 1: "B", 2: "C"}, key_added="ct"
    )
    assert adata.obs["ct"].tolist() == ["A", "A", "B", "B", "C", "C"]


def test_annotate_clusters_str_mapping_keys_still_work():
    adata = anndata.AnnData(X=np.zeros((4, 2)))
    adata.obs["cl"] = [0, 0, 1, 1]
    adata = annotate_clusters(
        adata, cluster_key="cl", mapping={"0": "A", "1": "B"}, key_added="ct"
    )
    assert adata.obs["ct"].tolist() == ["A", "A", "B", "B"]


# ---------------------------------------------------------------------------
# ml_train
# ---------------------------------------------------------------------------


def test_ml_train_stratified_split_keeps_rare_class():
    # Regression test: an unstratified split could drop a rare class from the test
    # fold entirely, then the internal classification_report(target_names=...) call
    # raised ValueError: "Number of classes, N, does not match size of target_names, M".
    rng = np.random.default_rng(0)
    n_common = 100
    X = rng.normal(size=(2 * n_common + 3, 5))
    X[:n_common] += 5
    X[n_common : 2 * n_common] -= 5
    labels = ["A"] * n_common + ["B"] * n_common + ["rare"] * 3
    adata = anndata.AnnData(X=X)
    adata.obs["cell_type"] = labels
    # Should not raise, and should complete training with all 3 classes represented.
    svc = ml_train(adata_train=adata, label="cell_type", mode="knn", showfig=False)
    assert set(svc.classes_) == {"A", "B", "rare"}


def test_ml_train_singleton_class_does_not_crash():
    # Even a class that appears only once (impossible to stratify or split into both
    # folds) should not crash classification_report -- it should just report zero
    # support for that class rather than raising.
    rng = np.random.default_rng(0)
    n_common = 100
    X = rng.normal(size=(2 * n_common + 1, 5))
    X[:n_common] += 5
    X[n_common : 2 * n_common] -= 5
    labels = ["A"] * n_common + ["B"] * n_common + ["singleton"] * 1
    adata = anndata.AnnData(X=X)
    adata.obs["cell_type"] = labels
    svc = ml_train(adata_train=adata, label="cell_type", mode="knn", showfig=False)
    assert set(svc.classes_) == {"A", "B", "singleton"}


# ---------------------------------------------------------------------------
# filter_interactions
# ---------------------------------------------------------------------------


def test_filter_interactions_raises_on_more_than_two_groups():
    # Regression test: >2 unique values in `comparison` used to silently collapse
    # dist_table toward empty (the pivot requires significance in every column at
    # once) with no indication why -- should now raise a clear ValueError instead.
    distance_pvals = pd.DataFrame(
        {
            "celltype1": ["X", "X", "X"],
            "celltype2": ["Y", "Y", "Y"],
            "condition": ["A", "B", "C"],
            "pvalue": [0.01, 0.01, 0.01],
            "observed_mean": [1.0, 1.0, 1.0],
            "logfold_group": [0.5, 0.5, 0.5],
            "interaction": ["X --> Y"] * 3,
        }
    )
    with pytest.raises(ValueError, match="exactly 2 unique values"):
        filter_interactions(distance_pvals, comparison="condition")


def test_filter_interactions_binary_comparison_still_works():
    distance_pvals = pd.DataFrame(
        {
            "celltype1": ["X", "X"],
            "celltype2": ["Y", "Y"],
            "condition": ["A", "B"],
            "pvalue": [0.01, 0.01],
            "observed_mean": [1.0, 1.0],
            "logfold_group": [0.5, -0.5],
            "interaction": ["X --> Y"] * 2,
        }
    )
    dist_table, sub = filter_interactions(
        distance_pvals, comparison="condition", logfold_group_abs=0.1
    )
    assert dist_table.shape == (1, 2)
    assert isinstance(sub, pd.DataFrame)


# ---------------------------------------------------------------------------
# suggest_annotation
# ---------------------------------------------------------------------------

_MARKERS = {"T cell": ["CD3", "CD4"], "B cell": ["CD19", "CD20"]}


def _make_marker_adata():
    """4 clusters: T-high, B-high, flat, and both-high."""
    rows = {
        "0": [5.0, 5.0, 0.0, 0.0],  # T cell
        "1": [0.0, 0.0, 5.0, 5.0],  # B cell
        "2": [1.0, 1.0, 1.0, 1.0],  # nothing stands out -> Unknown
        "3": [5.0, 5.0, 5.0, 5.0],  # both panels high -> Mixed
    }
    X = np.repeat(np.array(list(rows.values())), 3, axis=0)
    adata = anndata.AnnData(X=X)
    adata.var_names = ["CD3", "CD4", "CD19", "CD20"]
    adata.obs["leiden"] = np.repeat(list(rows.keys()), 3)
    return adata


def test_suggest_annotation_assigns_marker_high_clusters():
    mapping = suggest_annotation(_make_marker_adata(), "leiden", _MARKERS)
    assert mapping["0"] == "T cell"
    assert mapping["1"] == "B cell"


def test_suggest_annotation_flags_unknown_and_mixed():
    mapping = suggest_annotation(_make_marker_adata(), "leiden", _MARKERS)
    assert mapping["2"] == "Unknown"
    assert mapping["3"] == "Mixed"


def test_suggest_annotation_drops_absent_markers_with_warning(capsys):
    markers = {"T cell": ["CD3", "CD4", "NOPE"], "B cell": ["CD19", "CD20"]}
    mapping = suggest_annotation(_make_marker_adata(), "leiden", markers)
    assert mapping["0"] == "T cell"
    assert "NOPE" in capsys.readouterr().out


def test_suggest_annotation_skips_cell_type_without_markers(capsys):
    markers = dict(_MARKERS, Ghost=["NOPE1", "NOPE2"])
    mapping = suggest_annotation(_make_marker_adata(), "leiden", markers)
    assert "Ghost" not in mapping.values()
    assert "Ghost" in capsys.readouterr().out


def test_suggest_annotation_return_scores_shape():
    mapping, scores = suggest_annotation(
        _make_marker_adata(), "leiden", _MARKERS, return_scores=True
    )
    assert scores.shape == (4, 2)
    assert set(scores.columns) == set(_MARKERS)
    assert set(scores.index) == set(mapping)


def test_suggest_annotation_constant_marker_does_not_dominate():
    # A marker with zero variance across clusters must contribute 0, not a
    # huge z-score from dividing by a near-zero std.
    adata = _make_marker_adata()
    adata.X[:, 0] = 3.0  # CD3 constant everywhere
    mapping, scores = suggest_annotation(adata, "leiden", _MARKERS, return_scores=True)
    assert np.isfinite(scores.values).all()
    assert mapping["1"] == "B cell"


def test_suggest_annotation_rejects_single_cluster():
    adata = _make_marker_adata()
    adata.obs["one"] = "0"
    with pytest.raises(ValueError, match="at least 2"):
        suggest_annotation(adata, "one", _MARKERS)


def test_suggest_annotation_rejects_bad_inputs():
    adata = _make_marker_adata()
    with pytest.raises(ValueError, match="not found in adata.obs"):
        suggest_annotation(adata, "missing", _MARKERS)
    with pytest.raises(ValueError, match="Invalid mode"):
        suggest_annotation(adata, "leiden", _MARKERS, mode="nope")
    with pytest.raises(ValueError, match="None of the markers"):
        suggest_annotation(adata, "leiden", {"X": ["NOPE"]})


def test_suggest_annotation_feeds_annotate_clusters():
    adata = _make_marker_adata()
    mapping = suggest_annotation(adata, "leiden", _MARKERS)
    annotate_clusters(adata, "leiden", mapping)
    assert adata.obs["cell_type"].tolist()[:3] == ["T cell"] * 3
    assert "clustering_tree" in adata.uns
    assert "clustering_report" in adata.uns


def test_suggest_annotation_manual_shows_progress_and_legend(monkeypatch, capsys):
    import matplotlib.pyplot as plt

    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    figs = []
    monkeypatch.setattr(plt, "show", lambda: figs.append(plt.gcf()))

    suggest_annotation(_make_marker_adata(), "leiden", _MARKERS, mode="manual")

    out = capsys.readouterr().out
    assert "[1/4]" in out and "[4/4]" in out
    assert len(figs) == 4
    # The legend is the only way to read the color->cell-type mapping back off the
    # chart (the printed "Proposed: ..." line only names the winning candidate).
    assert all(fig.axes[0].get_legend() is not None for fig in figs)


def test_suggest_annotation_manual_patch_option_absent_without_masks_images(
    monkeypatch,
):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", fake_input)
    suggest_annotation(_make_marker_adata(), "leiden", _MARKERS, mode="manual")
    assert all("p = show patches" not in p for p in prompts)


def test_suggest_annotation_manual_patch_option_calls_patch_qc_and_reprompts_same_cluster(
    monkeypatch,
):
    import matplotlib.pyplot as plt

    import spacec.tools._general as tools_general

    monkeypatch.setattr(plt, "show", lambda: None)

    calls = []

    def fake_select_regions(adata, groupby, cluster, region_col, **kwargs):
        return ["regA"]

    def fake_patch_qc(adata, groupby, cluster, markers, masks, images, **kwargs):
        calls.append(
            {"cluster": cluster, "markers": markers, "region": kwargs.get("region")}
        )
        return plt.figure()

    monkeypatch.setattr(
        tools_general, "_select_regions_for_cluster", fake_select_regions
    )
    monkeypatch.setattr(tools_general, "patch_QC", fake_patch_qc)

    # cluster '0': peek at patches, then accept -- the other 3 clusters just accept outright.
    responses = iter(["p", "y", "y", "y", "y"])
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr("builtins.input", fake_input)

    mapping = suggest_annotation(
        _make_marker_adata(),
        "leiden",
        _MARKERS,
        mode="manual",
        masks={"regA": None},
        images={"regA": None},
    )

    assert len(calls) == 1
    assert calls[0]["cluster"] == "0"
    assert calls[0]["region"] == "regA"
    assert set(calls[0]["markers"]) <= {"CD3", "CD4", "CD19", "CD20"}
    assert "show patches (p/patch)" in prompts[0]
    # 'p' re-prompts the SAME cluster ('0') instead of advancing -- input() called twice
    # before cluster '0' is actually resolved.
    assert len(prompts) >= 2
    assert mapping["0"] == "T cell"


def test_suggest_annotation_manual_quit_returns_partial_mapping(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    # cluster '0' accepted, then 'quit' on cluster '1' -- '2'/'3' must never be reached.
    # "quit" (long form) and "q" (short form) must both work -- only "quit" checked here,
    # "q" is exercised by test_suggest_annotation_manual_quit_prompt_and_legend_mention_it.
    responses = iter(["y", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    mapping = suggest_annotation(
        _make_marker_adata(), "leiden", _MARKERS, mode="manual"
    )

    assert mapping == {"0": "T cell"}


def test_suggest_annotation_manual_quit_prompt_and_legend_mention_it(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return "q"

    monkeypatch.setattr("builtins.input", fake_input)
    suggest_annotation(_make_marker_adata(), "leiden", _MARKERS, mode="manual")
    assert "q/quit" in prompts[0]


def test_suggest_annotation_resume_dict_skips_already_decided_clusters():
    adata = _make_marker_adata()
    resume = {"0": "T cell", "1": "B cell"}
    mapping = suggest_annotation(adata, "leiden", _MARKERS, resume_dict=resume)
    # resumed entries carried over unchanged; only '2'/'3' actually (re)scored this call.
    assert mapping["0"] == "T cell"
    assert mapping["1"] == "B cell"
    assert mapping["2"] == "Unknown"
    assert mapping["3"] == "Mixed"


def test_suggest_annotation_resume_dict_scores_still_complete():
    # return_scores doesn't depend on resume_dict at all -- it's the same upfront
    # per-cluster z-score computation every call makes, resumed or not.
    adata = _make_marker_adata()
    mapping, scores = suggest_annotation(
        adata,
        "leiden",
        _MARKERS,
        resume_dict={"0": "T cell"},
        return_scores=True,
    )
    assert scores.shape == (4, 2)
    assert set(scores.index) == {"0", "1", "2", "3"}


def test_suggest_annotation_resume_dict_rejects_foreign_labels():
    adata = _make_marker_adata()
    with pytest.raises(ValueError, match="not in adata.obs"):
        suggest_annotation(
            adata, "leiden", _MARKERS, resume_dict={"not-a-real-cluster": "X"}
        )


def test_suggest_annotation_manual_resume_progress_counts_from_overall_total(
    monkeypatch,
):
    # 4 clusters total, 1 already resumed ('0') -> the first cluster reviewed this call
    # (whichever of '1'/'2'/'3' sorts first) must show as 2/4, not 1/3.
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    headers = []
    import spacec.tools._general as tools_general

    real_print = print

    def capturing_print(*args, **kwargs):
        headers.append(" ".join(str(a) for a in args))
        real_print(*args, **kwargs)

    monkeypatch.setattr(tools_general, "print", capturing_print, raising=False)

    suggest_annotation(
        _make_marker_adata(),
        "leiden",
        _MARKERS,
        mode="manual",
        resume_dict={"0": "T cell"},
    )
    first_cluster_header = next(h for h in headers if h.startswith("CLUSTER"))
    assert "[2/4]" in first_cluster_header


def test_suggest_annotation_manual_accepts_long_form_input(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    # long forms for accept/unknown, matching accept short-form's behavior for the rest.
    responses = iter(["yes", "unknown", "y", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    mapping = suggest_annotation(
        _make_marker_adata(), "leiden", _MARKERS, mode="manual"
    )
    assert mapping["0"] == "T cell"  # 'yes' == accept proposal
    assert (
        mapping["1"] == "Unknown"
    )  # 'unknown' overrides to Unknown regardless of proposal


def test_suggest_annotation_manual_legend_lists_all_candidate_cell_types(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    figs = []
    monkeypatch.setattr(plt, "show", lambda: figs.append(plt.gcf()))

    suggest_annotation(_make_marker_adata(), "leiden", _MARKERS, mode="manual")

    legend_labels = {t.get_text() for t in figs[0].axes[0].get_legend().get_texts()}
    assert legend_labels == set(_MARKERS)


def test_suggest_annotation_manual_patch_option_forwards_style(monkeypatch):
    import matplotlib.pyplot as plt

    import spacec.tools._general as tools_general
    from spacec.plotting._style import SPACEcStyle

    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(
        tools_general,
        "_select_regions_for_cluster",
        lambda adata, groupby, cluster, region_col, **kw: ["regA"],
    )
    captured = {}

    def fake_patch_qc(adata, groupby, cluster, markers, masks, images, **kwargs):
        captured["style"] = kwargs.get("style")
        return plt.figure()

    monkeypatch.setattr(tools_general, "patch_QC", fake_patch_qc)

    # cluster '0': peek at patches, then accept -- clusters '1'/'2'/'3' just accept.
    responses = iter(["p", "y", "y", "y", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    custom_style = SPACEcStyle.presentation()
    suggest_annotation(
        _make_marker_adata(),
        "leiden",
        _MARKERS,
        mode="manual",
        masks={"regA": None},
        images={"regA": None},
        style=custom_style,
    )
    assert captured["style"] is custom_style


def test_suggest_annotation_manual_despines_and_has_legend(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    figs = []
    monkeypatch.setattr(plt, "show", lambda: figs.append(plt.gcf()))

    suggest_annotation(_make_marker_adata(), "leiden", _MARKERS, mode="manual")

    assert len(figs) == 4
    for fig in figs:
        ax = fig.axes[0]
        assert ax.get_legend() is not None
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()


def test_suggest_annotation_manual_legends_false_hides_legend_and_options(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    figs = []
    monkeypatch.setattr(plt, "show", lambda: figs.append(plt.gcf()))

    suggest_annotation(
        _make_marker_adata(), "leiden", _MARKERS, mode="manual", legends=False
    )

    assert len(figs) == 4
    for fig in figs:
        ax = fig.axes[0]
        assert ax.get_legend() is None
        assert not any(t.get_text().startswith("Options:") for t in ax.texts)


# ---------------------------------------------------------------------------
# build_cn_map
# ---------------------------------------------------------------------------


def _make_cn_adata(n=400):
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "x": rng.uniform(0, 100, n),
            "y": rng.uniform(0, 100, n),
            "region": np.repeat(["r1", "r2"], n // 2),
            "cond": np.repeat(["a", "b"], n // 2),
            "CN": rng.choice(["A", "B", "C"], n),
        },
        index=[str(i) for i in range(n)],
    )
    return anndata.AnnData(np.zeros((n, 1)), obs=obs)


def test_build_cn_map_sub_col():
    # Regression test: sub_col was dropped before windowing -> KeyError.
    res = build_cn_map(
        _make_cn_adata(), "CN", "region", k=10, sub_list=["a"], sub_col="cond"
    )
    assert (res["w"]["cond"] == "a").all()


def test_build_cn_map_palette_with_absent_cn():
    # Regression test: palette keys missing from the data raised a KeyError.
    palette = {"A": "red", "B": "green", "C": "blue", "D": "black"}
    res = build_cn_map(_make_cn_adata(), "CN", "region", palette=palette, k=10)
    assert res["l"] == list(palette)
    assert (res["w"]["D"] == 0).all()


def test_calculate_neigh_combs_low_per_keep_thres():
    # Regression test: per_keep_thres below the top combination's share made
    # test_len 0 and the reported share negative (simp_sums[0] - simp_sums[-1]).
    import matplotlib.pyplot as plt

    w = pd.DataFrame({"A": [10] * 6 + [5, 0], "B": [0] * 6 + [5, 10], "C": 0})
    plt.close("all")
    simps, simp_freqs, simp_sums = tl_calculate_neigh_combs(
        w, ["A", "B", "C"], 10, per_keep_thres=0.1
    )
    assert len(plt.get_fignums()) == 1
    assert "per_keep_thres" not in plt.gca().get_title()
    assert "0 combinations" in plt.gcf().axes[0].get_title()
    assert len(simps) == len(w) and simp_sums.iloc[-1] == pytest.approx(1)
    plt.close("all")


def test_neighborhood_analysis_forwards_kmeans_kwargs(monkeypatch):
    import spacec.tools._general as tg

    calls = []
    real = tg.MiniBatchKMeans

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(tg, "MiniBatchKMeans", spy)

    adata = neighborhood_analysis(
        _make_cn_adata(), "region", "CN", k=10, n_neighborhoods=3
    )
    assert calls[-1] == {"n_clusters": 3, "random_state": 0}  # defaults unchanged
    assert adata.obs["CN_k10_n3"].nunique() == 3

    kw = {"batch_size": 100, "n_init": 3, "random_state": 7}
    neighborhood_analysis(
        _make_cn_adata(), "region", "CN", k=10, n_neighborhoods=3, kmeans_kwargs=kw
    )
    assert calls[-1] == {"n_clusters": 3, **kw}


# ---------------------------------------------------------------------------
# get_triangulation_distances
# ---------------------------------------------------------------------------


def test_get_triangulation_distances_keeps_float_coordinates():
    # Regression test: float coordinates were silently truncated to int.
    df = pd.DataFrame(
        {
            "id": ["c0", "c1", "c2"],
            "x": [0.0, 0.6, 1.5],
            "y": [0.0, 0.8, 0.2],
            "ct": ["T", "B", "T"],
            "region": ["r1"] * 3,
        }
    )
    out = get_triangulation_distances(df, "id", "x", "y", "ct", "region", num_cores=1)
    edge = out[(out["celltype1_index"] == "c0") & (out["celltype2_index"] == "c1")]
    assert edge["distance"].iloc[0] == pytest.approx(1.0)
    assert 0.6 in out["celltype1_X"].values


# ---------------------------------------------------------------------------
# check_download_tm_plugins
# ---------------------------------------------------------------------------


def test_check_download_tm_plugins_without_conda(tmp_path, monkeypatch):
    import spacec.tools._general as tools_general

    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(tools_general.tj, "__file__", str(tmp_path / "jupyter.py"))
    saved = []

    def fake_download(url, save_path):
        saved.append(save_path)
        open(save_path, "w").close()

    monkeypatch.setattr(tools_general, "download_file_tm", fake_download)

    tools_general.check_download_tm_plugins()
    assert len(saved) == 5
    assert all(p.startswith(str(tmp_path / "plugins")) for p in saved)

    saved.clear()
    tools_general.check_download_tm_plugins()  # already present: no downloads
    assert saved == []


def test_import_leaves_logging_alone():
    # `import spacec` must not disable logging process-wide or keep the root
    # logger at INFO (TissUUmaps sets it on import). Fresh interpreter, since
    # spacec is already imported in this one.
    import subprocess
    import sys

    code = (
        "import logging; logging.getLogger().setLevel(logging.WARNING); "
        "import spacec; "
        "assert logging.root.manager.disable == 0, 'logging disabled'; "
        "assert logging.getLogger().level == logging.WARNING, logging.getLogger().level"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
