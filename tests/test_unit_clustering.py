"""Unit tests for spacec.tools._general.clustering(), focused on the
key_filter/subset_cluster reintegration path and the compute_umap /
neighbors-skipping behavior.

These guard against a bug where the key_filter path merged on every obs
column instead of the shared index, which silently replaced adata.obs_names
with stringified row numbers (see the pd.merge -> .loc rewrite in
clustering()).
"""

from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from spacec.tools._general import annotate_clusters, clustering, subclustering


def _make_adata(n_cells=40, n_markers=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_cells, n_markers)).astype(np.float32)
    obs_names = [f"cell_{i}" for i in range(n_cells)]
    var_names = [f"marker_{i}" for i in range(n_markers)]
    obs = pd.DataFrame(index=obs_names)
    # coarse cluster assignment used as key_filter
    obs["coarse"] = pd.Categorical(["A"] * (n_cells // 2) + ["B"] * (n_cells // 2))
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    return adata


# ---------------------------------------------------------------------------
# key_filter / subset_cluster preserves obs_names and reintegrates labels
# ---------------------------------------------------------------------------


def test_clustering_key_filter_preserves_obs_names():
    adata = _make_adata()
    original_obs_names = adata.obs_names.tolist()

    result = clustering(
        adata,
        clustering="leiden",
        n_neighbors=5,
        resolution=1,
        reclustering=False,
        compute_umap=False,
        key_filter="coarse",
        subset_cluster=["A"],
        seed=0,
    )

    # obs_names must be unchanged -- previously they were silently replaced
    # with stringified row numbers ("0", "1", ...) by the pd.merge reindex.
    assert result.obs_names.tolist() == original_obs_names


def test_clustering_key_filter_reintegrates_labels():
    adata = _make_adata()
    key_added = "leiden_1"

    result = clustering(
        adata,
        clustering="leiden",
        n_neighbors=5,
        resolution=1,
        reclustering=False,
        compute_umap=False,
        key_filter="coarse",
        subset_cluster=["A"],
        key_added=key_added,
        seed=0,
    )

    is_a = result.obs["coarse"] == "A"
    # subset ("A") rows get real leiden cluster labels, distinct from "B"
    assert set(result.obs.loc[is_a, key_added]) <= set(
        result.obs.loc[is_a, key_added].unique()
    )
    assert result.obs.loc[is_a, key_added].notna().all()
    # rows outside subset_cluster fall back to their original key_filter value
    assert (result.obs.loc[~is_a, key_added] == "B").all()


# ---------------------------------------------------------------------------
# compute_umap / neighbors skipping
# ---------------------------------------------------------------------------


def test_clustering_compute_umap_false_skips_umap():
    adata = _make_adata()
    result = clustering(
        adata,
        clustering="leiden",
        n_neighbors=5,
        resolution=1,
        reclustering=False,
        compute_umap=False,
        seed=0,
    )
    assert "X_umap" not in result.obsm
    assert "leiden_1" in result.obs


def test_clustering_flowsom_compute_umap_false_skips_neighbors():
    # FlowSOM's SOM training is stubbed out here so the test only exercises
    # spacec's own neighbors/UMAP-skipping logic, not pyFlowSOM's internals
    # (which impose their own constraints on synthetic input size/variance).
    adata = _make_adata()
    n_cells = adata.n_obs
    fake_clusters = np.zeros(n_cells)

    with patch("spacec.tools._general.som", return_value=None) as mock_som, patch(
        "spacec.tools._general.map_data_to_nodes",
        return_value=(fake_clusters, fake_clusters),
    ), patch("spacec.tools._general.sc.pp.neighbors") as mock_neighbors, patch(
        "spacec.tools._general.sc.tl.umap"
    ) as mock_umap:
        result = clustering(
            adata,
            clustering="flowSOM",
            n_neighbors=5,
            reclustering=False,
            compute_umap=False,
            seed=0,
        )

    mock_som.assert_called_once()
    mock_neighbors.assert_not_called()
    mock_umap.assert_not_called()
    assert "flowSOM_1" in result.obs


def test_clustering_flowsom_compute_umap_true_computes_neighbors():
    adata = _make_adata()
    n_cells = adata.n_obs
    fake_clusters = np.zeros(n_cells)

    with patch("spacec.tools._general.som", return_value=None), patch(
        "spacec.tools._general.map_data_to_nodes",
        return_value=(fake_clusters, fake_clusters),
    ), patch("spacec.tools._general.sc.pp.neighbors") as mock_neighbors, patch(
        "spacec.tools._general.sc.tl.umap"
    ) as mock_umap:
        clustering(
            adata,
            clustering="flowSOM",
            n_neighbors=5,
            reclustering=False,
            compute_umap=True,
            seed=0,
        )

    # UMAP needs the neighbor graph, so requesting it pulls neighbors in
    # even for FlowSOM, which doesn't use it for clustering itself.
    mock_neighbors.assert_called_once()
    mock_umap.assert_called_once()


# ---------------------------------------------------------------------------
# subclustering()
# ---------------------------------------------------------------------------


def _blob(rng, center, n, dim=6, scale=0.3):
    return rng.normal(loc=center, scale=scale, size=(n, dim)).astype(np.float32)


def _make_subcluster_adata(n_neighbors=5, seed=0):
    """Three well-separated coarse groups ('0', '1', '2'), where '0' and '1'
    each contain enough internal sub-structure (two widely separated
    sub-blobs) that leiden reliably splits them further under restrict_to
    -- restrict_to only clusters the induced subgraph of the restricted
    category, so cross-group separation doesn't matter, only within-group
    structure does. '2' has no sub-structure and is used to check that
    splitting '0'/'1' leaves it untouched.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            _blob(rng, 0.0, 15),  # '0' sub-blob a
            _blob(rng, 20.0, 15),  # '0' sub-blob b
            _blob(rng, -20.0, 15),  # '1' sub-blob a
            _blob(rng, -40.0, 15),  # '1' sub-blob b
            _blob(rng, 60.0, 30),  # '2' (single blob)
        ]
    )
    n_cells = X.shape[0]
    obs_names = [f"cell_{i}" for i in range(n_cells)]
    obs = pd.DataFrame(index=obs_names)
    obs["coarse"] = pd.Categorical(["0"] * 30 + ["1"] * 30 + ["2"] * 30)
    var = pd.DataFrame(index=[f"marker_{i}" for i in range(X.shape[1])])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    return adata


def test_subclustering_splits_only_named_cluster():
    adata = _make_subcluster_adata()
    result = subclustering(
        adata, "coarse", ["0"], resolution=1.0, seed=0, n_neighbors=5
    )

    key_added = "coarse_subcluster"
    assert key_added in result.obs
    is_zero = result.obs["coarse"] == "0"
    # cluster '0' cells got new "0,i"-style labels
    assert result.obs.loc[is_zero, key_added].astype(str).str.match(r"^0,\d+$").all()
    # untouched groups keep their original label exactly
    assert (result.obs.loc[result.obs["coarse"] == "1", key_added] == "1").all()
    assert (result.obs.loc[result.obs["coarse"] == "2", key_added] == "2").all()
    # cluster '0' actually split into more than one subcluster
    assert result.obs.loc[is_zero, key_added].astype(str).nunique() > 1


def test_subclustering_preserves_obs_names():
    adata = _make_subcluster_adata()
    original_obs_names = adata.obs_names.tolist()
    result = subclustering(
        adata, "coarse", ["0"], resolution=1.0, seed=0, n_neighbors=5
    )
    assert result.obs_names.tolist() == original_obs_names


def test_subclustering_sequential_chaining_across_two_clusters():
    """Splitting ['0', '1'] in one call must split each cluster independently
    (sequential restrict_to chaining), not pool them into one joint split."""
    adata = _make_subcluster_adata()
    result = subclustering(
        adata, "coarse", ["0", "1"], resolution=1.0, seed=0, n_neighbors=5
    )
    key_added = "coarse_subcluster"
    labels = result.obs[key_added].astype(str)

    zero_labels = labels[result.obs["coarse"] == "0"]
    one_labels = labels[result.obs["coarse"] == "1"]
    two_labels = labels[result.obs["coarse"] == "2"]

    assert zero_labels.str.match(r"^0,\d+$").all()
    assert zero_labels.nunique() > 1
    assert one_labels.str.match(r"^1,\d+$").all()
    assert one_labels.nunique() > 1
    # cluster '2' was never requested and must be carried forward unchanged
    assert (two_labels == "2").all()


def test_subclustering_bad_cluster_key_raises():
    adata = _make_subcluster_adata()
    with pytest.raises(ValueError, match="not found in adata.obs"):
        subclustering(adata, "does_not_exist", ["0"])


def test_subclustering_unknown_cluster_raises():
    adata = _make_subcluster_adata()
    with pytest.raises(ValueError, match="99"):
        subclustering(adata, "coarse", ["99"])


def test_subclustering_flowsom_raises():
    adata = _make_subcluster_adata()
    with pytest.raises(ValueError, match="flowSOM"):
        subclustering(adata, "coarse", ["0"], method="flowSOM")


def test_subclustering_recompute_graph_true():
    adata = _make_subcluster_adata()
    original_obs_names = adata.obs_names.tolist()
    result = subclustering(
        adata,
        "coarse",
        ["0"],
        resolution=1.0,
        seed=0,
        recompute_graph=True,
        n_neighbors=5,
    )
    key_added = "coarse_subcluster"
    is_zero = result.obs["coarse"] == "0"
    assert result.obs.loc[is_zero, key_added].astype(str).str.match(r"^0,\d+$").all()
    assert result.obs.loc[is_zero, key_added].astype(str).nunique() > 1
    assert (result.obs.loc[result.obs["coarse"] == "1", key_added] == "1").all()
    assert result.obs_names.tolist() == original_obs_names


def test_subclustering_record_report_false_writes_no_uns_keys():
    adata = _make_subcluster_adata()
    subclustering(
        adata,
        "coarse",
        ["0"],
        resolution=1.0,
        seed=0,
        n_neighbors=5,
        _record_report=False,
    )
    assert "clustering_report" not in adata.uns
    assert "clustering_tree" not in adata.uns


def test_subclustering_records_report_and_tree():
    adata = _make_subcluster_adata()
    subclustering(adata, "coarse", ["0"], resolution=1.0, seed=0, n_neighbors=5)
    assert "clustering_report" in adata.uns
    assert "clustering_tree" in adata.uns


# ---------------------------------------------------------------------------
# annotate_clusters()
# ---------------------------------------------------------------------------


def _make_annotation_adata():
    adata = _make_subcluster_adata()
    return adata


def test_annotate_clusters_missing_keep_preserves_unmapped_label():
    adata = _make_annotation_adata()
    result = annotate_clusters(adata, "coarse", {"0": "B cell"}, key_added="cell_type")
    cats = set(result.obs["cell_type"].astype(str))
    assert "B cell" in cats
    # '1' and '2' were not in the mapping and must keep their original label
    assert "1" in cats
    assert "2" in cats
    assert result.obs["cell_type"].isna().sum() == 0


def test_annotate_clusters_missing_nan_sets_unmapped_to_nan():
    adata = _make_annotation_adata()
    result = annotate_clusters(
        adata, "coarse", {"0": "B cell"}, key_added="cell_type", missing="nan"
    )
    is_zero = result.obs["coarse"] == "0"
    assert (result.obs.loc[is_zero, "cell_type"] == "B cell").all()
    assert result.obs.loc[~is_zero, "cell_type"].isna().all()


def test_annotate_clusters_missing_raise_raises_on_unmapped():
    adata = _make_annotation_adata()
    with pytest.raises(ValueError, match="Unmapped label"):
        annotate_clusters(adata, "coarse", {"0": "B cell"}, missing="raise")


def test_annotate_clusters_missing_raise_succeeds_when_complete():
    adata = _make_annotation_adata()
    mapping = {"0": "B cell", "1": "T cell", "2": "Noise"}
    result = annotate_clusters(adata, "coarse", mapping, missing="raise")
    assert set(result.obs["cell_type"].astype(str)) == {"B cell", "T cell", "Noise"}


def test_annotate_clusters_invalid_missing_raises():
    adata = _make_annotation_adata()
    with pytest.raises(ValueError, match="missing"):
        annotate_clusters(adata, "coarse", {"0": "B cell"}, missing="bogus")


def test_annotate_clusters_bad_cluster_key_raises():
    adata = _make_annotation_adata()
    with pytest.raises(ValueError, match="not found in adata.obs"):
        annotate_clusters(adata, "does_not_exist", {"0": "B cell"})


def test_annotate_clusters_unknown_mapping_key_warns(capsys):
    adata = _make_annotation_adata()
    annotate_clusters(adata, "coarse", {"0": "B cell", "typo_label": "X"})
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "typo_label" in captured.out


def test_annotate_clusters_writes_category_dtype():
    adata = _make_annotation_adata()
    result = annotate_clusters(adata, "coarse", {"0": "B cell"})
    assert str(result.obs["cell_type"].dtype) == "category"


def test_annotate_clusters_records_report_and_tree():
    adata = _make_annotation_adata()
    annotate_clusters(adata, "coarse", {"0": "B cell"})
    assert "clustering_report" in adata.uns
    assert "clustering_tree" in adata.uns


def test_annotate_clusters_record_report_false_writes_no_uns_keys():
    adata = _make_annotation_adata()
    annotate_clusters(adata, "coarse", {"0": "B cell"}, _record_report=False)
    assert "clustering_report" not in adata.uns
    assert "clustering_tree" not in adata.uns


def test_legacy_louvain_without_package_points_to_extra():
    # sys.modules[name] = None makes `import louvain` raise, installed or not.
    with patch.dict("sys.modules", {"louvain": None}):
        with pytest.raises(ImportError, match=r"spacec\[legacy\]"):
            clustering(_make_adata(), clustering="louvain", flavor="legacy", seed=0)
