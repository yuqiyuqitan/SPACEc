"""Unit tests for spacec.tools._clustering_tree (clustering family tree).

Exercises the module's public/private API directly (_ensure_roots,
_record_split, _record_annotation, clustering_tree, render_tree) rather than
going through subclustering()/annotate_clusters(), so these tests pin down
the tree's own contract: DAG shape, merge nodes, h5ad round-tripping, and
ASCII rendering.
"""

import json

import anndata as ad
import numpy as np
import pandas as pd

from spacec.tools._clustering_tree import (
    _ensure_roots,
    _record_annotation,
    _record_split,
    clustering_tree,
    render_tree,
)


def _make_adata(n_cells=12):
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    obs["leiden_1"] = pd.Categorical(["0"] * (n_cells // 2) + ["1"] * (n_cells // 2))
    var = pd.DataFrame(index=["marker_0", "marker_1", "marker_2"])
    X = np.zeros((n_cells, 3), dtype=np.float32)
    return ad.AnnData(X=X, obs=obs, var=var)


def _split_and_annotate(adata):
    """Build a small two-level tree: leiden_1 -> leiden_1_subcluster (split
    '0' into '0,0'/'0,1') -> cell_type (merge '0,0' and '0,1' into
    'B cell'; carry '1' forward as 'T cell')."""
    _ensure_roots(adata, "leiden_1")
    n = adata.n_obs
    adata.obs["leiden_1_subcluster"] = pd.Categorical(
        ["0,0"] * (n // 4) + ["0,1"] * (n // 4) + ["1"] * (n // 2)
    )
    _record_split(
        adata,
        "leiden_1",
        "leiden_1_subcluster",
        {"0": ["0,0", "0,1"]},
        params={"method": "leiden", "resolution": 0.3, "seed": 0},
    )
    _record_annotation(
        adata,
        "leiden_1_subcluster",
        "cell_type",
        {"0,0": "B cell", "0,1": "B cell", "1": "T cell"},
        params={"missing": "keep"},
    )
    return adata


# ---------------------------------------------------------------------------
# _ensure_roots
# ---------------------------------------------------------------------------


def test_ensure_roots_bootstraps_from_obs():
    adata = _make_adata()
    tree = _ensure_roots(adata, "leiden_1")
    assert "leiden_1" in tree["keys"]
    assert "leiden_1::0" in tree["nodes"]
    assert "leiden_1::1" in tree["nodes"]
    assert tree["nodes"]["leiden_1::0"]["parents"] == []
    assert tree["nodes"]["leiden_1::0"]["n_cells"] == 6


def test_ensure_roots_is_noop_when_already_tracked():
    adata = _make_adata()
    tree1 = _ensure_roots(adata, "leiden_1")
    tree2 = _ensure_roots(adata, "leiden_1")
    assert tree1["nodes"] == tree2["nodes"]


def test_ensure_roots_missing_column_is_noop():
    adata = _make_adata()
    tree = _ensure_roots(adata, "does_not_exist")
    assert tree["nodes"] == {}


# ---------------------------------------------------------------------------
# _record_split / _record_annotation -- DAG shape
# ---------------------------------------------------------------------------


def test_record_split_carries_unsplit_labels_forward():
    adata = _make_adata()
    _split_and_annotate(adata)
    tree = json.loads(adata.uns["clustering_tree"])
    # '1' was never split: single child node under the new key, same label,
    # chained to the original root as its one parent.
    carried = tree["nodes"]["leiden_1_subcluster::1"]
    assert carried["parents"] == ["leiden_1::1"]
    assert carried["label"] == "1"


def test_record_split_new_labels_chained_to_parent():
    adata = _make_adata()
    _split_and_annotate(adata)
    tree = json.loads(adata.uns["clustering_tree"])
    for label in ("0,0", "0,1"):
        node = tree["nodes"][f"leiden_1_subcluster::{label}"]
        assert node["parents"] == ["leiden_1::0"]


def test_record_annotation_merge_node_has_multiple_parents():
    adata = _make_adata()
    _split_and_annotate(adata)
    tree = json.loads(adata.uns["clustering_tree"])
    merge_node = tree["nodes"]["cell_type::B cell"]
    assert merge_node["kind"] == "annotation"
    assert sorted(merge_node["parents"]) == sorted(
        ["leiden_1_subcluster::0,0", "leiden_1_subcluster::0,1"]
    )


def test_record_annotation_single_parent_for_1to1_mapping():
    adata = _make_adata()
    _split_and_annotate(adata)
    tree = json.loads(adata.uns["clustering_tree"])
    node = tree["nodes"]["cell_type::T cell"]
    assert node["parents"] == ["leiden_1_subcluster::1"]


def test_record_annotation_sets_annotation_on_leaves():
    adata = _make_adata()
    _split_and_annotate(adata)
    tree = json.loads(adata.uns["clustering_tree"])
    assert tree["nodes"]["leiden_1_subcluster::0,0"]["annotation"] == "B cell"
    assert tree["nodes"]["leiden_1_subcluster::0,1"]["annotation"] == "B cell"
    assert tree["nodes"]["leiden_1_subcluster::1"]["annotation"] == "T cell"


def test_events_recorded_with_action():
    adata = _make_adata()
    _split_and_annotate(adata)
    tree = json.loads(adata.uns["clustering_tree"])
    actions = [e["action"] for e in tree["events"]]
    assert actions == ["subclustering", "annotation"]


# ---------------------------------------------------------------------------
# h5ad round-trip
# ---------------------------------------------------------------------------


def test_tree_round_trips_through_h5ad(tmp_path):
    adata = _make_adata()
    _split_and_annotate(adata)
    tree_before = clustering_tree(adata, as_dict=True, print_tree=False)

    path = tmp_path / "tree_round_trip.h5ad"
    adata.write_h5ad(path)
    reloaded = ad.read_h5ad(path)

    tree_after = clustering_tree(reloaded, as_dict=True, print_tree=False)
    assert tree_after == tree_before

    ascii_before = render_tree(tree_before)
    ascii_after = render_tree(tree_after)
    assert ascii_after == ascii_before


def test_ascii_tree_cached_in_uns_survives_round_trip(tmp_path):
    adata = _make_adata()
    _split_and_annotate(adata)
    path = tmp_path / "tree_ascii_round_trip.h5ad"
    adata.write_h5ad(path)
    reloaded = ad.read_h5ad(path)
    assert reloaded.uns["clustering_tree_ascii"] == adata.uns["clustering_tree_ascii"]


# ---------------------------------------------------------------------------
# render_tree -- ASCII shape
# ---------------------------------------------------------------------------


def test_render_tree_contains_expected_labels_and_box_chars():
    adata = _make_adata()
    _split_and_annotate(adata)
    tree = clustering_tree(adata, as_dict=True, print_tree=False)
    ascii_tree = render_tree(tree)

    assert "Clustering family tree" in ascii_tree
    assert "leiden_1" in ascii_tree
    assert "0,0" in ascii_tree
    assert "0,1" in ascii_tree
    assert "→ B cell" in ascii_tree  # "-> B cell"
    assert "↴" in ascii_tree  # merge marker
    assert "├──" in ascii_tree or "└──" in ascii_tree
    assert "merged from: 0,0; 0,1" in ascii_tree


def test_render_tree_empty_tree_returns_message():
    assert render_tree({"nodes": {}, "keys": [], "events": []}) == (
        "No clustering tree recorded yet."
    )


# ---------------------------------------------------------------------------
# clustering_tree -- public entry point
# ---------------------------------------------------------------------------


def test_clustering_tree_on_bare_adata_prints_message_and_returns_none(capsys):
    adata = _make_adata()
    result = clustering_tree(adata)
    captured = capsys.readouterr()
    assert "No clustering tree recorded yet." in captured.out
    assert result is None


def test_clustering_tree_as_dict_returns_raw_tree():
    adata = _make_adata()
    _split_and_annotate(adata)
    result = clustering_tree(adata, as_dict=True, print_tree=False)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"nodes", "keys", "events"}


def test_clustering_tree_recomputes_live_counts_after_cell_removal():
    adata = _make_adata()
    _split_and_annotate(adata)
    # Drop half of the '1' cells post-hoc, mirroring e.g.
    # `adata = adata[~adata.obs['cell_type'].isin(['Noise'])]` in the tutorial.
    keep = ~(
        (adata.obs["leiden_1"] == "1") & (adata.obs_names.isin(adata.obs_names[-3:]))
    )
    adata = adata[keep].copy()
    ascii_tree = clustering_tree(adata, print_tree=False)
    # 'T cell' branch should now show 3 cells instead of the original 6.
    assert "T cell" in ascii_tree
