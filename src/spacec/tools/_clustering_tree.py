"""Clustering family tree: lineage tracking for clustering/subclustering/annotation.

:func:`spacec.tools.clustering`, :func:`spacec.tools.subclustering`, and
:func:`spacec.tools.annotate_clusters` each record what they did into
``adata.uns["clustering_tree"]`` -- a DAG of nodes keyed by
``f"{obs_column}::{label}"``. Subclustering chains a cluster's node to its
children under the new key; annotation chains one or more source cluster
nodes to a single node under the annotation's ``key_added`` column (a
many-to-one *merge* when more than one source cluster shares an annotation),
and that merge node can itself be subclustered again later. :func:`clustering_tree`
renders the whole thing back out as an ASCII tree.

This module is intentionally self-contained (no imports from ``_general.py``,
which is already large) and defensive: every function that mutates or reads
``adata.uns`` catches its own exceptions and prints a warning instead of
raising, so a bookkeeping bug here can never break the clustering pipeline
that called it -- same rationale as ``_append_clustering_report`` in
``_general.py``.
"""

from __future__ import annotations

import json

_TREE_KEY = "clustering_tree"
_ASCII_KEY = "clustering_tree_ascii"


def _empty_tree():
    return {"nodes": {}, "keys": [], "events": []}


def _load_tree(adata):
    """Load the clustering tree from ``adata.uns``, or an empty tree if none
    has been recorded yet. Tolerant of a missing/corrupt entry -- never raises.
    """
    raw = adata.uns.get(_TREE_KEY)
    if not raw:
        return _empty_tree()
    try:
        tree = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return _empty_tree()
    tree.setdefault("nodes", {})
    tree.setdefault("keys", [])
    tree.setdefault("events", [])
    return tree


def _save_tree(adata, tree):
    """Persist ``tree`` to ``adata.uns`` and refresh the cached ASCII render.

    Stored as a JSON string, not a nested dict, because h5ad cannot serialize
    nested dicts -- same pattern (and rationale) as
    ``_append_clustering_report`` in ``_general.py``. Never raises.
    """
    try:
        adata.uns[_TREE_KEY] = json.dumps(tree, default=str)
        adata.uns[_ASCII_KEY] = render_tree(tree)
    except Exception as e:
        print(f"Warning: failed to record clustering tree: {e}")


def _node_id(key, label):
    return f"{key}::{label}"


def _ensure_node(tree, key, label, parents=None, kind="cluster", n_cells=None):
    """Get-or-create a node, registering ``key`` in the ordered ``keys`` list."""
    nid = _node_id(key, label)
    if nid not in tree["nodes"]:
        tree["nodes"][nid] = {
            "key": key,
            "label": str(label),
            "parents": list(parents) if parents else [],
            "kind": kind,
            "n_cells": n_cells,
            "annotation": None,
        }
    if key not in tree["keys"]:
        tree["keys"].append(key)
    return tree["nodes"][nid]


def _ensure_roots(adata, key):
    """Bootstrap root nodes for ``key`` from ``adata.obs[key]`` when the tree
    doesn't already track it -- so :func:`subclustering`/:func:`annotate_clusters`
    still work if the user ran raw ``sc.tl.leiden``/``.map()`` instead of going
    through :func:`clustering` first. No-op if ``key`` is already tracked, or
    absent from ``adata.obs``. Never raises.
    """
    try:
        tree = _load_tree(adata)
        if key in tree["keys"] or key not in adata.obs:
            return tree
        counts = adata.obs[key].astype(str).value_counts()
        for label, n in counts.items():
            _ensure_node(tree, key, label, parents=[], kind="cluster", n_cells=int(n))
        _save_tree(adata, tree)
        return tree
    except Exception as e:
        print(f"Warning: failed to seed clustering tree roots for '{key}': {e}")
        return _load_tree(adata)


def _record_split(adata, cluster_key, key_added, per_cluster_new_labels, params=None):
    """Record a :func:`subclustering` step.

    ``per_cluster_new_labels`` maps each *split* source label to the list of
    new labels it produced. Source labels absent from it are carried forward
    unchanged: same identity, new key -- a fresh node under ``key_added`` with
    the same label, chained to the original node as its single parent.
    """
    try:
        tree = _ensure_roots(adata, cluster_key)
        counts = (
            adata.obs[key_added].astype(str).value_counts().to_dict()
            if key_added in adata.obs
            else {}
        )
        source_labels = adata.obs[cluster_key].astype(str).unique().tolist()
        for label in source_labels:
            parent_id = _node_id(cluster_key, label)
            if parent_id not in tree["nodes"]:
                # Defensive: a source label the tree didn't already track.
                _ensure_node(tree, cluster_key, label, parents=[], kind="cluster")
            new_labels = per_cluster_new_labels.get(label)
            if new_labels:
                for nl in new_labels:
                    n = int(counts.get(nl, 0)) if counts else None
                    _ensure_node(
                        tree,
                        key_added,
                        nl,
                        parents=[parent_id],
                        kind="cluster",
                        n_cells=n,
                    )
            else:
                n = int(counts.get(label, 0)) if counts else None
                _ensure_node(
                    tree,
                    key_added,
                    label,
                    parents=[parent_id],
                    kind="cluster",
                    n_cells=n,
                )
        if params:
            tree["events"].append({"action": "subclustering", **params})
        _save_tree(adata, tree)
    except Exception as e:
        print(f"Warning: failed to record clustering tree split: {e}")


def _record_annotation(adata, cluster_key, key_added, resolved_mapping, params=None):
    """Record an :func:`annotate_clusters` step.

    ``resolved_mapping`` is the final, fully-resolved ``source label -> target
    label`` mapping actually applied to ``adata.obs[key_added]`` (i.e. after
    ``missing=`` has already been applied). Every distinct target label gets
    one node under ``key_added``, parented by every source node that mapped
    to it -- a single parent for an ordinary 1:1 annotation, multiple parents
    for a many-to-one merge (which :func:`render_tree` expands separately, and
    which can be subclustered again later like any other node). Each source
    leaf also gets its ``annotation`` field set, so the tree can show it
    inline even when it's a leaf with no explicit downstream node yet.
    """
    try:
        tree = _ensure_roots(adata, cluster_key)
        counts = (
            adata.obs[cluster_key].astype(str).value_counts().to_dict()
            if cluster_key in adata.obs
            else {}
        )

        targets = {}
        for src, tgt in resolved_mapping.items():
            targets.setdefault(str(tgt), []).append(str(src))

        for tgt, sources in targets.items():
            parent_ids = []
            for src in sources:
                parent_id = _node_id(cluster_key, src)
                if parent_id not in tree["nodes"]:
                    _ensure_node(
                        tree,
                        cluster_key,
                        src,
                        parents=[],
                        kind="cluster",
                        n_cells=int(counts.get(src, 0)) if counts else None,
                    )
                tree["nodes"][parent_id]["annotation"] = tgt
                parent_ids.append(parent_id)
            n_cells = int(sum(counts.get(s, 0) for s in sources)) if counts else None
            _ensure_node(
                tree,
                key_added,
                tgt,
                parents=parent_ids,
                kind="annotation",
                n_cells=n_cells,
            )

        if params:
            tree["events"].append({"action": "annotation", **params})
        _save_tree(adata, tree)
    except Exception as e:
        print(f"Warning: failed to record clustering tree annotation: {e}")


def _sort_key(label):
    """Numeric-aware sort key for comma-separated labels like ``'0,1'``."""
    parts = []
    for p in str(label).split(","):
        try:
            parts.append((0, float(p), ""))
        except ValueError:
            parts.append((1, 0.0, p))
    return parts


def _resolve_display(nodes, children, nid, visited=None):
    """Follow a chain of pass-through nodes (same label, ``kind='cluster'``,
    exactly one child) to whichever node should determine this branch's
    displayed cell count / annotation -- e.g. a cluster carried forward
    unsplit across a later :func:`subclustering` call under a new key.
    """
    visited = visited if visited is not None else set()
    cur = nid
    while cur not in visited:
        visited.add(cur)
        node = nodes[cur]
        kids = children.get(cur, [])
        if len(kids) != 1:
            break
        child = kids[0]
        child_node = nodes[child]
        if child_node["kind"] != "cluster" or child_node["label"] != node["label"]:
            break
        cur = child
    return cur


def render_tree(tree, counts=None):
    """Render a clustering tree dict (as produced by :func:`_load_tree`) as an
    ASCII string. Box-drawing style follows ``display_tile_progress``
    (``_segmentation.py``). Never raises -- returns a warning string instead,
    so a rendering bug can never break a pipeline that prints this for progress.

    Parameters
    ----------
    tree : dict
        ``{"nodes": {...}, "keys": [...], "events": [...]}``.
    counts : dict, optional
        ``{obs_column: {label: n_cells}}`` overrides for live cell counts
        (recomputed from ``adata.obs`` by :func:`clustering_tree`); falls back
        to each node's recorded ``n_cells`` when absent.
    """
    try:
        nodes = tree.get("nodes") or {}
        keys = tree.get("keys") or []
        if not nodes or not keys:
            return "No clustering tree recorded yet."
        counts = counts or {}

        children = {}
        for nid, node in nodes.items():
            for p in node.get("parents", []):
                children.setdefault(p, []).append(nid)

        def n_of(node):
            live = counts.get(node["key"], {})
            if node["label"] in live:
                return live[node["label"]]
            return node.get("n_cells")

        def fmt(n):
            return f"{n:,}" if isinstance(n, (int, float)) else "?"

        lines = ["Clustering family tree"]
        root_key = keys[0]
        root_ids = sorted(
            (
                nid
                for nid, n in nodes.items()
                if n["key"] == root_key and not n["parents"]
            ),
            key=lambda nid: _sort_key(nodes[nid]["label"]),
        )
        total = sum((n_of(nodes[nid]) or 0) for nid in root_ids)
        lines.append(f"{root_key} -- {len(root_ids)} cluster(s), {fmt(total)} cells")
        lines.append("─" * 64)

        merge_queue = []
        queued = set()

        def render_branch(nid, prefix, is_last, out):
            term = _resolve_display(nodes, children, nid)
            term_node = nodes[term]
            label = nodes[nid]["label"]
            n = n_of(term_node)
            connector = "└── " if is_last else "├── "
            text = f"{connector}{label}  ({fmt(n)})"

            ann = term_node.get("annotation")
            kids = children.get(term, [])
            if ann and kids:
                target_id = kids[0]
                is_merge = len(nodes[target_id]["parents"]) > 1
                has_further = bool(children.get(target_id))
                marker = " ↴" if (is_merge or has_further) else ""
                # An identity annotation (missing="keep" carrying a label
                # forward unchanged) would render as "X -> X"; show only the
                # marker in that case.
                if str(ann) != str(label):
                    text += f"  → {ann}{marker}"
                else:
                    text += marker
                out.append(prefix + text)
                if (is_merge or has_further) and target_id not in queued:
                    queued.add(target_id)
                    merge_queue.append(target_id)
                return

            out.append(prefix + text)
            child_prefix = prefix + ("    " if is_last else "│   ")
            real_kids = sorted(kids, key=lambda k: _sort_key(nodes[k]["label"]))
            for i, k in enumerate(real_kids):
                render_branch(k, child_prefix, i == len(real_kids) - 1, out)

        for i, nid in enumerate(root_ids):
            render_branch(nid, "", i == len(root_ids) - 1, lines)

        idx = 0
        while idx < len(merge_queue):
            mid = merge_queue[idx]
            idx += 1
            mnode = nodes[mid]
            sources = "; ".join(
                nodes[p]["label"]
                for p in sorted(
                    mnode["parents"], key=lambda p: _sort_key(nodes[p]["label"])
                )
            )
            lines.append("")
            lines.append(
                f"↴ '{mnode['label']}'  [{mnode['key']}, {fmt(n_of(mnode))} cells]"
                f"  merged from: {sources}"
            )

            def _is_passthrough(k, merge_label=mnode["label"]):
                """True when ``k`` just carries the merge node's own label
                forward unchanged -- rendering it would only repeat the block
                header."""
                term = _resolve_display(nodes, children, k)
                if str(nodes[term]["label"]) != str(merge_label):
                    return False
                return all(
                    str(nodes[c]["label"]) == str(merge_label) and not children.get(c)
                    for c in children.get(term, [])
                )

            kids = sorted(
                (k for k in children.get(mid, []) if not _is_passthrough(k)),
                key=lambda k: _sort_key(nodes[k]["label"]),
            )
            for i, k in enumerate(kids):
                render_branch(k, "   ", i == len(kids) - 1, lines)

        lines.append("")
        lines.append("Legend: → annotation, ↴ merged node expanded below")
        return "\n".join(lines)
    except Exception as e:
        return f"Warning: failed to render clustering tree: {e}"


def clustering_tree(adata, print_tree=True, as_dict=False):
    """
    Render the clustering / subclustering / annotation lineage recorded on
    ``adata`` as an ASCII family tree.

    Every call to :func:`clustering`, :func:`subclustering`, and
    :func:`annotate_clusters` records into ``adata.uns["clustering_tree"]``
    (stored as JSON, so it survives ``.h5ad`` save/reload). This renders that
    record as a tree: a cluster that was later split fans out into its
    subclusters, and a cluster that was annotated shows the assigned cell
    type inline (``→ cell type``); when several clusters were merged into
    one annotation, that merge is expanded once below the main tree
    (``↴``), even if it was later subclustered again.

    Parameters
    ----------
    adata : AnnData
    print_tree : bool, optional
        If True, print the rendered tree. Defaults to True.
    as_dict : bool, optional
        If True, return the raw tree dict (``{"nodes", "keys", "events"}``)
        instead of the rendered ASCII string. Defaults to False.

    Returns
    -------
    str, dict, or None
        The rendered ASCII string (default), the raw tree dict if
        ``as_dict=True``, or ``None`` if no tree has been recorded yet.

    Notes
    -----
    Cell counts are recomputed live from ``adata.obs`` for any recorded key
    still present there, so counts stay accurate after e.g.
    ``adata = adata[~adata.obs['cell_type'].isin(['Noise'])]``; keys no
    longer present in ``adata.obs`` fall back to the count recorded at
    write time.
    """
    tree = _load_tree(adata)
    if not tree.get("nodes"):
        if print_tree:
            print("No clustering tree recorded yet.")
        return None

    counts = {}
    for key in tree.get("keys", []):
        if key in adata.obs:
            counts[key] = adata.obs[key].astype(str).value_counts().to_dict()

    ascii_tree = render_tree(tree, counts=counts)
    if print_tree:
        print(ascii_tree)
    return tree if as_dict else ascii_tree
