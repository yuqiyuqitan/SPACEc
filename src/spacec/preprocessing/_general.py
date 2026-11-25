# load required packages
import glob
import os
import sys
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy import sparse
from scipy.linalg import lstsq
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.stats import norm, zscore


# read the data frame output from the segmentation functions
def read_segdf(
    segfile_list,
    seg_method="cellpose",
    region_list=None,  # optional information #please make sure the length of each list matches
    meta_list=None,  # optional information
    add_unique_id=True,
):
    """
    Read the data frame output from the segmentation functions.

    Parameters
    ----------
    segfile_list : list
        List of segmented csv files to be read.
    seg_method : str, optional
        Name of the segmentation method that generated the input files, by
        default ``'cellpose'``. This parameter is primarily a legacy
        compatibility feature: ``'cellseg'`` activates special handling for
        older CellSeg-style output files, including column-name cleanup,
        resetting the original index into ``first_index``, and renaming
        ``size`` to ``area``. For other values, files are read using the
        standard format; in particular, when ``add_unique_id=True``, the
        original per-file mask identifier is retained in a ``label`` column.
    region_list : list, optional
        List of regions, by default None. Please make sure the length of each list matches.
    meta_list : list or dict, optional
        Metadata to attach per segfile, by default None. A flat list adds one value per
        segfile into a single `condition` column (legacy behavior, unchanged). A dict of
        `{column_name: [value_per_segfile, ...]}` instead adds one column per key, for
        multiple metadata fields at once (e.g. `{"condition": [...], "patient_id": [...]}`).
        Each list's length must match `segfile_list`.
    add_unique_id : bool, optional
        If True (default), add a ``unique_cell_id`` column built from the region
        identifier and each row's original per-file index (the mask label for
        Cellpose output, or the legacy ``cell_id`` for ``seg_method='cellseg'``).
        This is needed because per-file labels/cell_ids restart near 1 for each
        region, so they collide once multiple regions are concatenated;
        ``unique_cell_id`` stays unique across the whole dataset and is picked up
        automatically by :func:`spacec.hf.make_anndata` to set ``adata.obs_names``.
        For ``seg_method != 'cellseg'`` this also adds a plain ``label`` column
        holding the raw per-region mask label (otherwise it only lives in the
        DataFrame index, which functions like :func:`remove_noise` discard via
        ``reset_index(drop=True)``) -- keeping ``label`` as an explicit column
        is what lets a cell be traced back to its exact pixels in the original
        segmentation mask via :func:`spacec.hf.hf_attach_masks`, even after
        filtering and normalization. Set to False to reproduce the pre-existing
        output exactly.

    Returns
    -------
    df : pandas.DataFrame
        The concatenated DataFrame from all the segmentation files.

    Raises
    ------
    SystemExit
        If the length of region_list or meta_list does not match with segfile_list.
    """
    if region_list is not None:
        if len(region_list) != len(segfile_list):
            sys.exit("length of each list does not match!")
    if meta_list is not None:
        if isinstance(meta_list, dict):
            for col, values in meta_list.items():
                if len(values) != len(segfile_list):
                    sys.exit(
                        f"length of meta_list[{col!r}] does not match segfile_list!"
                    )
        elif len(meta_list) != len(segfile_list):
            sys.exit("length of each list does not match!")

    df = pd.DataFrame()
    # concat old dataframes
    for i in range(len(segfile_list)):
        tmp = pd.read_csv(segfile_list[i], index_col=0)
        tmp["region_num"] = str(i)
        if region_list is not None:
            tmp["unique_region"] = str(region_list[i])
        if meta_list is not None:
            if isinstance(meta_list, dict):
                for col, values in meta_list.items():
                    tmp[col] = str(values[i])
            else:
                tmp["condition"] = str(meta_list[i])
        if add_unique_id:
            # tmp.index here is always the raw per-file identifier (mask label
            # for Cellpose output, legacy cell_id for 'cellseg') regardless of
            # seg_method, since index_col=0 above applies uniformly before any
            # seg_method-specific handling below.
            region_key = str(region_list[i]) if region_list is not None else str(i)
            tmp["unique_cell_id"] = region_key + "_" + tmp.index.astype(str)
            if seg_method != "cellseg":
                # The 'cellseg' branch below already promotes the raw index
                # into an explicit 'cell_id' column itself. For all other
                # (real) paths nothing else preserves it as a column, so add
                # it here under its natural name.
                tmp["label"] = tmp.index
        df = pd.concat([df, tmp], axis=0)

    if seg_method == "cellseg":
        # See resultant dataframe
        df.columns = df.columns.str.split(":").str[-1].tolist()
        df = df.reset_index().rename(columns={"index": "first_index"})
        df.columns = df.columns.str.split(":").str[-1].tolist()
        df.rename(columns={"size": "area"}, inplace=True)
    return df


def plot_filter_dashboard(
    df_flagged,
    summary,
    thresholds,
    area_col="area",
    nuc_col="DAPI",
    region_col="unique_region",
    top_regions_for_scatter=12,
    flag_cols=None,
    style=None,
):
    """
    Render a QC dashboard showing the effect of area + nuclear-marker filtering.

    5 panels if no flags were used, 6 if they were (an extra "why flagged"
    panel; see below).

    Parameters
    ----------
    df_flagged : pandas.DataFrame
        Input data with additional boolean column ``keep`` and string column
        ``removal_reason`` (value ``"kept"`` for passing cells). If
        `flag_cols` is given, must also carry those boolean columns.
    summary : pandas.DataFrame
        Per-region aggregation with columns ``region_col``, ``n_total``,
        ``n_kept``, ``pct_kept``, and -- if `flag_cols` is given --
        ``pct_flagged`` (share of *kept* cells with any flag set).
    thresholds : dict
        Keys ``area_low_global``, ``area_high_global``, ``nuc_low_global``,
        ``nuc_high_global``, ``area_scope`` (``"global"``/``"region"``),
        ``nuc_scope`` (``"global"``/``"region"``). Optionally also
        ``area_flag_low_global``, ``area_flag_high_global``,
        ``nuc_flag_low_global`` -- looser, flag-only thresholds drawn as a
        dashed orange line alongside the black drop-threshold line.
    area_col : str
        Column name for cell area / size.
    nuc_col : str
        Column name for nuclear marker intensity.
    region_col : str
        Column name used for per-region grouping.
    top_regions_for_scatter : int
        Maximum number of regions shown in the scatter panel.
    flag_cols : list of str, optional
        Names of the boolean ``qc_flag_*`` columns present in `df_flagged`
        (excluding the combined ``qc_flag_any``). ``None`` (default) omits
        every flag-specific addition -- panel A's second (stacked) segment,
        the "why flagged" panel, the orange threshold lines/legend entry --
        and renders the original 5-panel dashboard.
    style : SPACEcStyle, str, or pathlib.Path, optional
        Style to render with. ``None`` (default) uses the active style set
        via ``sp.pl.set_style``/``with sp.pl.style(...):``, NOT hardcoded to
        the publication preset -- so e.g. a notebook-wide
        ``sp.pl.set_style(sp.pl.SPACEcStyle.presentation())`` is respected
        here too, rather than silently reverting to publication's small
        (6-10pt) fonts.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled dashboard figure. Not shown or saved here -- that is
        the caller's responsibility.
    """
    from spacec.plotting._style import _apply_spines, _apply_style

    _style = _apply_style(style)
    flag_cols = flag_cols or []

    # human-readable flag labels, phrased like removal_reason ("low DAPI", "small area", ...)
    # rather than the literal qc_flag_* column names
    flag_labels = {
        "qc_flag_low_dapi": f"low {nuc_col}",
        "qc_flag_low_size": f"small {area_col}",
        "qc_flag_large_size": f"large {area_col}",
        "qc_flag_any": "any flag",
    }

    area_global = thresholds.get("area_scope", "global") == "global"
    nuc_global = thresholds.get("nuc_scope", "global") == "global"

    area_low_g = thresholds.get("area_low_global")
    area_high_g = thresholds.get("area_high_global")
    nuc_low_g = thresholds.get("nuc_low_global")
    nuc_high_g = thresholds.get("nuc_high_global")
    area_flag_low_g = thresholds.get("area_flag_low_global")
    area_flag_high_g = thresholds.get("area_flag_high_global")
    nuc_flag_low_g = thresholds.get("nuc_flag_low_global")

    # proxy handles for the dashed threshold lines, shared by the distribution/scatter panels'
    # legends
    threshold_handles = [
        Line2D(
            [0], [0], color="black", linestyle="--", linewidth=1, label="drop threshold"
        ),
    ]
    if flag_cols:
        threshold_handles.append(
            Line2D(
                [0],
                [0],
                color="orange",
                linestyle="--",
                linewidth=1,
                label="flag threshold",
            )
        )
    threshold_labels = [h.get_label() for h in threshold_handles]

    with _style.context():
        fig = plt.figure(figsize=(22, 14), constrained_layout=True)
        # NOT using _style.subplot_hspace/wspace here (default 0.4) -- checked, and under
        # constrained_layout that value means something far more aggressive than the same
        # number does at its one other real call site (plotting/_general.py, a plain
        # subplots(gridspec_kw=...) call with no constrained_layout) -- applying it here blew
        # the gap between the two rows out to roughly half the figure. Using a small
        # constrained-layout-native bump instead: h_pad (suptitle-to-first-row gap) and
        # hspace (between the two rows) -- neither has a working _style.py equivalent under
        # this layout engine.
        fig.get_layout_engine().set(h_pad=0.12, hspace=0.04)
        gs = fig.add_gridspec(2, 3)

        # A) Retention by region -- one stacked bar per region (kept, split into
        # not-flagged/flagged) when flags were given, a plain "% kept" bar otherwise
        ax1 = fig.add_subplot(gs[0, 0])
        s1 = summary.sort_values("pct_kept", ascending=True).reset_index(drop=True)
        ypos = np.arange(len(s1))
        if flag_cols:
            flagged_abs = s1["pct_kept"] * s1["pct_flagged"] / 100
            not_flagged_abs = s1["pct_kept"] - flagged_abs
            ax1.barh(
                ypos,
                not_flagged_abs,
                height=0.8,
                color="#2a9d8f",
                label="kept, not flagged",
            )
            ax1.barh(
                ypos,
                flagged_abs,
                left=not_flagged_abs,
                height=0.8,
                color="#f4a261",
                label="kept, flagged",
            )
            ax1.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
        else:
            ax1.barh(ypos, s1["pct_kept"], height=0.8, color="#2a9d8f")
        ax1.set_yticks(ypos)
        ax1.set_yticklabels(s1[region_col].astype(str))
        ax1.set_title("Cells Kept by Region")
        ax1.set_xlabel("% of total cells")
        ax1.set_ylabel("Region")
        ax1.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax1.set_xlim(0, 100)

        # B) Why cells were removed
        ax2 = fig.add_subplot(gs[0, 1])
        removed_df = df_flagged.loc[~df_flagged["keep"]]
        if len(removed_df) > 0:
            reason_counts = (
                removed_df["removal_reason"]
                .value_counts(normalize=True)
                .mul(100)
                .sort_values(ascending=True)
            )
            ax2.barh(reason_counts.index, reason_counts.values, color="#e76f51")
        ax2.set_title("Why Cells Were Removed")
        ax2.set_xlabel("% of removed cells")
        ax2.xaxis.set_major_formatter(PercentFormatter(xmax=100))

        axes_list = [ax1, ax2]

        # B2) Why kept cells were flagged -- own panel, only when flags were given: each
        # flag's own marginal rate (including the combined qc_flag_any), plus a bar for
        # every multi-flag combination actually observed in the data (not every
        # theoretical combination -- mirrors removal_reason's own "small & low DAPI" style)
        if flag_cols:
            ax2b = fig.add_subplot(gs[0, 2])
            kept_df = df_flagged.loc[df_flagged["keep"]]
            bar_labels2, bar_values2 = [], []
            for col in flag_cols + ["qc_flag_any"]:
                bar_labels2.append(flag_labels.get(col, col))
                bar_values2.append(kept_df[col].mean() * 100)
            if len(flag_cols) > 1:
                combo = kept_df[flag_cols].apply(
                    lambda row: " & ".join(
                        flag_labels.get(c, c) for c in flag_cols if row[c]
                    ),
                    axis=1,
                )
                # % of ALL kept cells (matching the individual-flag bars above), not of just
                # the multi-flagged subset -- value_counts(normalize=True) on the already-
                # filtered-to-multi-flag subset would normalize against that subset's own
                # total instead, silently inflating these percentages (found via review: a
                # combo bar showing ~80% of kept cells was actually ~80% of the much smaller
                # multi-flagged subset).
                combo_counts = (
                    combo[combo.str.contains(" & ", regex=False)]
                    .value_counts()
                    .div(len(kept_df))
                    .mul(100)
                )
                for label, val in combo_counts.items():
                    bar_labels2.append(label)
                    bar_values2.append(val)
            order = np.argsort(bar_values2)
            ax2b.barh(
                [bar_labels2[i] for i in order],
                [bar_values2[i] for i in order],
                color="#f4a261",
            )
            ax2b.set_title("Why Kept Cells Were Flagged")
            ax2b.set_xlabel("% of kept cells")
            ax2b.xaxis.set_major_formatter(PercentFormatter(xmax=100))
            axes_list.append(ax2b)

        # Remaining panels: distribution/scatter positions shift down a row (and the
        # scatter shrinks to one column slot) once the B2 panel above claims row 0's
        # third slot -- otherwise this is the original 5-panel layout.
        if flag_cols:
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            ax5 = fig.add_subplot(gs[1, 2])
        else:
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[1, 0])
            ax5 = fig.add_subplot(gs[1, 1:])
        axes_list += [ax3, ax4, ax5]

        # C) Area distribution
        sns.kdeplot(
            data=np.log10(df_flagged[area_col].clip(lower=1)),
            fill=True,
            alpha=0.25,
            label="Before",
            ax=ax3,
            color="#264653",
        )
        sns.kdeplot(
            data=np.log10(df_flagged.loc[df_flagged["keep"], area_col].clip(lower=1)),
            fill=True,
            alpha=0.25,
            label="After",
            ax=ax3,
            color="#2a9d8f",
        )
        if area_global and area_low_g is not None:
            ax3.axvline(
                np.log10(max(area_low_g, 1)), linestyle="--", color="black", linewidth=1
            )
        if area_global and area_high_g is not None:
            ax3.axvline(
                np.log10(max(area_high_g, 1)),
                linestyle="--",
                color="black",
                linewidth=1,
            )
        if area_global and area_flag_low_g is not None:
            ax3.axvline(
                np.log10(max(area_flag_low_g, 1)),
                linestyle="--",
                color="orange",
                linewidth=1,
            )
        if area_global and area_flag_high_g is not None:
            ax3.axvline(
                np.log10(max(area_flag_high_g, 1)),
                linestyle="--",
                color="orange",
                linewidth=1,
            )
        ax3.set_title("Area Distribution (log10)")
        ax3.set_xlabel(f"log10({area_col})")
        h3, l3 = ax3.get_legend_handles_labels()
        ax3.legend(
            handles=h3 + threshold_handles, labels=l3 + threshold_labels, frameon=False
        )

        # D) Nuclear marker distribution
        sns.kdeplot(
            data=np.log10(df_flagged[nuc_col].clip(lower=1)),
            fill=True,
            alpha=0.25,
            label="Before",
            ax=ax4,
            color="#264653",
        )
        sns.kdeplot(
            data=np.log10(df_flagged.loc[df_flagged["keep"], nuc_col].clip(lower=1)),
            fill=True,
            alpha=0.25,
            label="After",
            ax=ax4,
            color="#2a9d8f",
        )
        if nuc_global and nuc_low_g is not None:
            ax4.axvline(
                np.log10(max(nuc_low_g, 1)), linestyle="--", color="black", linewidth=1
            )
        if nuc_global and nuc_high_g is not None:
            ax4.axvline(
                np.log10(max(nuc_high_g, 1)), linestyle="--", color="black", linewidth=1
            )
        if nuc_global and nuc_flag_low_g is not None:
            ax4.axvline(
                np.log10(max(nuc_flag_low_g, 1)),
                linestyle="--",
                color="orange",
                linewidth=1,
            )
        ax4.set_title(f"{nuc_col} Distribution (log10)")
        ax4.set_xlabel(f"log10({nuc_col})")
        h4, l4 = ax4.get_legend_handles_labels()
        ax4.legend(
            handles=h4 + threshold_handles, labels=l4 + threshold_labels, frameon=False
        )

        # E) Scatter — top regions by pct_kept, subsampled for performance
        top_regions = (
            summary[region_col].astype(str).head(top_regions_for_scatter).tolist()
        )
        dsub = df_flagged[df_flagged[region_col].astype(str).isin(top_regions)].copy()
        sns.scatterplot(
            data=dsub,
            x=nuc_col,
            y=area_col,
            hue="keep",
            style=region_col,
            alpha=0.45,
            s=18,
            ax=ax5,
            palette={True: "#2a9d8f", False: "#e76f51"},
        )
        ax5.set_xscale("log")
        ax5.set_yscale("log")
        if nuc_global and nuc_low_g is not None:
            ax5.axvline(nuc_low_g, linestyle="--", color="black", linewidth=1)
        if nuc_global and nuc_high_g is not None:
            ax5.axvline(nuc_high_g, linestyle="--", color="black", linewidth=1)
        if area_global and area_low_g is not None:
            ax5.axhline(area_low_g, linestyle="--", color="black", linewidth=1)
        if area_global and area_high_g is not None:
            ax5.axhline(area_high_g, linestyle="--", color="black", linewidth=1)
        if nuc_global and nuc_flag_low_g is not None:
            ax5.axvline(nuc_flag_low_g, linestyle="--", color="orange", linewidth=1)
        if area_global and area_flag_low_g is not None:
            ax5.axhline(area_flag_low_g, linestyle="--", color="orange", linewidth=1)
        if area_global and area_flag_high_g is not None:
            ax5.axhline(area_flag_high_g, linestyle="--", color="orange", linewidth=1)
        ax5.set_title("Threshold QC Scatter (Top Regions by Kept %)")
        ax5.set_xlabel(nuc_col)
        ax5.set_ylabel(area_col)
        h5, l5 = ax5.get_legend_handles_labels()
        ax5.legend(
            handles=h5 + threshold_handles,
            labels=l5 + threshold_labels,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            frameon=False,
        )

        for ax in axes_list:
            _apply_spines(ax, _style)

        fig.suptitle("QC Report: Area + Nuclear-Marker Threshold Filtering")

    return fig


def filter_data(
    df,
    nuc_thres=1,
    size_thres=1,
    nuc_thres_high=None,
    size_thres_high=None,
    nuc_marker="DAPI",
    cell_size="area",
    region_column="region_num",
    palette="Paired",
    alpha=0.8,
    size=0.4,
    log_scale=False,
    qc_dashboard=False,
    nuc_flag_thres=None,
    size_flag_thres=None,
    size_upper_flag_thres=None,
    return_summary=False,
    style=None,
    color_by=None,
):
    """
    Filter cells based on nuclear-marker intensity and cell-size thresholds,
    with an optional QC dashboard explaining which cells were removed and why.

    Parameters
    ----------
    df : pandas.DataFrame
        Segmentation output DataFrame.
    nuc_thres : float
        Lower nuclear-marker intensity cutoff; cells at or below this value
        are removed.
    size_thres : float
        Lower cell-size cutoff; cells at or below this value are removed.
    nuc_thres_high : float, optional
        Upper nuclear-marker intensity cutoff (e.g. to remove saturated cells).
        ``None`` disables the upper bound.
    size_thres_high : float, optional
        Upper cell-size cutoff. ``None`` disables the upper bound.
    nuc_marker : str
        Column name for the nuclear marker, by default ``"DAPI"``.
    cell_size : str
        Column name for cell area / size, by default ``"area"``.
    region_column : str
        Column used to identify tissue regions -- the grouping key for every panel of the
        QC dashboard (region labels, per-region crosstabs, and the scatter panel's marker
        shape). By default ``"region_num"``.
    palette : str
        Has no effect -- the current QC dashboard hardcodes its own colors (kept vs.
        removed) instead. Accepted only for backward compatibility with SPACEc 0.1.2,
        where it controlled an older, simpler scatter plot that this dashboard replaced.
    alpha : float
        Has no effect, for the same reason as ``palette``.
    size : float
        Has no effect, for the same reason as ``palette``.
    log_scale : bool
        Legacy parameter; the dashboard always uses log scale for the scatter.
    qc_dashboard : bool
        When ``True``, build the 5-panel QC dashboard (renamed from ``plot``)
        and return it as a second return value -- see ``Returns`` below. The
        figure is only built and returned, never shown or saved; that is the
        caller's responsibility (``fig.show()``/``fig.savefig()``).
    nuc_flag_thres : float, optional
        A looser nuclear-marker threshold. Cells that pass ``nuc_thres`` above
        but still fall at or below this value are flagged (not dropped) via
        a new ``qc_flag_low_dapi`` column. ``None`` (default) adds no flag
        columns and leaves the returned DataFrame's columns unchanged.
    size_flag_thres : float, optional
        A looser lower size threshold, flagged the same way as
        ``nuc_flag_thres`` via a new ``qc_flag_low_size`` column. ``None`` by
        default.
    size_upper_flag_thres : float, optional
        An upper size threshold. Cells at or above this value are flagged via
        a new ``qc_flag_large_size`` column -- independent of
        ``size_thres_high``, which (if set) drops those cells outright
        instead. ``None`` by default.
    return_summary : bool
        When ``True``, also return the per-region summary table (see
        ``Returns``). ``False`` by default.
    style : SPACEcStyle, str, or pathlib.Path, optional
        Style for the QC dashboard (ignored if ``qc_dashboard=False``).
        ``None`` uses the active style (``sp.pl.set_style``), not a
        hardcoded preset.
    color_by : str, optional
        Deprecated alias for ``region_column``, kept only for backwards compatibility --
        raises a ``DeprecationWarning`` and overrides ``region_column`` if given. Despite
        the name, it was never a true per-point color control (see ``palette`` above);
        it only ever set the grouping key documented under ``region_column``. Will be
        removed in a future version.

    Returns
    -------
    df_nuc : pandas.DataFrame
        Filtered DataFrame containing only cells that pass all thresholds.
        If any ``*_flag_thres`` argument is given, also carries the
        corresponding ``qc_flag_*`` boolean column(s) plus a combined
        ``qc_flag_any``. Always the first return value.
    qc_dashboard_fig : matplotlib.figure.Figure
        The QC dashboard. Only returned if ``qc_dashboard=True`` -- i.e.
        ``df_nuc, qc_dashboard_fig = filter_data(..., qc_dashboard=True)``,
        or just ``df_nuc = filter_data(...)`` otherwise.
    summary : pandas.DataFrame
        Per-region ``n_total``/``n_kept``/``pct_kept`` (+ ``pct_flagged`` if
        any flag threshold was given). Only returned if ``return_summary=True``,
        appended after ``qc_dashboard_fig`` if that is also requested --
        e.g. ``df_nuc, fig, summary = filter_data(..., qc_dashboard=True, return_summary=True)``
        or ``df_nuc, summary = filter_data(..., return_summary=True)``.
    """
    if color_by is not None:
        warnings.warn(
            "'color_by' is deprecated and will be removed in a future version. "
            "Use 'region_column' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        region_column = color_by
    color_by = region_column

    for name, value, default in (
        ("palette", palette, "Paired"),
        ("alpha", alpha, 0.8),
        ("size", size, 0.4),
    ):
        if value != default:
            warnings.warn(
                f"'{name}' has no effect -- the current QC dashboard hardcodes its own "
                "colors/sizing instead. Accepted only for backward compatibility with "
                "SPACEc 0.1.2.",
                UserWarning,
                stacklevel=2,
            )

    # --- per-cell removal masks ---
    mask_size_low = df[cell_size] <= size_thres
    mask_nuc_low = df[nuc_marker] <= nuc_thres
    mask_size_high = (
        df[cell_size] >= size_thres_high
        if size_thres_high is not None
        else pd.Series(False, index=df.index)
    )
    mask_nuc_high = (
        df[nuc_marker] >= nuc_thres_high
        if nuc_thres_high is not None
        else pd.Series(False, index=df.index)
    )

    removed_mask = mask_size_low | mask_size_high | mask_nuc_low | mask_nuc_high

    # assign removal reason (most informative label wins)
    reason = pd.Series("kept", index=df.index)
    reason[mask_size_low] = f"small {cell_size}"
    reason[mask_size_high] = f"large {cell_size}"
    reason[mask_nuc_low] = f"low {nuc_marker}"
    reason[mask_nuc_high] = f"high {nuc_marker}"
    reason[mask_size_low & mask_nuc_low] = f"small & low {nuc_marker}"

    df_flagged = df.assign(keep=~removed_mask, removal_reason=reason)

    df_nuc = df_flagged[df_flagged["keep"]].drop(columns=["keep", "removal_reason"])

    # --- non-destructive QC flags on the surviving cells (looser than the drop thresholds) ---
    flag_cols = []
    if nuc_flag_thres is not None:
        df_nuc["qc_flag_low_dapi"] = df_nuc[nuc_marker] <= nuc_flag_thres
        flag_cols.append("qc_flag_low_dapi")
    if size_flag_thres is not None:
        df_nuc["qc_flag_low_size"] = df_nuc[cell_size] <= size_flag_thres
        flag_cols.append("qc_flag_low_size")
    if size_upper_flag_thres is not None:
        df_nuc["qc_flag_large_size"] = df_nuc[cell_size] >= size_upper_flag_thres
        flag_cols.append("qc_flag_large_size")
    if flag_cols:
        df_nuc["qc_flag_any"] = df_nuc[flag_cols].any(axis=1)
        # propagate onto df_flagged too (False for already-removed cells) so
        # plot_filter_dashboard's own "kept" subset of df_flagged carries them.
        # Uses the boolean `keep` mask (not `.loc[df_nuc.index]`) since concatenated
        # multi-region data commonly has a non-unique index, which breaks index-based
        # assignment (each duplicate label would match multiple rows).
        keep_mask = df_flagged["keep"]
        for col in flag_cols + ["qc_flag_any"]:
            df_flagged[col] = False
            df_flagged.loc[keep_mask, col] = df_nuc[col].values

    # human-readable flag labels, phrased like removal_reason ("low DAPI", "small area", ...)
    # rather than the literal qc_flag_* column names -- shared by the summary table below and
    # the printed text summary further down
    flag_labels = {
        "qc_flag_low_dapi": f"low {nuc_marker}",
        "qc_flag_low_size": f"small {cell_size}",
        "qc_flag_large_size": f"large {cell_size}",
        "qc_flag_any": "any flag",
    }

    # per-region summary (built after flags so the per-reason/per-flag columns can be included).
    # Deliberately kept per-region (not collapsed to one overall row) so the same table can
    # answer both "what does region X look like" and, summed across rows, "what does the whole
    # dataset look like" -- matching (and exceeding) the detail in the printed text summary
    # below: counts, not just percentages, since counts are what's actually summable across
    # regions.
    summary = (
        df_flagged.groupby(color_by)
        .agg(n_total=(cell_size, "count"), n_kept=("keep", "sum"))
        .reset_index()
    )
    summary["n_removed"] = summary["n_total"] - summary["n_kept"]
    summary["pct_kept"] = summary["n_kept"] / summary["n_total"] * 100

    reason_by_region = pd.crosstab(
        df_flagged[color_by], df_flagged["removal_reason"]
    ).drop(columns=["kept"], errors="ignore")
    if not reason_by_region.empty:
        reason_by_region.columns = [f"removed: {c}" for c in reason_by_region.columns]
        summary = summary.merge(reason_by_region.reset_index(), on=color_by, how="left")
        reason_cols = list(reason_by_region.columns)
        summary[reason_cols] = summary[reason_cols].fillna(0).astype(int)

    if flag_cols:
        flagged_by_region = df_nuc.groupby(color_by)["qc_flag_any"].mean().mul(100)
        summary["pct_flagged"] = summary[color_by].map(flagged_by_region).fillna(0.0)

        flag_count_cols = flag_cols + ["qc_flag_any"]
        flag_by_region = df_nuc.groupby(color_by)[flag_count_cols].sum()
        flag_by_region.columns = [
            f"flagged: {flag_labels.get(c, c)}" for c in flag_count_cols
        ]
        summary = summary.merge(flag_by_region.reset_index(), on=color_by, how="left")
        summary[list(flag_by_region.columns)] = (
            summary[list(flag_by_region.columns)].fillna(0).astype(int)
        )

    qc_dashboard_fig = None
    if qc_dashboard:
        thresholds = dict(
            area_low_global=size_thres,
            area_high_global=size_thres_high,
            nuc_low_global=nuc_thres,
            nuc_high_global=nuc_thres_high,
            area_scope="global",
            nuc_scope="global",
            area_flag_low_global=size_flag_thres,
            area_flag_high_global=size_upper_flag_thres,
            nuc_flag_low_global=nuc_flag_thres,
        )
        qc_dashboard_fig = plot_filter_dashboard(
            df_flagged,
            summary,
            thresholds,
            area_col=cell_size,
            nuc_col=nuc_marker,
            region_col=color_by,
            flag_cols=flag_cols,
            style=style,
        )

    # printed text summary -- whenever the dashboard was requested, or there are flags to
    # report on (matches the pre-flag behaviour of always printing flag rates regardless of
    # `plot`/`qc_dashboard`, just folded into one block instead of a separate print loop)
    if qc_dashboard or flag_cols:
        n_total = len(df)
        n_kept = len(df_nuc)
        removed = df_flagged[~df_flagged["keep"]]
        print("\nFilter summary")
        print("─" * 46)
        print(f"  Total cells          : {n_total:>6,}")
        print(
            f"  Kept                 : {n_kept:>6,}  ({n_kept / n_total * 100:.1f} %)"
        )
        for r, cnt in removed["removal_reason"].value_counts().items():
            print(f"  Removed ({r:<15}): {cnt:>6,}  ({cnt / n_total * 100:.1f} %)")
        if flag_cols:
            print("─" * 46)
            for col in flag_cols + ["qc_flag_any"]:
                cnt = int(df_nuc[col].sum())
                label = flag_labels.get(col, col)
                print(
                    f"  Flagged ({label:<12}): {cnt:>6,}  ({df_nuc[col].mean() * 100:.1f} % of kept)"
                )
        print("─" * 46)

    results = [df_nuc]
    if qc_dashboard:
        results.append(qc_dashboard_fig)
    if return_summary:
        results.append(summary)
    return results[0] if len(results) == 1 else tuple(results)


def format(
    data,
    list_out=None,
    list_keep=None,
    method="zscore",
    ArcSin_cofactor=150,
    markers=None,
):
    """
    This function formats the data based on the specified method. It supports four methods: "zscore", "double_zscore", "MinMax", and "ArcSin".

    Normalization only ever applies to a subset of columns -- the rest (coordinates, IDs,
    region/condition labels, or any marker you don't want normalized) must pass through
    untouched. There are two ways to specify that subset:

    - ``markers`` (recommended): directly list the columns to normalize. Every other
      column (after ``list_out`` drops) passes through untouched automatically.
    - ``list_out`` + ``list_keep`` (legacy): ``list_out`` columns are dropped entirely;
      ``list_keep`` columns pass through untouched; everything *not* in either list is
      what gets normalized. Note ``list_keep`` is the "leave alone" bucket, not the
      "normalize these" bucket, despite the name -- it's the opposite of ``markers``.

    ``markers`` and ``list_keep`` are mutually exclusive (``markers`` already implies
    everything else is kept untouched, so combining the two is ambiguous); ``list_out``
    can still be combined with either, e.g. to drop leftover columns before normalizing.

    Parameters
    ----------
    data : DataFrame
        The input data to be formatted.
    list_out : list, optional
        Columns to drop entirely from the output. Combinable with ``markers``. Default
        is None (nothing dropped).
    list_keep : list, optional
        Columns to pass through to the output unchanged, excluded from normalization
        (e.g. ``x``, ``y``, region/condition labels, or a marker like DAPI that
        shouldn't be normalized). Mutually exclusive with ``markers``. Default is None
        (treated as empty -- only valid together with ``markers``, or if you want to
        normalize every remaining column).
    method : str, optional
        The method to be used for normalizing the data. It can be "zscore", "double_zscore", "MinMax", or "ArcSin". By default, it is "zscore".
    ArcSin_cofactor : int, optional
        The cofactor to be used in the ArcSin transformation. By default, it is 150.
    markers : list, optional
        Columns to normalize, specified directly (e.g. your existing clustering marker
        list). Every column not in ``markers`` (and not dropped via ``list_out``) passes
        through unchanged. Default is None (use the legacy ``list_out``/``list_keep``
        complement instead).

    Returns
    -------
    DataFrame
        The formatted data.

    Raises
    ------
    ValueError
        If the specified method is not supported, or if both ``markers`` and
        ``list_keep`` are given.
    """

    list = ["zscore", "double_zscore", "MinMax", "ArcSin"]

    if method not in list:
        raise ValueError(f"method={method!r} is not supported; choose from {list}.")

    if markers is not None and list_keep is not None:
        raise ValueError(
            "'markers' and 'list_keep' are mutually exclusive -- 'markers' already "
            "implies every other column is kept untouched. Pass one or the other."
        )

    # Drop column list (note: `list` the builtin is shadowed by the method-names list
    # above, so this function avoids calling list(...) anywhere below)
    list1 = [col for col in data.columns if "blank" in col]
    list_out1 = list1 + (list_out or [])

    # Drop columns not interested in
    dfin = data.drop(list_out1, axis=1)

    if markers is not None:
        # dataframe for normalization
        df_markers = dfin.loc[:, markers]
        # save columns for later
        df_loc = dfin.drop(columns=markers)
    else:
        # save columns for later
        df_loc = dfin.loc[:, (list_keep or [])]
        # dataframe for normalization
        df_markers = dfin.drop((list_keep or []), axis=1)

    ##ArcSin transformation
    if method == "ArcSin":
        # parameters seit in function
        # Only decrease the background if the median is higher than the background
        dfa = df_markers.apply(lambda x: np.arcsinh(x / ArcSin_cofactor))

        # Add back labels for normalization type
        dfz_all = pd.concat([dfa, df_loc], axis=1, join="inner")

        return dfz_all

    ##Double Z normalization
    elif method == "double_zscore":
        # zscore of the column markers
        dfz1 = pd.DataFrame(
            zscore(df_markers, 0),
            index=df_markers.index,
            columns=[i for i in df_markers.columns],
        )

        # zscore rows
        dfz2 = pd.DataFrame(
            zscore(dfz1, 1), index=dfz1.index, columns=[i for i in dfz1.columns]
        )

        # Take cumulative density function to find probability of z score across a row
        dfz3 = pd.DataFrame(
            norm.cdf(dfz2), index=dfz2.index, columns=[i for i in dfz2.columns]
        )

        # First 1-probability and then take negative logarithm so greater values demonstrate positive cell type
        dflog = dfz3.apply(lambda x: -np.log(1 - x))

        # Add back labels for normalization type
        dfz_all = pd.concat([dflog, df_loc], axis=1, join="inner")

        # print("the number of regions = "+str(len(dfz_all.region_num.unique())))

        return dfz_all

    # Min Max normalization
    elif method == "MinMax":
        dfmm = df_markers

        for col in dfmm.columns:
            max_value = dfmm[col].quantile(0.99)
            min_value = dfmm[col].quantile(0.01)
            dfmm[col].loc[dfmm[col] > max_value] = max_value
            dfmm[col].loc[dfmm[col] < min_value] = min_value
            dfmm[col] = (dfmm[col] - min_value) / (max_value - min_value)

        # Add back labels for normalization type
        dfz_all = pd.concat([dfmm, df_loc], axis=1, join="inner")

        return dfz_all

    ## Z normalization
    else:
        # zscore of the column markers
        dfz1 = pd.DataFrame(
            zscore(df_markers, 0),
            index=df_markers.index,
            columns=[i for i in df_markers.columns],
        )

        # Add back labels for normalization type
        dfz_all = pd.concat([dfz1, df_loc], axis=1, join="inner")

        # print("the number of regions = "+str(len(dfz_all.region_num.unique())))

        return dfz_all


# Only useful for "classic CODEX" where samples are covered by multiple regions
# Could also be used for montages of multiple samples (tiles arraged in grid)
def xycorr(data, y_rows, x_columns, X_pix, Y_pix):
    """
    Corrects the x and y coordinates of the data for "classic CODEX" where samples are covered by multiple regions.
    This function could also be used for montages of multiple samples (tiles arranged in a grid).

    Parameters
    ----------
    data : DataFrame
        The input data to be corrected.
    y_rows : int
        The number of rows in the y direction.
    x_columns : int
        The number of columns in the x direction.
    X_pix : int
        The number of pixels in the x direction.
    Y_pix : int
        The number of pixels in the y direction.

    Returns
    -------
    DataFrame
        The corrected data with added 'Xcorr' and 'Ycorr' columns representing the corrected x and y coordinates respectively.
    """
    # Make a copy for xy correction
    df_XYcorr = data.copy()
    df_XYcorr["Xcorr"] = 0
    df_XYcorr["Ycorr"] = 0
    dict_test = dict(enumerate(df_XYcorr.region_num.unique()))
    dict_map = {v: k + 1 for k, v in dict_test.items()}
    df_XYcorr["regloop"] = df_XYcorr["region_num"].map(dict_map)
    region_num = df_XYcorr.regloop.max()

    # first value of tuple is y and second is x
    d = list(product(range(0, y_rows, 1), range(0, x_columns, 1)))
    e = list(range(1, region_num + 1, 1))
    dict_corr = {}
    dict_corr = dict(zip(e, d))

    # Adding the pixels with the dictionary
    for reg_num in list(df_XYcorr["regloop"].unique()):
        df_XYcorr["Xcorr"].loc[df_XYcorr["regloop"] == reg_num] = (
            df_XYcorr["x"].loc[df_XYcorr["regloop"] == reg_num]
            + dict_corr[reg_num][1] * X_pix
        )

    for reg_num in list(df_XYcorr["regloop"].unique()):
        df_XYcorr["Ycorr"].loc[df_XYcorr["regloop"] == reg_num] = (
            df_XYcorr["y"].loc[df_XYcorr["regloop"] == reg_num]
            + dict_corr[reg_num][0] * Y_pix
        )

    df_XYcorr.drop(columns=["regloop"], inplace=True)
    return df_XYcorr


# Get rid of noisy cells from dataset
def marker_score_thres(
    dfz,
    col_num=None,
    cut_off=0.01,
    count_bin=50,
    score_sum_bin=50,
    figsize=(10, 5),
    qc_dashboard=True,
    flag_cut_off=None,
    style=None,
    positive_thres=0,
    marker_list=None,
):
    """
    Compute the noise-removal threshold(s), with an optional Count/Score-sum
    histogram figure. The default cut off is the top 1%.

    Works on the output of any ``sp.pp.format`` method, provided ``positive_thres`` is
    set appropriately for that method's output range: ``0`` (the default) for
    ``zscore`` (positive = above that marker's own population mean), or e.g. ``~0.5``
    for ``MinMax``'s ``[0, 1]`` midpoint, or a value informed by your own
    ``double_zscore``/``ArcSin`` distribution otherwise -- see ``positive_thres`` below.

    Parameters
    ----------
    dfz : DataFrame
        The input data from which the threshold is to be determined.
    col_num : int, optional
        Deprecated alias for ``marker_list``, kept only for backwards compatibility --
        raises a ``DeprecationWarning``. Selects the markers positionally (the first
        ``col_num + 1`` columns of ``dfz``) rather than by name, which silently breaks
        if ``dfz``'s column order ever changes. Mutually exclusive with ``marker_list``.
        Will be removed in a future version.
    marker_list : list of str, optional
        The marker columns to score, specified directly by name (e.g. the same list
        you passed to ``sp.pp.format``'s ``markers`` argument). Required unless the
        deprecated ``col_num`` is given instead.
    cut_off : float, optional
        The cut off percentage for the threshold. By default, it is 0.01 (1%).
    count_bin : int, optional
        The number of bins for the count histogram (ignored if
        `qc_dashboard` is False). By default, it is 50.
    score_sum_bin : int, optional
        The number of bins for the score-sum histogram (ignored if
        `qc_dashboard` is False). By default, it is 50.
    figsize : tuple, optional
        The size of the figure to be plotted (ignored if `qc_dashboard` is
        False). By default, it is (10, 5).
    qc_dashboard : bool, optional
        When True (default), also build the two-panel histogram figure and
        return it (see ``Returns``). When False, no figure is constructed at
        all -- only the thresholds are computed and returned.
    flag_cut_off : float, optional
        A second, looser cut off percentage, for comparing a stricter "drop"
        cut off against a looser "flag" cut off (e.g. for
        `sp.pp.remove_noise`'s `count_flag_thres`/`score_sum_flag_thres`
        arguments). Adds ``count_flag_cut_off``/``score_sum_flag_cut_off`` to the
        returned thresholds regardless of `qc_dashboard`; when `qc_dashboard`
        is also True, additionally drawn as an orange dashed line alongside
        the black ``cut_off`` line. ``None`` by default (no flag thresholds).
    style : SPACEcStyle, str, or pathlib.Path, optional
        Style to render with (ignored if `qc_dashboard` is False). ``None``
        uses the active style.
    positive_thres : float, optional
        Per-marker threshold above which a cell counts as "positive" for that marker
        (feeds into ``Count``, below). By default ``0``, which only means something
        meaningful for ``zscore``-normalized data (positive = above that marker's own
        population mean). For the other ``sp.pp.format`` methods, pass a threshold
        suited to that method's output range instead -- e.g. ``~0.5`` for ``MinMax``'s
        natural midpoint, or a value informed by your own ``double_zscore``/``ArcSin``
        distribution. Left at ``0`` on non-``zscore`` input, this raises (see
        ``Raises`` below) rather than silently returning a meaningless ``Count``.

    Returns
    -------
    thresholds : dict
        Keys ``count_cut_off``, ``score_sum_cut_off`` (the values at the
        ``1 - cut_off`` quantile), plus ``count_flag_cut_off``,
        ``score_sum_flag_cut_off`` if `flag_cut_off` was given.
    fig : matplotlib.figure.Figure
        Only returned if `qc_dashboard` is True -- i.e.
        ``thresholds, fig = marker_score_thres(..., qc_dashboard=True)``, or just
        ``thresholds = marker_score_thres(...)`` (the default) otherwise. Not
        shown or saved here -- that is the caller's responsibility.

    Raises
    ------
    ValueError
        If neither ``marker_list`` nor ``col_num`` is given, or both are given
        (mutually exclusive). Also raised if ``positive_thres`` is left at its default
        (``0``) but none of the values in ``marker_list`` are below it -- i.e. every
        marker would count as "positive" for every cell. This is the signature of
        non-``zscore`` input (``double_zscore``/``MinMax``/``ArcSin`` are all
        non-negative by construction), not a usable result, so it's raised rather than
        silently returned.
    """
    import warnings

    from spacec.plotting._style import _apply_spines, _apply_style

    if marker_list is not None and col_num is not None:
        raise ValueError(
            "'marker_list' and 'col_num' are mutually exclusive -- pass one or the "
            "other."
        )
    if marker_list is None:
        if col_num is None:
            raise ValueError(
                "marker_score_thres() requires either 'marker_list' or the "
                "deprecated 'col_num'."
            )
        warnings.warn(
            "'col_num' is deprecated and will be removed in a future version. "
            "Use 'marker_list' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        marker_list = list(dfz.columns[: col_num + 1])

    dfz_copy = dfz.copy()
    marker_cols = dfz.loc[:, marker_list]
    if positive_thres == 0 and not (marker_cols < 0).to_numpy().any():
        raise ValueError(
            "positive_thres=0 (the default) but none of the values in marker_list "
            "are negative -- this almost always means the input wasn't normalized "
            "with method='zscore' (the only sp.pp.format method centered around a "
            "meaningful positive/negative split at 0; 'double_zscore', 'MinMax', and "
            "'ArcSin' are all non-negative by construction, so positive_thres=0 "
            "would count every marker as 'positive' for every cell). Pass an "
            "explicit positive_thres suited to your normalization method (e.g. ~0.5 "
            "for MinMax's natural midpoint) to proceed."
        )
    dfz_copy["Count"] = marker_cols.ge(positive_thres).sum(axis=1)
    dfz_copy["score_sum"] = marker_cols.sum(axis=1)
    count_cutoff = dfz_copy["Count"].quantile(1 - cut_off)
    score_sum_cutoff = dfz_copy["score_sum"].quantile(1 - cut_off)
    count_flag_cutoff = (
        dfz_copy["Count"].quantile(1 - flag_cut_off)
        if flag_cut_off is not None
        else None
    )
    score_sum_flag_cutoff = (
        dfz_copy["score_sum"].quantile(1 - flag_cut_off)
        if flag_cut_off is not None
        else None
    )

    thresholds = {"count_cut_off": count_cutoff, "score_sum_cut_off": score_sum_cutoff}
    if flag_cut_off is not None:
        thresholds["count_flag_cut_off"] = count_flag_cutoff
        thresholds["score_sum_flag_cut_off"] = score_sum_flag_cutoff

    if not qc_dashboard:
        return thresholds

    _style = _apply_style(style)
    fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=figsize)
    axes[0].hist(dfz_copy["Count"], bins=count_bin)
    axes[0].set_title("Count", fontsize=_style.font_size_medium)
    axes[0].axvline(
        count_cutoff,
        color="k",
        linestyle="dashed",
        linewidth=1,
    )
    axes[0].text(
        0.75,
        0.75,
        "Cut off: {:.2f}".format(count_cutoff),
        ha="right",
        va="bottom",
        transform=axes[0].transAxes,
        fontsize=_style.font_size_annotation,
    )
    axes[1].hist(dfz_copy["score_sum"], bins=score_sum_bin)
    axes[1].title.set_text("Score sum")
    axes[1].title.set_fontsize(_style.font_size_medium)
    axes[1].axvline(
        score_sum_cutoff,
        color="k",
        linestyle="dashed",
        linewidth=1,
    )
    axes[1].text(
        0.75,
        0.75,
        "Cut off: {:.2f}".format(score_sum_cutoff),
        ha="right",
        va="bottom",
        transform=axes[1].transAxes,
        fontsize=_style.font_size_annotation,
    )
    if flag_cut_off is not None:
        axes[0].axvline(
            count_flag_cutoff, color="orange", linestyle="dashed", linewidth=1
        )
        axes[0].text(
            0.75,
            0.65,
            "Flag cut off: {:.2f}".format(count_flag_cutoff),
            ha="right",
            va="bottom",
            color="orange",
            transform=axes[0].transAxes,
            fontsize=_style.font_size_annotation,
        )
        axes[1].axvline(
            score_sum_flag_cutoff, color="orange", linestyle="dashed", linewidth=1
        )
        axes[1].text(
            0.75,
            0.65,
            "Flag cut off: {:.2f}".format(score_sum_flag_cutoff),
            ha="right",
            va="bottom",
            color="orange",
            transform=axes[1].transAxes,
            fontsize=_style.font_size_annotation,
        )
    for ax in axes:
        _apply_spines(ax, _style)
    return thresholds, fig


def zcount_thres(
    dfz,
    col_num,
    cut_off=0.01,
    count_bin=50,
    zsum_bin=50,
    figsize=(10, 5),
    qc_dashboard=True,
    flag_cut_off=None,
    style=None,
    positive_thres=0,
):
    """
    Deprecated alias for :func:`marker_score_thres`, kept only for backwards
    compatibility -- raises a ``DeprecationWarning``. Will be removed in a future
    version.

    Renamed because ``z_sum``/``zsum`` implied this only works on z-scored data; it now
    works on any ``sp.pp.format`` method via ``positive_thres`` (see
    :func:`marker_score_thres`). Same parameters under their old names (``zsum_bin``
    here is ``score_sum_bin`` there), and the same return shape, with
    ``score_sum_cut_off``/``score_sum_flag_cut_off`` translated back to
    ``zsum_cut_off``/``zsum_flag_cut_off``. ``col_num`` here is also forwarded as the
    (separately deprecated) ``col_num`` of :func:`marker_score_thres`, so this raises
    two ``DeprecationWarning``\\ s -- one for the function, one for the parameter.
    """
    import warnings

    warnings.warn(
        "'zcount_thres' is deprecated and will be removed in a future version. "
        "Use 'marker_score_thres' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = marker_score_thres(
        dfz,
        col_num=col_num,
        cut_off=cut_off,
        count_bin=count_bin,
        score_sum_bin=zsum_bin,
        figsize=figsize,
        qc_dashboard=qc_dashboard,
        flag_cut_off=flag_cut_off,
        style=style,
        positive_thres=positive_thres,
    )
    thresholds = result[0] if qc_dashboard else result
    if "score_sum_cut_off" in thresholds:
        thresholds["zsum_cut_off"] = thresholds.pop("score_sum_cut_off")
    if "score_sum_flag_cut_off" in thresholds:
        thresholds["zsum_flag_cut_off"] = thresholds.pop("score_sum_flag_cut_off")
    return (thresholds, result[1]) if qc_dashboard else thresholds


def remove_noise(
    df,
    col_num=None,
    score_sum_thres=None,
    count_thres=None,
    score_sum_flag_thres=None,
    count_flag_thres=None,
    positive_thres=0,
    z_sum_thres=None,
    z_count_thres=None,
    z_sum_flag_thres=None,
    z_count_flag_thres=None,
    marker_list=None,
):
    """
    Removes noisy cells from the dataset based on the given thresholds.

    ``positive_thres`` must match whatever was passed to the ``sp.pp.marker_score_thres``
    call that produced ``score_sum_thres``/``count_thres`` -- see its docstring for why
    the default (``0``) only makes sense for ``zscore``-normalized input.

    Parameters
    ----------
    df : DataFrame
        The input data from which noisy cells are to be removed.
    col_num : int, optional
        Deprecated alias for ``marker_list``, kept only for backwards compatibility --
        raises a ``DeprecationWarning``. Selects the markers positionally (the first
        ``col_num + 1`` columns of ``df``) rather than by name. Mutually exclusive with
        ``marker_list``. Will be removed in a future version.
    marker_list : list of str, optional
        The marker columns to score, specified directly by name. Must be the same list
        (or the equivalent ``col_num``) used in the ``marker_score_thres`` call that
        produced ``score_sum_thres``/``count_thres``. Required unless the deprecated
        ``col_num`` is given instead.
    score_sum_thres : float
        The threshold for the sum of scores. Cells with a score sum greater than this
        threshold are considered noisy. Required (directly, or via the deprecated
        ``z_sum_thres`` alias).
    count_thres : int
        The threshold for the count of positive markers. Cells with a count greater
        than this threshold are considered noisy. Required (directly, or via the
        deprecated ``z_count_thres`` alias).
    score_sum_flag_thres : float, optional
        A looser threshold for the sum of scores. Cells that pass the drop
        threshold above but still exceed this value are flagged (not
        dropped) via a new ``qc_flag_noise`` column. ``None`` by default,
        which adds no flag column and leaves the returned columns unchanged.
    count_flag_thres : int, optional
        A looser threshold for the count of positive markers, flagged the same way
        as ``score_sum_flag_thres``. ``None`` by default.
    positive_thres : float, optional
        Per-marker "positive" threshold, by default ``0``. Must be the same value used
        in the ``marker_score_thres`` call that produced ``score_sum_thres``/
        ``count_thres``; see :func:`marker_score_thres` for details.
    z_sum_thres : float, optional
        Deprecated alias for ``score_sum_thres``, kept only for backwards compatibility
        -- raises a ``DeprecationWarning`` and overrides ``score_sum_thres`` if given.
        Will be removed in a future version.
    z_count_thres : int, optional
        Deprecated alias for ``count_thres``, kept only for backwards compatibility --
        raises a ``DeprecationWarning`` and overrides ``count_thres`` if given. Will be
        removed in a future version.
    z_sum_flag_thres : float, optional
        Deprecated alias for ``score_sum_flag_thres``. Same treatment as
        ``z_sum_thres`` above.
    z_count_flag_thres : int, optional
        Deprecated alias for ``count_flag_thres``. Same treatment as ``z_sum_thres``
        above.

    Returns
    -------
    df_want : DataFrame
        The cleaned data with noisy cells removed. Carries a new
        ``qc_flag_noise`` boolean column if either ``*_flag_thres`` argument
        is given.
    cc : DataFrame
        The data of the noisy cells that were removed from the original data.

    Raises
    ------
    ValueError
        If neither ``marker_list`` nor ``col_num`` is given, or both are given.
        Also raised if ``positive_thres`` is left at its default (``0``) but none of
        the values in ``marker_list`` are below it -- see :func:`marker_score_thres`.
    TypeError
        If neither ``score_sum_thres``/``z_sum_thres`` nor ``count_thres``/
        ``z_count_thres`` is given.
    """
    import warnings

    if marker_list is not None and col_num is not None:
        raise ValueError(
            "'marker_list' and 'col_num' are mutually exclusive -- pass one or the "
            "other."
        )
    if marker_list is None:
        if col_num is None:
            raise ValueError(
                "remove_noise() requires either 'marker_list' or the deprecated "
                "'col_num'."
            )
        warnings.warn(
            "'col_num' is deprecated and will be removed in a future version. "
            "Use 'marker_list' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        marker_list = list(df.columns[: col_num + 1])

    if z_sum_thres is not None:
        warnings.warn(
            "'z_sum_thres' is deprecated and will be removed in a future version. "
            "Use 'score_sum_thres' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        score_sum_thres = z_sum_thres
    if z_count_thres is not None:
        warnings.warn(
            "'z_count_thres' is deprecated and will be removed in a future version. "
            "Use 'count_thres' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        count_thres = z_count_thres
    if z_sum_flag_thres is not None:
        warnings.warn(
            "'z_sum_flag_thres' is deprecated and will be removed in a future "
            "version. Use 'score_sum_flag_thres' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        score_sum_flag_thres = z_sum_flag_thres
    if z_count_flag_thres is not None:
        warnings.warn(
            "'z_count_flag_thres' is deprecated and will be removed in a future "
            "version. Use 'count_flag_thres' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        count_flag_thres = z_count_flag_thres

    if score_sum_thres is None or count_thres is None:
        raise TypeError(
            "remove_noise() missing required argument(s): 'score_sum_thres' and "
            "'count_thres' (or their deprecated 'z_sum_thres'/'z_count_thres' "
            "aliases) must both be given."
        )

    df_z_1_copy = df.copy()
    marker_cols = df_z_1_copy.loc[:, marker_list]
    if positive_thres == 0 and not (marker_cols < 0).to_numpy().any():
        raise ValueError(
            "positive_thres=0 (the default) but none of the values in marker_list "
            "are negative -- this almost always means the input wasn't normalized "
            "with method='zscore'. See sp.pp.marker_score_thres's docstring; pass "
            "the same explicit positive_thres used there."
        )
    df_z_1_copy["Count"] = marker_cols.ge(positive_thres).sum(axis=1)
    df_z_1_copy["score_sum"] = marker_cols.sum(axis=1)
    cc = df_z_1_copy[
        (df_z_1_copy["score_sum"] > score_sum_thres)
        | (df_z_1_copy["Count"] > count_thres)
    ]
    df_want = df_z_1_copy[
        ~(
            (df_z_1_copy["score_sum"] > score_sum_thres)
            | (df_z_1_copy["Count"] > count_thres)
        )
    ].copy()
    percent_removed = np.round(
        1 - (df_want.shape[0] / df_z_1_copy.shape[0]), decimals=3
    )
    print(str(percent_removed * 100) + "% cells are removed.")

    if score_sum_flag_thres is not None or count_flag_thres is not None:
        flagged = pd.Series(False, index=df_want.index)
        if score_sum_flag_thres is not None:
            flagged |= df_want["score_sum"] > score_sum_flag_thres
        if count_flag_thres is not None:
            flagged |= df_want["Count"] > count_flag_thres
        df_want["qc_flag_noise"] = flagged
        print(f"qc_flag_noise: {flagged.mean():.2%} of kept cells flagged")

    df_want.drop(columns=["Count", "score_sum"], inplace=True)
    df_want.reset_index(inplace=True, drop=True)
    return df_want, cc


class ImageProcessor:
    """
    A class used to process images and compute channel means and sums.

    ...

    Attributes
    ----------
    flatmasks : ndarray
        2D numpy array containing masks for each cell, relabeled to
        consecutive integers ``1..n_masks`` (background ``0``).
    labels : ndarray
        The original mask label for each row of every array returned by
        :meth:`compute_channel_means_sums_compensated`, i.e. row ``i``
        corresponds to mask label ``labels[i]``. Callers must use this to
        join results back onto a DataFrame by label rather than by
        position -- required whenever the mask has non-consecutive
        labels, a missing background value, or more labels than the
        DataFrame has rows (e.g. cells removed by a ``size_cutoff``
        during segmentation).

    Methods
    -------
    compute_channel_means_sums_compensated(channels):
        Computes the channel means and sums for each cell and compensates them.
    """

    def __init__(self, flatmasks):
        """
        Constructs all the necessary attributes for the ImageProcessor object.

        Parameters
        ----------
            flatmasks : ndarray
                2D numpy array containing masks for each cell. Labels need
                not be consecutive; background, if present, must be ``0``.
        """
        flatmasks = np.asarray(flatmasks).squeeze()
        if flatmasks.ndim != 2:
            raise ValueError(
                "masks must be 2D after squeezing singleton dimensions; "
                f"got shape {flatmasks.shape}"
            )
        # Label discovery via bincount rather than np.unique: masks are
        # non-negative integers, so counting occurrences and taking the
        # non-zero bins is ~18x faster than sorting the whole array (2.3s
        # -> 0.13s on a 6000x6000 mask). Falls back to np.unique when the
        # label space is sparse enough that the count array would be the
        # more expensive allocation.
        max_label = int(flatmasks.max()) if flatmasks.size else 0
        if (
            np.issubdtype(flatmasks.dtype, np.integer)
            and flatmasks.min() >= 0
            and max_label <= flatmasks.size
        ):
            counts = np.bincount(flatmasks.ravel(), minlength=max_label + 1)
            labels = np.flatnonzero(counts)
        else:
            labels = np.unique(flatmasks)
        self.labels = labels[labels != 0]
        lut = np.zeros(max_label + 1, dtype=np.int32)
        lut[self.labels] = np.arange(1, len(self.labels) + 1, dtype=np.int32)
        # Relabeled mask: consecutive 1..n_masks, background 0. Row i of
        # every array returned below now corresponds to self.labels[i],
        # regardless of what the original mask labels were.
        self.flatmasks = lut[flatmasks]

    def _build_adjacency(self):
        """
        Build the sparse cell-adjacency matrix used for compensation.

        Reproduces the matrix defined by CellSeg's ``update_adjacency_matrix``
        exactly -- 4-connected neighbours, background counting as a border but
        not as a neighbour, image edges excluded -- but assembles it as a
        sparse matrix rather than a dense one. The matrix has only ~7
        non-zeros per row (a cell touches a handful of neighbours), so the
        dense form is almost entirely zeros and costs O(n_masks^2) memory:
        57 GB at 120,000 cells, which is simply unallocatable.

        Returns
        -------
        scipy.sparse.csc_matrix
            Square ``(n_masks, n_masks)`` matrix ``A`` with ``A[i, i] == 1``
            and ``A[i, j]`` the fraction of cell ``i``'s border shared with
            cell ``j``, halved -- the coefficients of the spillover model.
        """
        M = self.flatmasks
        n_masks = len(self.labels)

        rows, cols = [], []
        # Left, right, up, and down neighbours. Each pair is
        # (origin, neighbour) over the overlapping slice, which reproduces
        # CellSeg's per-pixel bounds checks: a pixel in column 0 has no left
        # neighbour, a pixel in row 0 has no upward neighbour, and so on.
        for orig, neighbor in (
            (M[:, 1:], M[:, :-1]),
            (M[:, :-1], M[:, 1:]),
            (M[1:, :], M[:-1, :]),
            (M[:-1, :], M[1:, :]),
        ):
            cond = (orig != 0) & (neighbor != 0) & (orig != neighbor)
            rows.append(orig[cond] - 1)
            cols.append(neighbor[cond] - 1)

        # Diagonal: one count per pixel that borders anything other than its
        # own cell -- including background, which is a border but not a
        # neighbour. This is the cell's border-pixel count, the denominator
        # of the normalization below.
        has_border = np.zeros_like(M, dtype=bool)
        has_border[:, 1:] |= (M[:, 1:] != 0) & (M[:, 1:] != M[:, :-1])
        has_border[:, :-1] |= (M[:, :-1] != 0) & (M[:, :-1] != M[:, 1:])
        has_border[1:, :] |= (M[1:, :] != 0) & (M[1:, :] != M[:-1, :])
        has_border[:-1, :] |= (M[:-1, :] != 0) & (M[:-1, :] != M[1:, :])
        diag_ids = M[has_border] - 1  # has_border implies M != 0
        rows.append(diag_ids)
        cols.append(diag_ids)

        data = [np.ones(len(r), dtype=np.float64) for r in rows]
        # Explicit zero on every diagonal position, so that a cell with no
        # border pixels at all still has a diagonal slot in the sparsity
        # pattern and setting the diagonal below is not a structural change.
        rows.append(np.arange(n_masks))
        cols.append(np.arange(n_masks))
        data.append(np.zeros(n_masks, dtype=np.float64))

        # coo_matrix sums duplicate (row, col) entries when converted, which
        # is the accumulation np.add.at was doing -- but in C, and without a
        # random-access scatter into a multi-gigabyte array.
        adj = sparse.coo_matrix(
            (
                np.concatenate(data),
                (
                    np.concatenate(rows).astype(np.int64, copy=False),
                    np.concatenate(cols).astype(np.int64, copy=False),
                ),
            ),
            shape=(n_masks, n_masks),
        ).tocsr()

        # Normalize each row by twice its border-pixel count, then set the
        # diagonal to 1 -- identical to CellSeg's per-row loop. Scaling
        # adj.data in place keeps the sparsity pattern untouched.
        scale = 1.0 / (np.maximum(adj.diagonal(), 1) * 2)
        adj.data *= np.repeat(scale, np.diff(adj.indptr))
        adj.setdiag(1.0)
        return adj.tocsc()

    def compute_channel_means_sums_compensated(self, channels):
        """
        Compute per-cell channel means and compensate them for signal
        bleeding between adjacent cells.

        Implements the compensation of Lee et al., *CellSeg* (BMC
        Bioinformatics 23:46, 2022), which in turn follows REDSEA (Bai et
        al., Front. Immunol. 12:652631, 2021). For each channel it solves

            A x = m

        where ``m[i]`` is cell ``i``'s raw mean intensity, ``A[i, i] = 1``,
        and ``A[i, j]`` is the fraction of cell ``i``'s border shared with
        cell ``j`` (halved). Each row states that a cell's measured mean is
        its own signal plus a share of each neighbour's signal proportional
        to their shared boundary. ``x`` is the compensated mean, clipped at
        zero.

        The model is unchanged from CellSeg; only the numerics differ. ``A``
        is square, so its least-squares solution is its exact solution, and
        it is solved directly in float64 as a sparse system instead of via a
        dense O(n_masks^3) least-squares factorization.

        Parameters
        ----------
        channels : sequence of np.ndarray
            One 2D array per channel, each matching the mask's shape. Passed
            as separate arrays rather than a stacked ``(H, W, C)`` volume so
            that no copy of the image is ever materialized.

        Returns
        -------
        compensated_means : np.ndarray
            ``(n_masks, n_channels)`` compensated mean intensities.
        means : np.ndarray
            ``(n_masks, n_channels)`` raw (uncompensated) mean intensities.
        channel_counts : np.ndarray
            ``(n_masks,)`` pixel count of each cell mask.

        Notes
        -----
        Row ``i`` of every returned array corresponds to mask label
        ``self.labels[i]``, which is not in general ``i + 1``. Callers must
        join results back by label, not by position.

        See Also
        --------
        scipy.sparse.linalg.spsolve : the sparse direct solver used here.
        scipy.linalg.lstsq : CellSeg's original solver, used as a fallback
            if the sparse factorization reports a singular matrix.

        Examples
        --------
        >>> processor = ImageProcessor(masks)
        >>> channels = [np.random.rand(100, 100).astype(np.float32) for _ in range(3)]
        >>> comp_means, orig_means, counts = processor.compute_channel_means_sums_compensated(channels)
        """
        channels = list(channels)
        n_channels = len(channels)
        M = self.flatmasks
        for i, channel in enumerate(channels):
            if channel.shape != M.shape:
                raise ValueError(
                    f"channel {i} has shape {channel.shape}, which does not "
                    f"match mask shape {M.shape}"
                )

        n_masks = len(self.labels)
        if n_masks == 0:
            zeros = np.zeros((n_masks, n_channels), dtype=np.float32)
            zeros_counts = np.zeros(n_masks, dtype=np.float32)
            return zeros, zeros, zeros_counts

        # Per-cell sums, one channel at a time. Binning the full mask
        # (background lands in bin 0 and is dropped) means each channel is
        # read through a view: no boolean-indexed copy of the image, and no
        # stacked (H, W, C) volume to hold alongside it.
        flat_mask = M.ravel()
        channel_sums = np.empty((n_masks, n_channels), dtype=np.float64)
        for ch, channel in enumerate(channels):
            channel_sums[:, ch] = np.bincount(
                flat_mask, weights=channel.ravel(), minlength=n_masks + 1
            )[1:]
        counts = np.bincount(flat_mask, minlength=n_masks + 1)[1:]

        channel_counts = np.tile(counts, (n_channels, 1)).T
        means = np.true_divide(
            channel_sums,
            channel_counts,
            out=np.zeros_like(channel_sums),
            where=channel_counts != 0,
        )

        adj = self._build_adjacency()

        # A is square and, for any realistic segmentation, strongly
        # diagonally dominant (unit diagonal against off-diagonal row sums
        # averaging ~0.74), so the direct sparse solve is exact. Dominance
        # is not guaranteed for every row of a perfectly tessellated mask,
        # though, so a singular factorization falls back to the dense
        # least-squares solver CellSeg used, which returns the minimum-norm
        # solution instead of failing.
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                results = spsolve(adj, means)
            except (MatrixRankWarning, RuntimeError):
                results = None
        if results is None or not np.all(np.isfinite(results)):
            warnings.warn(
                "the cell adjacency matrix is singular; falling back to a "
                "dense least-squares solve. This needs a "
                f"{n_masks:,}x{n_masks:,} dense matrix "
                f"(~{n_masks * n_masks * 8 / 1e9:.1f} GB) and may be slow.",
                stacklevel=2,
            )
            results = lstsq(adj.toarray(), means)[0]

        # spsolve returns a 1D array when means has a single column.
        results = np.asarray(results).reshape(n_masks, n_channels)
        compensated_means = np.maximum(results, 0)

        return compensated_means, means, channel_counts[:, 0]


def compensate_cell_matrix(
    df,
    image_dict,
    masks,
    overwrite=True,
    *,
    exclude_channels=None,
    region=None,
    region_col="unique_region",
    label_col="label",
    verbose=True,
):
    """
    Compensate cell matrix by computing channel means and sums.

    Compensated values are joined back onto ``df`` by mask label (via
    ``df[label_col]``), not by row position, so this is correct even when
    ``df`` has fewer rows than the mask has labels (e.g. cells removed by a
    ``size_cutoff`` during segmentation) or when its rows are not in mask
    label order. If ``df`` has no ``label_col`` column -- only possible for
    the legacy ``seg_method="cellseg"`` path -- values are joined by row
    position instead, which requires ``len(df) == number of mask labels``
    and is only safe if row order matches mask label order exactly.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to which the compensated means will be added. Must
        contain a single region's cells: if it has a ``region_col`` column
        with more than one unique value, `region` must be given to select
        which one to compensate (mask labels are only unique within a
        region, so joining across regions is ambiguous).
    image_dict : dict
        Dictionary containing images for each channel, for the same region
        as `masks`.
    masks : ndarray
        2D array of integer mask labels for one region (background ``0``).
        Labels need not be consecutive.
    overwrite : bool, optional
        If True, overwrite existing columns in df. If False, add new
        ``<channel>_compensated`` columns instead. Default is True.
    exclude_channels : str or sequence of str, optional
        Channel(s) in `image_dict` to leave uncompensated -- typically the
        nuclei stain (e.g. ``"DAPI"``), plus any blank or autofluorescence
        channels, none of which are panel markers whose signal needs
        neighbour spillover correction. Excluded channels are not solved
        for at all: they keep their raw values in the returned DataFrame
        when ``overwrite=True``, get no ``<channel>_compensated`` column
        when ``overwrite=False``, and are not required to be present in
        `df.columns` under ``overwrite=True``. The synthetic
        ``"segmentation_channel"`` written by :func:`combine_channels` is
        always excluded regardless of this argument. Default None
        (compensate every channel).
    region : str, optional
        Which region's cells (rows of `df`) to compensate, matched against
        `region_col`. Required whenever `df` spans multiple regions.
    region_col : str, optional
        Column in `df` identifying the region each row belongs to, by
        default ``"unique_region"`` (as written by :func:`read_segdf`).
    label_col : str, optional
        Column in `df` holding the mask label each row corresponds to, by
        default ``"label"`` (as written by :func:`read_segdf`).
    verbose : bool, optional
        If True (default), print the CellSeg citation notice.

    Returns
    -------
    DataFrame
        A new DataFrame with compensated means added; `df` itself is never
        mutated.

    Raises
    ------
    ValueError
        If `masks` is not 2D after squeezing, if `masks` and any
        `image_dict` image have different shapes, if `df` spans multiple
        regions and `region` was not given, if `overwrite=True` and some
        `image_dict` channels are missing from `df.columns`, if `df`
        contains labels absent from `masks` (a sign `df` and `masks` do not
        correspond to the same region), if `exclude_channels` names a
        channel that is not in `image_dict`, or if `exclude_channels`
        excludes every channel there is to compensate.

    Notes
    -----
    Compensation runs on the CPU and needs memory proportional to the
    number of cells, not its square: the cell adjacency matrix is stored
    sparsely (~7 non-zeros per row) and solved directly. A region with
    120,000 cells takes a few seconds and tens of megabytes.

    See Also
    --------
    spacec.pp.read_segdf : adds the `label` / `region_col` columns this
        function relies on for a robust, order-independent join.
    """
    if verbose:
        print(
            "This function uses code from Lee, M.Y., Bedia, J.S., Bhate, S.S. et al. CellSeg: a robust, pre-trained nucleus segmentation and pixel quantification software for highly multiplexed fluorescence images. BMC Bioinformatics 23, 46 (2022) please consider citing the original work."
        )

    if exclude_channels is None:
        excluded = []
    elif isinstance(exclude_channels, str):
        excluded = [exclude_channels]
    else:
        excluded = list(exclude_channels)

    # A typo'd channel name would otherwise silently compensate the very
    # channel the caller meant to protect, so refuse names we can't match.
    unknown = [c for c in excluded if c not in image_dict]
    if unknown:
        raise ValueError(
            f"exclude_channels entries {unknown} are not channels in "
            f"image_dict; available channels are {list(image_dict.keys())}"
        )

    excluded_set = set(excluded)
    keys = [
        channel
        for channel in image_dict.keys()
        if channel != "segmentation_channel" and channel not in excluded_set
    ]
    if not keys:
        raise ValueError(
            "exclude_channels excludes every channel in image_dict; there is "
            "nothing left to compensate."
        )

    if overwrite:
        missing_cols = [c for c in keys if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"image_dict channels {missing_cols} are not present in "
                "df.columns; cannot overwrite columns that don't exist. "
                "Pass overwrite=False to add them as new "
                "'<channel>_compensated' columns instead."
            )

    if region is not None and region_col not in df.columns:
        raise ValueError(
            f"region={region!r} was given but df has no {region_col!r} column"
        )

    if region_col in df.columns:
        regions = df[region_col].unique()
        if region is None:
            if len(regions) > 1:
                raise ValueError(
                    f"df spans multiple regions {list(regions)}; pass "
                    f"region=<one of these> to compensate one region at a "
                    "time (mask labels are only unique within a region)."
                )
            region_df = df
        else:
            region_df = df[df[region_col] == region]
            if region_df.empty:
                raise ValueError(f"no rows in df where {region_col!r} == {region!r}")
    else:
        region_df = df

    # Passed as separate 2D arrays rather than stacked into one (H, W, C)
    # volume: the stack would be a full copy of every channel image, and
    # compensation never needs them together.
    image_list = [image_dict[channel_name] for channel_name in keys]

    processor = ImageProcessor(masks)
    (
        compensated_means,
        means,
        _channel_counts,
    ) = processor.compute_channel_means_sums_compensated(image_list)

    if label_col in region_df.columns:
        # processor.labels is sorted, so searchsorted maps each df label to
        # its row in one vectorized pass.
        mask_labels = processor.labels.astype(np.int64)
        df_labels = region_df[label_col].to_numpy().astype(np.int64)
        missing = df_labels[~np.isin(df_labels, mask_labels)]
        if missing.size:
            missing_labels = np.unique(missing)
            preview = missing_labels[:5].tolist()
            raise ValueError(
                f"df has {label_col!r} values not present in masks (e.g. "
                f"{preview}{'...' if len(missing_labels) > 5 else ''}); df "
                "and masks appear to be mismatched. Cells removed from df "
                "by a size cutoff should still have their label present in "
                "the (unfiltered) mask -- this error means the mask and df "
                "do not correspond to the same region."
            )
        row_idx = np.searchsorted(mask_labels, df_labels)
    else:
        warnings.warn(
            f"df has no {label_col!r} column; joining compensated values to "
            "df by row position instead of by mask label. This is only "
            "correct if df's rows are in the same order as the mask labels "
            "with no cells removed. Prefer sp.pp.read_segdf(...), which adds "
            "a 'label' column, for a robust label-based join.",
            stacklevel=2,
        )
        if len(region_df) != len(processor.labels):
            raise ValueError(
                f"cannot join positionally: df has {len(region_df)} rows but "
                f"masks have {len(processor.labels)} labels. Pass a df with "
                f"a {label_col!r} column (see sp.pp.read_segdf) to join by "
                "label instead, which tolerates a mismatched row count."
            )
        row_idx = np.arange(len(region_df))

    comp_vals = pd.DataFrame(
        compensated_means[row_idx], columns=keys, index=region_df.index
    )
    raw_vals = pd.DataFrame(means[row_idx], columns=keys, index=region_df.index)

    common_cols = [k for k in keys if k in region_df.columns]
    if common_cols:
        observed = region_df[common_cols].astype(float)
        diff = (observed - raw_vals[common_cols]).abs()
        tol = 1e-3 * observed.abs().clip(lower=1.0)
        if (diff.to_numpy() > tol.to_numpy()).any():
            warnings.warn(
                "recomputed raw channel means from the mask/image do not "
                "match df's existing intensity columns for the joined "
                "cells -- this usually means the mask and df are from "
                "different regions, or df's rows have been reordered or "
                "filtered since segmentation in a way not reflected by the "
                "label join. Compensated values may be wrong.",
                stacklevel=2,
            )

    df = df.copy()
    if overwrite:
        df.update(comp_vals)
    else:
        rename = {k: f"{k}_compensated" for k in keys}
        new_cols = comp_vals.rename(columns=rename)
        for col in new_cols.columns:
            if col not in df.columns:
                df[col] = np.nan
        df.update(new_cols)

    return df
