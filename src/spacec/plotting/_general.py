# load required packages
import json
import math
import os as os
import pathlib
import textwrap
import warnings

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import skimage
import tensorly as tl
from scipy import stats
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tensorly.decomposition import non_negative_tucker

from ..helperfunctions._general import *
from ._style import (
    SPACEcStyle,
    _add_scalebar,
    _apply_spines,
    _apply_style,
    _get_palette,
    _resolve_categorical_palette,
    _savefig,
    _style_colorbar,
    _style_legend,
)

# Setup — style is now controlled via SPACEcStyle; set_style() or context manager
sns.set_style("ticks")


def _graphviz_layout(G, prog="dot", seed=1):
    """Position the nodes of ``G`` with Graphviz, falling back to a spring layout.

    Goes through ``nx_agraph``/pygraphviz rather than ``nx_pydot``: pygraphviz
    2.0+ runs the layout in-process against the libgvc bundled in its wheel, so
    no system Graphviz installation or ``dot`` binary is required. ``nx_pydot``
    shells out to ``dot``, which is what used to force the conda-forge
    ``graphviz`` dependency.

    Parameters
    ----------
    G : networkx.Graph
        Graph to lay out.
    prog : str, optional
        Graphviz layout engine, e.g. ``"dot"`` or ``"neato"``. Default is ``"dot"``.
    seed : int, optional
        Seed for the ``nx.spring_layout`` fallback. Default is 1.

    Returns
    -------
    dict
        Dictionary of positions keyed by node.
    """
    try:
        return nx.nx_agraph.graphviz_layout(G, prog=prog)
    except (ImportError, FileNotFoundError, OSError):
        # pygraphviz missing, or its bundled libraries failed to load.
        return nx.spring_layout(G, seed=seed)


# plotting functions
############################################################


"""
This is a function that creates a stacked bar plot of the percentage of observations in each category in a dataset. The input data can be normalized by a grouping variable or not, and the output can be saved as a file.

data: a Pandas DataFrame containing the data to be plotted.
per_cat: a string representing the column name containing the categories to be plotted.
grouping: a string representing the column name used to group the data.
cell_list: a list of strings representing the categories to be plotted.
output_dir: a string representing the output directory to save the plot.
norm: a boolean value indicating whether to normalize the data or not (default: True).
save_name: a string representing the filename to save the plot (default: None).
col_order: a list of strings representing the order of the columns in the plot (default: None).
sub_col: a string representing the column name used to subset the data (default: None).
name_cat: a string representing the name of the category column in the plot (default: 'Cell Type').
fig_sizing: a tuple representing the size of the plot (default: (8,4)).
plot_order: a list of strings representing the order of the categories in the plot (default: None).
color_dic: a dictionary containing color codes for the categories in the plot (default: None).
remove_leg: a boolean value indicating whether to remove the legend or not (default: False).

The function returns a Pandas DataFrame and a list of strings. The DataFrame contains the data used to create the plot, and the list of strings represents the order of the categories in the plot.
"""


def pl_stacked_bar_plot(
    data,
    per_cat,
    grouping,
    cell_list,
    output_dir,
    norm=True,
    save_name=None,
    col_order=None,
    sub_col=None,
    name_cat="Cell Type",
    fig_sizing=(8, 4),
    plot_order=None,
    color_dic=None,
    remove_leg=False,
    style=None,
):
    """
    Plot a stacked bar plot based on the given data.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing the necessary information for plotting.
    per_cat : str
        The column name representing the categories.
    grouping : str
        The column name representing the grouping.
    cell_list : list
        The list of cell types to include in the plot.
    output_dir : str
        The output directory for saving the plot.
    norm : bool, optional
        Flag indicating whether to normalize the values. Defaults to True.
    save_name : str, optional
        The name to use when saving the plot. Defaults to None.
    col_order : list, optional
        The order of columns/categories for plotting. Defaults to None.
    sub_col : str, optional
        The column name representing sub-categories. Defaults to None.
    name_cat : str, optional
        The name for the category column in the plot. Defaults to 'Cell Type'.
    fig_sizing : tuple, optional
        The size of the figure (width, height) in inches. Defaults to (8, 4).
    plot_order : list, optional
        The order of categories for plotting. Defaults to None.
    color_dic : dict, optional
        A dictionary mapping categories to colors for custom colorization. Defaults to None.
    remove_leg : bool, optional
        Flag indicating whether to remove the legend. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        The pivoted data used for plotting.
    list
        The order of categories used for plotting.
    """

    _style = _apply_style(style)
    # Find Percentage of cell type
    if norm == True:
        if sub_col is None:
            test1 = data.loc[data[per_cat].isin(cell_list)]
            sub_cell_list = list(test1[per_cat].unique())
        else:
            test1 = data.loc[data[sub_col].isin(cell_list)]
            sub_cell_list = list(test1[per_cat].unique())
    else:
        if sub_col is None:
            test1 = data.copy()
            sub_cell_list = list(
                data.loc[data[per_cat].isin(cell_list)][per_cat].unique()
            )
        else:
            test1 = data.copy()
            sub_cell_list = list(
                data.loc[data[sub_col].isin(cell_list)][per_cat].unique()
            )

    test1[per_cat] = test1[per_cat].astype("category")
    test_freq = test1.groupby(grouping).apply(
        lambda x: x[per_cat].value_counts(normalize=True, sort=False) * 100
    )
    test_freq.columns = test_freq.columns.astype(str)

    test_freq.reset_index(inplace=True)
    sub_cell_list.append(grouping)
    test_freq = test_freq[sub_cell_list]
    melt_test = pd.melt(
        test_freq, id_vars=[grouping]
    )  # , value_vars=test_freq.columns)
    melt_test.rename(columns={per_cat: name_cat, "value": "percent"}, inplace=True)

    if norm == True:
        if col_order is None:
            bb = melt_test.groupby([grouping, per_cat]).sum().reset_index()
            col_order = (
                bb.loc[bb[per_cat] == bb[per_cat][0]]
                .sort_values(by="percent")[grouping]
                .to_list()
            )
    else:
        if col_order is None:
            col_order = (
                melt_test.groupby(grouping)
                .sum()
                .reset_index()
                .sort_values(by="percent")[grouping]
                .to_list()
            )

    if plot_order is None:
        plot_order = list(melt_test[per_cat].unique())

    # Set up for plotting
    melt_test_piv = pd.pivot_table(
        melt_test, columns=[name_cat], index=[grouping], values=["percent"]
    )
    melt_test_piv.columns = melt_test_piv.columns.droplevel(0)
    melt_test_piv.reset_index(inplace=True)
    melt_test_piv.set_index(grouping, inplace=True)
    melt_test_piv = melt_test_piv.reindex(col_order)
    melt_test_piv = melt_test_piv[plot_order]

    # Get color dictionary
    if color_dic is None:
        # first subplot
        ax1 = melt_test_piv.plot.bar(
            alpha=_style.data_alpha_bar,
            linewidth=_style.data_linewidth_box,
            figsize=fig_sizing,
            rot=90,
            stacked=True,
            edgecolor=_style.edge_color_default,
        )

    else:
        # first subplot
        ax1 = melt_test_piv.plot.bar(
            alpha=_style.data_alpha_bar,
            linewidth=_style.data_linewidth_box,
            color=[color_dic.get(x) for x in melt_test_piv.columns],
            figsize=fig_sizing,
            rot=90,
            stacked=True,
            edgecolor=_style.edge_color_default,
        )

    for line in ax1.lines:
        line.set_color("black")
    _apply_spines(ax1, _style)
    if remove_leg == True:
        ax1.set_ylabel("")
        ax1.set_xlabel("")
    else:
        ax1.set_ylabel("percent")
    # ax1.spines['left'].set_position(('data', 1.0))
    # ax1.set_xticks(np.arange(1,melt_test.day.max()+1,1))
    # ax1.set_ylim([0, int(ceil(max(max(melt_test_piv.sum(axis=1)), max(tm_piv.sum(axis=1)))))])
    plt.xticks(
        list(range(len(list(melt_test_piv.index)))),
        list(melt_test_piv.index),
        rotation=90,
    )
    lgd2 = ax1.legend(
        loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False
    )
    if save_name:
        _savefig(None, output_dir, save_name, "_stacked_barplot", _style)
    return melt_test_piv, plot_order


#############

"""
data: pandas DataFrame containing the data to be plotted
grouping: name of the column containing the grouping variable for the swarm boxplot
replicate: name of the column containing the replicate variable for the swarm boxplot
sub_col: name of the column containing the subsetting variable for the swarm boxplot
sub_list: list of values to subset the data by
per_cat: name of the column containing the categorical variable for the swarm boxplot
output_dir: directory where the output plot will be saved
norm: boolean (default True) to normalize data by subsetting variable before plotting
figure_sizing: tuple (default (10,5)) containing the size of the output plot
save_name: name of the file to save the output plot (if output_dir is provided)
plot_order: list of values to specify the order of the horizontal axis
col_in: list of values to subset the data by the per_cat column
color_dic: seaborn color palette for the boxplot and swarmplot
flip: boolean (default False) to flip the orientation of the plot
"""
# This function creates a box plot and swarm plot from the given data
# and returns a plot object.


def pl_swarm_box(
    data,
    grouping,
    per_cat,
    replicate,
    sub_col,
    sub_list,
    output_dir,
    norm=True,
    figure_sizing=(10, 5),
    save_name=None,
    plot_order=None,
    col_in=None,
    color_dic=None,
    flip=False,
    style=None,
):
    """
    Create a combined box plot and swarm plot for cell type percentages.

    Computes the percentage of each category in ``per_cat`` across groups
    defined by ``grouping`` and ``replicate``, then visualises the
    distributions using overlaid box and swarm plots.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing all relevant columns.
    grouping : str
        Column name used to separate the x-axis groups.
    per_cat : str
        Column whose value counts define the y-axis (percentage of each
        category).
    replicate : str
        Column name identifying biological or technical replicates.
    sub_col : str
        Column used to subset the data before computing percentages.
    sub_list : list
        Values in ``sub_col`` that are retained for the analysis.
    output_dir : str
        Directory where the plot is saved. Pass ``None`` to skip saving.
    norm : bool, optional
        If True, percentage is computed only within the subset defined by
        ``sub_list``. If False, all rows are kept and the subset is used only
        to determine which categories to display. Default is True.
    figure_sizing : tuple, optional
        Figure size as (width, height) in inches. Default is (10, 5).
    save_name : str, optional
        File-name stem for the saved plot. Required when ``output_dir`` is
        set. Default is None.
    plot_order : list, optional
        Explicit order for the x-axis groups. When None the order is
        determined by mean percentage. Default is None.
    col_in : list, optional
        Subset of categories from ``per_cat`` to display. Default is None
        (show all).
    color_dic : dict or str, optional
        Seaborn colour palette or dictionary mapping groups to colours.
        Default is None.
    flip : bool, optional
        If True, orient the plot so that ``grouping`` appears on the x-axis.
        Default is False.
    style : SPACEcStyle or None, optional
        SPACEc style configuration object. Default is None.

    Returns
    -------
    pandas.DataFrame
        Melted data frame with columns for ``grouping``, ``replicate``,
        ``per_cat``, and ``percentage``.
    """
    _style = _apply_style(style)
    # Find Percentage of cell type
    test = data.copy()
    sub_list1 = sub_list.copy()

    if norm == True:
        test1 = test.loc[test[sub_col].isin(sub_list1)]
        immune_list = list(test1[per_cat].unique())
    else:
        test1 = test.copy()
        immune_list = list(test.loc[test[sub_col].isin(sub_list1)][per_cat].unique())

    test1[per_cat] = test1[per_cat].astype("category")
    test_freq = test1.groupby([grouping, replicate]).apply(
        lambda x: x[per_cat].value_counts(normalize=True, sort=False) * 100
    )
    test_freq.columns = test_freq.columns.astype(str)
    test_freq.reset_index(inplace=True)
    immune_list.extend([grouping, replicate])
    test_freq1 = test_freq[immune_list]

    melt_per_plot = pd.melt(
        test_freq1,
        id_vars=[
            grouping,
            replicate,
        ],
    )  # ,value_vars=immune_list)
    melt_per_plot.rename(columns={"value": "percentage"}, inplace=True)

    if col_in:
        melt_per_plot = melt_per_plot.loc[melt_per_plot[per_cat].isin(col_in)]
    else:
        melt_per_plot = melt_per_plot

    if plot_order is None:
        plot_order = list(melt_per_plot[grouping].unique())

    # swarmplot to compare clustering
    plt.figure(figsize=figure_sizing)
    if flip == True:
        plt.figure(figsize=figure_sizing)
        if color_dic is None:
            ax = sns.boxplot(
                data=melt_per_plot,
                x=grouping,
                y="percentage",
                dodge=True,
                order=plot_order,
            )
            ax = sns.swarmplot(
                data=melt_per_plot,
                x=grouping,
                y="percentage",
                dodge=True,
                order=plot_order,
                edgecolor=_style.edge_color_default,
                linewidth=_style.data_linewidth_box,
                color="white",
                palette=color_dic,
            )
        else:
            ax = sns.boxplot(
                data=melt_per_plot,
                x=grouping,
                y="percentage",
                dodge=True,
                order=plot_order,
                palette=color_dic,
            )
            ax = sns.swarmplot(
                data=melt_per_plot,
                x=grouping,
                y="percentage",
                dodge=True,
                order=plot_order,
                edgecolor=_style.edge_color_default,
                linewidth=_style.data_linewidth_box,
                palette=color_dic,
            )

        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, _style.data_alpha_box))
        plt.xticks(rotation=90)
        plt.xlabel("")
        plt.ylabel("")
        plt.title(sub_list[0])
        _apply_spines(ax, _style)

    else:
        if color_dic is None:
            ax = sns.boxplot(
                data=melt_per_plot,
                x=grouping,
                y="percentage",
                dodge=True,
                order=plot_order,
            )
            ax = sns.swarmplot(
                data=melt_per_plot,
                x=grouping,
                y="percentage",
                dodge=True,
                order=plot_order,
                edgecolor=_style.edge_color_default,
                linewidth=_style.data_linewidth_box,
                color="white",
            )
        else:
            ax = sns.boxplot(
                data=melt_per_plot,
                x=grouping,
                y="percentage",
                dodge=True,
                order=plot_order,
                palette=color_dic,
            )
            ax = sns.swarmplot(
                data=melt_per_plot,
                x=grouping,
                y="percentage",
                dodge=True,
                order=plot_order,
                edgecolor=_style.edge_color_default,
                linewidth=_style.data_linewidth_box,
                palette=color_dic,
            )
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, _style.data_alpha_box))
            plt.xlabel("")
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(
            handles[: len(melt_per_plot[grouping].unique())],
            labels[: len(melt_per_plot[grouping].unique())],
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
            frameon=False,
        )
        plt.xticks(rotation=90)

        ax.set(ylim=(0, melt_per_plot["percentage"].max() + 1))
        _apply_spines(ax, _style)

    if output_dir:
        if save_name:
            _savefig(None, output_dir, save_name, "_swarm_boxplot", _style)
        else:
            print("define save_name")
    else:
        print("plot was not saved - to save the plot specify an output directory")
    return melt_per_plot


#############


def pl_Shan_div(
    tt,
    test_results,
    res,
    grouping,
    color_dic,
    sub_list,
    output_dir,
    save=False,
    plot_order=None,
    fig_size=1.5,
    style=None,
):
    """
    Plot Shannon Diversity using boxplot and swarmplot.

    Parameters
    ----------
    tt : unused
        Not used in the function.
    test_results : float
        The p-value from the statistical test.
    res : pandas.DataFrame
        The input data containing the results and grouping information.
    grouping : str
        The column name representing the grouping.
    color_dic : dict
        A dictionary mapping groups to colors for custom colorization.
    sub_list : list
        The list of sub-groups.
    output_dir : str
        The output directory for saving the plots.
    save : bool, optional
        Flag indicating whether to save the plots. Defaults to False.
    plot_order : list, optional
        The order of groups for plotting. Defaults to None.
    fig_size : float, optional
        The size of the figure. Defaults to 1.5.

    Returns
    -------
    pandas.DataFrame or bool
        The Tukey's test results if the p-value is less than 0.05, otherwise False.
    """

    _style = _apply_style(style)
    # Order by average
    if color_dic is None:
        if plot_order is None:
            plot_order = res[grouping].unique()
        else:
            plot_order = plot_order
        # Plot the swarmplot of results
        plt.figure(figsize=(fig_size, 3))

        ax = sns.boxplot(
            data=res, x=grouping, y="Shannon Diversity", dodge=True, order=plot_order
        )

        ax = sns.swarmplot(
            data=res,
            x=grouping,
            y="Shannon Diversity",
            dodge=True,
            order=plot_order,
            edgecolor=_style.edge_color_default,
            linewidth=_style.data_linewidth_box,
            color="white",
        )

    else:
        if plot_order is None:
            plot_order = res[grouping].unique()
        else:
            plot_order = plot_order
        # Plot the swarmplot of results
        plt.figure(figsize=(fig_size, 3))

        ax = sns.boxplot(
            data=res,
            x=grouping,
            y="Shannon Diversity",
            dodge=True,
            order=plot_order,
            palette=color_dic,
        )
        ax = sns.swarmplot(
            data=res,
            x=grouping,
            y="Shannon Diversity",
            dodge=True,
            order=plot_order,
            edgecolor=_style.edge_color_default,
            linewidth=_style.data_linewidth_box,
            palette=color_dic,
        )

    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, _style.data_alpha_box))
    plt.xticks(rotation=90)
    plt.xlabel("")
    plt.ylabel("Shannon Diversity")
    plt.title("")
    _apply_spines(ax, _style)
    if save == True:
        _savefig(None, output_dir, sub_list[0], "_Shannon", _style)

    plt.show()
    if test_results < 0.05:
        plt.figure(figsize=(fig_size, fig_size))
        tukey = pairwise_tukeyhsd(
            endog=res["Shannon Diversity"], groups=res[grouping], alpha=0.05
        )
        tukeydf = pd.DataFrame(
            data=tukey._results_table.data[1:], columns=tukey._results_table.data[0]
        )
        tukedf_rev = tukeydf.copy()
        tukedf_rev.rename(
            columns={"group1": "groupa", "group2": "groupb"}, inplace=True
        )
        tukedf_rev.rename(
            columns={"groupa": "group2", "groupb": "group1"}, inplace=True
        )
        tukedf_rev = tukedf_rev[tukeydf.columns]
        tukey_all = pd.concat([tukedf_rev, tukeydf])

        # Plot with tissue order preserved
        table1 = pd.pivot_table(
            tukey_all, values="p-adj", index=["group1"], columns=["group2"]
        )
        table1 = table1[plot_order]
        table1 = table1.reindex(plot_order)

        plt.figure(figsize=(5, 5))
        ax = sns.heatmap(table1, cmap="coolwarm", center=0.05, vmax=0.05)
        ax.set_title("Shannon Diversity")
        ax.set_ylabel("")
        ax.set_xlabel("")
        if save == True:
            _savefig(None, output_dir, sub_list[0], "_tukey", _style)
        plt.show()
    else:
        table1 = False


#############


def pl_cell_type_composition_vis(
    data,
    sample_column="sample",
    cell_type_column="Cell Type",
    figsize=(10, 10),
    output_dir=None,
    style=None,
):
    """
    Visualize cell type composition using stacked and unstacked bar plots.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing the sample and cell type information.
    sample_column : str, optional
        The column name representing the sample. Defaults to "sample".
    cell_type_column : str, optional
        The column name representing the cell type. Defaults to "Cell Type".
    figsize : tuple, optional
        The size of the figure (width, height) in inches. Defaults to (10, 10).
    output_dir : str, optional
        The output directory for saving the plots. Defaults to None.

    Returns
    -------
    None
    """

    _style = _apply_style(style)
    if output_dir == None:
        print("You have defined no output directory!")

    # plotting option1
    # pd.crosstab(df['sample'], df['final_cell_types']).plot(kind='barh', stacked=True,figsize = (10,12))
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # plt.show()

    # plotting option2
    ax = pd.crosstab(data[sample_column], data[cell_type_column]).plot(
        kind="barh", stacked=True, figsize=figsize
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig = ax.get_figure()
    ax.set(xlabel="count")
    _savefig(fig, output_dir, "cell_types_composition_hstack", "", _style)

    # plotting option1
    # pd.crosstab(df['sample'], df['final_cell_types']).plot(kind='barh', figsize = (10,10))
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # plt.show()

    # plotting option2
    ax = pd.crosstab(data[sample_column], data[cell_type_column]).plot(
        kind="barh", stacked=False, figsize=figsize
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig = ax.get_figure()
    ax.set(xlabel="count")
    _savefig(fig, output_dir, "cell_types_composition_hUNstack", "", _style)

    # Cell type percentage
    st = pd.crosstab(data[sample_column], data[cell_type_column])
    df_perc = (st / np.sum(st, axis=1)[:, None]) * 100
    df_perc
    # df_perc['sample'] = df_perc.index
    # df_perc

    tmp = st.T.apply(lambda x: 100 * x / x.sum())

    ax = tmp.T.plot(kind="barh", stacked=True, figsize=figsize)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig = ax.get_figure()
    ax.set(xlabel="percentage")
    _savefig(fig, output_dir, "cell_types_composition_perc_hstack", "", _style)


##############


def pl_regions_per_sample(data, sample_col, region_col, bar_color="grey"):
    """
    Plot the number of unique regions per sample as a bar chart.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing sample and region columns.
    sample_col : str
        Column name identifying each sample (x-axis).
    region_col : str
        Column name identifying regions; the number of unique values per
        sample is plotted on the y-axis.
    bar_color : str, optional
        Colour of the bars. Default is ``"grey"``.

    Returns
    -------
    None
    """
    # Group the dataframe by the specified sample column and count the unique regions
    region_counts = data.groupby(sample_col)[region_col].nunique()

    # Create a bar chart with the specified color
    plt.bar(region_counts.index, region_counts.values, color=bar_color)

    # Set chart title and axis labels
    plt.title("Count of Unique Regions per Sample")
    plt.xlabel("Samples")
    plt.ylabel("Unique Regions")

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)

    # Display the chart
    plt.show()


##############


def pl_neighborhood_analysis_2(
    data,
    k_centroids,
    values,
    sum_cols,
    X="x",
    Y="y",
    reg="unique_region",
    output_dir=None,
    k=35,
    plot_specific_neighborhoods=None,
    size=3,
    axis="on",
    ticks_fontsize=15,
    show_spatial_plots=True,
    palette="tab20",
    style=None,
):
    """
    Perform neighborhood analysis and visualize results.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing the neighborhood information.
    k_centroids : numpy.ndarray
        The centroids of the neighborhoods.
    values : numpy.ndarray
        The values associated with each cell.
    sum_cols : list
        The column names to sum.
    X : str, optional
        The column name representing the x-coordinate. Defaults to 'x'.
    Y : str, optional
        The column name representing the y-coordinate. Defaults to 'y'.
    reg : str, optional
        The column name representing the region. Defaults to 'unique_region'.
    output_dir : str, optional
        The output directory for saving the plots. Defaults to None.
    k : int, optional
        The number of neighborhoods. Defaults to 35.
    plot_specific_neighborhoods : bool or int, optional
        Flag indicating whether to plot specific neighborhoods or not. If True, all neighborhoods will be plotted.
        If an integer, only the specified neighborhood will be plotted. Defaults to None.

    Returns
    -------
    None
    """

    _style = _apply_style(style)
    if show_spatial_plots == True:
        # modify figure size aesthetics for each neighborhood
        figs = pl_catplot(
            data,
            X=X,
            Y=Y,
            exp=reg,
            hue="neighborhood" + str(k),
            invert_y=True,
            size=size,
            axis=axis,
            ticks_fontsize=ticks_fontsize,
            palette=palette,
        )

        # Save Plots for Publication
        for n, f in enumerate(figs):
            _savefig(
                f, output_dir, "neighborhood_" + str(k) + "_id" + str(n), "", _style
            )

    # this plot shows the types of cells (ClusterIDs) in the different niches (0-9)
    k_to_plot = k
    niche_clusters = k_centroids[k_to_plot]
    tissue_avgs = values.mean(axis=0)
    fc = np.log2(
        (
            (niche_clusters + tissue_avgs)
            / (niche_clusters + tissue_avgs).sum(axis=1, keepdims=True)
        )
        / tissue_avgs
    )
    fc = pd.DataFrame(fc, columns=sum_cols)
    s = sns.clustermap(
        fc,
        vmin=_style.heatmap_vmin,
        vmax=_style.heatmap_vmax,
        cmap=_style.diverging_cmap,
        row_colors=sns.color_palette(palette, len(fc)),
    )
    _savefig(
        s,
        output_dir,
        "celltypes_perniche__" + str(k),
        "",
        _style,
        dpi=_style.figure_dpi_lineart,
    )

    if plot_specific_neighborhoods is True:
        # this plot shows the types of cells (ClusterIDs) in the different niches (0-9)
        k_to_plot = k
        niche_clusters = k_centroids[k_to_plot]
        tissue_avgs = values.mean(axis=0)
        fc = np.log2(
            (
                (niche_clusters + tissue_avgs)
                / (niche_clusters + tissue_avgs).sum(axis=1, keepdims=True)
            )
            / tissue_avgs
        )
        fc = pd.DataFrame(fc, columns=sum_cols)
        s = sns.clustermap(
            fc.iloc[plot_specific_neighborhoods, :],
            vmin=_style.heatmap_vmin,
            vmax=_style.heatmap_vmax,
            cmap=_style.diverging_cmap,
        )
        _savefig(
            s,
            output_dir,
            "celltypes_perniche__" + str(k) + "_specific",
            "",
            _style,
            dpi=_style.figure_dpi_lineart,
        )


##############


def pl_highlighted_dot(
    df,
    x_col,
    y_col,
    group_col,
    highlight_group,
    highlight_color="red",
    region_col="unique_region",
    subset_col=None,
    subset_list=None,
    style=None,
):
    """
    Plots an XY dot plot colored by a grouping column for each unique region.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    x_col : str
        Name of the column to be plotted on the x-axis.
    y_col : str
        Name of the column to be plotted on the y-axis.
    group_col : str
        Name of the column used for grouping and coloring the dots.
    highlight_group : object
        Value of the group to be highlighted.
    highlight_color : str, optional
        Color of the dots for the highlighted group (default: "red").
    region_col : str, optional
        Name of the column with information about the unique regions (default: "unique_region").
    subset_col : str, optional
        Name of the column to subset the data (default: None).
    subset_list : list, optional
        List of values to subset the data from the subset column (default: None).

    Returns
    -------
    None
    """

    _style = _apply_style(style)
    # Create a colormap dictionary for coloring dots
    colormap = {highlight_group: highlight_color, "default": "grey"}

    # Subset the data based on the subset column and list
    if subset_col and subset_list:
        df = df[df[subset_col].isin(subset_list)]

    unique_regions = df[region_col].unique()

    # Determine the number of plots and the grid layout
    num_plots = len(unique_regions)

    if len(unique_regions) > 3:
        num_cols = len(unique_regions) // 2  # Number of columns in the grid
    else:
        num_cols = 2

    num_rows = (num_plots - 1) // num_cols + 1  # Number of rows in the grid

    # Create the figure and subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # Iterate over unique regions and corresponding subplots
    for region, ax in zip(unique_regions, axs.flatten()):
        # Filter the dataframe for the current region
        region_df = df[df[region_col] == region]

        # Iterate over unique groups in the current region
        for group in region_df[group_col].unique():
            # Get x and y values for the group
            x = region_df.loc[region_df[group_col] == group, x_col]
            y = region_df.loc[region_df[group_col] == group, y_col]

            # Get the color for the group
            color = colormap.get(group, colormap["default"])

            # Set the alpha value: highlighted group is fully opaque, background is muted
            alpha = (
                _style.data_alpha_spatial
                if group == highlight_group
                else _style.data_alpha_background
            )

            # Plot the dots
            ax.scatter(
                x,
                y,
                color=color,
                label=group,
                alpha=alpha,
                zorder=2,
                s=_style.scatter_size_cell,
            )

        # Remove legend
        _apply_spines(ax, _style)
        ax.legend().remove()

        # Set axis labels
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        # Set the title to the current region
        ax.set_title(f"Region: {region}")

    # Show the plot
    plt.tight_layout()
    plt.show()


##############


def pl_create_pie_charts(
    data,
    group_column,
    count_column,
    plot_order=None,
    show_percentages=True,
    color_dict=None,
):
    """
    Create pie charts for each group based on a grouping column, showing the percentage of total rows based on a
    count column.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        group_column (str): The column name for grouping the data.
        count_column (str): The column name used for counting occurrences.
        plot_order (list, optional): The order of groups for plotting. Defaults to None.
        show_percentages (bool, optional): Whether to show the percentage numbers on the pie charts. Defaults to True.
        color_dict (dict, optional): A dictionary to manually set colors for neighborhoods. Defaults to None.

    Returns:
        None
    """
    # Group the data by the grouping column
    grouped_data = data.groupby(group_column)

    # Sort the groups based on the plot_order if provided
    if plot_order:
        grouped_data = sorted(grouped_data, key=lambda x: plot_order.index(x[0]))

    # Calculate the number of rows and columns for subplots
    num_groups = len(grouped_data)
    num_cols = 3  # Number of columns for subplots
    num_rows = (num_groups - 1) // num_cols + 1

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten the subplots array

    # Create a color dictionary if not provided
    if color_dict is None:
        color_dict = {}

    # Generate a color dictionary for consistent colors if not provided
    if not color_dict:
        neighborhoods = data[count_column].unique()
        color_cycle = plt.cm.tab20.colors
        color_dict = {
            neighborhood: color_cycle[i % 20]
            for i, neighborhood in enumerate(neighborhoods)
        }

    # Iterate over each group and create a pie chart
    for i, (group, group_df) in enumerate(grouped_data):
        # Count the occurrences of each neighborhood within the group
        neighborhood_counts = group_df[count_column].value_counts()

        # Calculate the percentage of total rows for each neighborhood
        percentages = neighborhood_counts / group_df.shape[0] * 100

        # Create a color list for neighborhoods in the count column
        colors = [
            color_dict.get(neighborhood, "gray") for neighborhood in percentages.index
        ]

        if show_percentages:
            wedges, texts, autotexts = axes[i].pie(
                percentages, labels=percentages.index, autopct="%1.1f%%", colors=colors
            )
            axes[i].set_title(f"Group: {group}")
        else:
            wedges, texts = axes[i].pie(
                percentages, labels=percentages.index, colors=colors
            )
            axes[i].set_title(f"Group: {group}")

    # Remove unused subplots
    for j in range(num_groups, num_rows * num_cols):
        fig.delaxes(axes[j])

    # Adjust spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()


##############


def pl_cell_types_de(data, pvals, neigh_num, output_dir, figsize=(20, 10), style=None):
    """
    Plot cell types differential expression as a heatmap.

    Parameters
    ----------
    data : pandas.DataFrame
       The input data containing the differential expression values.
    pvals : numpy.ndarray
       The p-values associated with the differential expression values.
    neigh_num : dict
       A dictionary mapping neighborhood numbers to labels.
    output_dir : str
       The output directory for saving the plot.
    figsize : tuple, optional
       The size of the figure (width, height) in inches. Defaults to (20, 10).

    Returns
    -------
    None
    """
    _style = _apply_style(style)
    # plot as heatmap
    f, ax = plt.subplots(figsize=figsize)
    g = sns.heatmap(data, cmap="bwr", vmin=-1, vmax=1, cbar=False, ax=ax)
    for a, b in zip(*np.where(pvals < 0.05)):
        plt.text(b + 0.5, a + 0.55, "*", fontsize=20, ha="center", va="center")
    plt.tight_layout()

    inv_map = {v: k for k, v in neigh_num.items()}
    inv_map

    # plot as heatmap
    plt.style.use(["default"])
    import matplotlib as _mpl

    _mpl.rcParams.update(_apply_style(None).to_rcparams())

    data_2 = data.rename(index=inv_map)

    # Sort both axes
    sort_sum = data_2.abs().sum(axis=1).to_frame()
    sort_sum.columns = ["sum_col"]
    xx = sort_sum.sort_values(by="sum_col")
    sort_x = xx.index.values.tolist()
    sort_sum_y = data_2.abs().sum(axis=0).to_frame()
    sort_sum_y.columns = ["sum_col"]
    yy = sort_sum_y.sort_values(by="sum_col")
    sort_y = yy.index.values.tolist()
    df_sort = data_2.reindex(index=sort_x, columns=sort_y)

    f, ax = plt.subplots(figsize=figsize)
    g = sns.heatmap(df_sort, cmap="bwr", vmin=-1, vmax=1, cbar=True, ax=ax)
    for a, b in zip(*np.where(pvals < 0.05)):
        plt.text(b + 0.5, a + 0.55, "*", fontsize=20, ha="center", va="center")
    plt.tight_layout()

    _savefig(f, output_dir, "tissue_neighborhood_coeff_pvalue_bar", "", _style)

    df_sort.abs().sum()


##############


def pl_community_analysis_2(
    data,
    values,
    sum_cols,
    output_dir,
    # neighborhood_name,
    k_centroids,
    X="x",
    Y="y",
    reg="unique_region",
    save_path=None,
    k=100,
    size=3,
    axis="on",
    ticks_fontsize=15,
    plot_specific_community=None,
    show_spatial_plots=True,
    palette="tab20",
    style=None,
):
    """
    Plot community analysis.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing the community information.
    output_dir : str
        The output directory for saving the plots.
    neighborhood_name : str
        The name of the neighborhood.
    figsize : tuple, optional
        The size of the figure. Defaults to (10, 10).
    plot_specific_community : bool, optional
        Whether to plot a specific community. Defaults to None.

    Returns
    -------
    None
    """

    _style = _apply_style(style)
    output_dir2 = output_dir + "community_analysis/"
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)

    # cells = data.copy()

    # #modify figure size aesthetics for each neighborhood
    # plt.rcParams["legend.markerscale"] = 10
    # figs = pl_catplot(cells,X = X,Y=Y,exp = reg,
    #                hue = neighborhood_name,invert_y=True,size = size,figsize=8, axis=axis, ticks_fontsize=ticks_fontsize)

    # #Save Plots for Publication
    # for n,f in enumerate(figs):
    #     f.savefig(output_dir2+neighborhood_name+'_id{}.png'.format(n))

    # if plot_specific_community is True:
    #     #this plot shows the types of cells (ClusterIDs) in the different niches (0-9)
    #     k_to_plot = k
    #     niche_clusters = (k_centroids[k_to_plot])
    #     tissue_avgs = values.mean(axis = 0)
    #     fc = np.log2(((niche_clusters+tissue_avgs)/(niche_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
    #     fc = pd.DataFrame(fc,columns = sum_cols)
    #     s=sns.clustermap(fc.iloc[plot_specific_community,:], vmin =-3,vmax = 3,cmap = 'bwr',figsize=(10,5))
    #     s.savefig(output_dir2+"celltypes_perniche_"+"_"+str(k)+".png", dpi=600)

    # #this plot shows the types of cells (ClusterIDs) in the different niches (0-9)
    # k_to_plot = k
    # niche_clusters = (k_centroids[k_to_plot])
    # tissue_avgs = values.mean(axis = 0)
    # fc = np.log2(((niche_clusters+tissue_avgs)/(niche_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
    # fc = pd.DataFrame(fc,columns = sum_cols)
    # s=sns.clustermap(fc, vmin =-3,vmax = 3,cmap = 'bwr', figsize=(10,10))
    # s.savefig(output_dir2+"celltypes_perniche_"+"_"+str(k)+".png", dpi=600)

    if show_spatial_plots == True:
        # modify figure size aesthetics for each neighborhood
        figs = pl_catplot(
            data,
            X=X,
            Y=Y,
            exp=reg,
            hue="community" + str(k),
            invert_y=True,
            size=size,
            axis=axis,
            ticks_fontsize=ticks_fontsize,
            palette=palette,
        )

        # Save Plots for Publication
        for n, f in enumerate(figs):
            _savefig(f, output_dir, "community_" + str(k) + "_id" + str(n), "", _style)

    # this plot shows the types of cells (ClusterIDs) in the different niches (0-9)
    k_to_plot = k
    niche_clusters = k_centroids[k_to_plot]
    tissue_avgs = values.mean(axis=0)
    fc = np.log2(
        (
            (niche_clusters + tissue_avgs)
            / (niche_clusters + tissue_avgs).sum(axis=1, keepdims=True)
        )
        / tissue_avgs
    )
    fc = pd.DataFrame(fc, columns=sum_cols)
    s = sns.clustermap(
        fc,
        vmin=_style.heatmap_vmin,
        vmax=_style.heatmap_vmax,
        cmap=_style.diverging_cmap,
        row_colors=sns.color_palette(palette, len(fc)),
    )
    _savefig(
        s,
        output_dir,
        "celltypes_perniche__" + str(k),
        "",
        _style,
        dpi=_style.figure_dpi_lineart,
    )

    if plot_specific_community is True:
        # this plot shows the types of cells (ClusterIDs) in the different niches (0-9)
        k_to_plot = k
        niche_clusters = k_centroids[k_to_plot]
        tissue_avgs = values.mean(axis=0)
        fc = np.log2(
            (
                (niche_clusters + tissue_avgs)
                / (niche_clusters + tissue_avgs).sum(axis=1, keepdims=True)
            )
            / tissue_avgs
        )
        fc = pd.DataFrame(fc, columns=sum_cols)
        s = sns.clustermap(
            fc.iloc[plot_specific_neighborhoods, :],
            vmin=_style.heatmap_vmin,
            vmax=_style.heatmap_vmax,
            cmap=_style.diverging_cmap,
        )
        _savefig(
            s,
            output_dir,
            "celltypes_perniche__" + str(k) + "_specific",
            "",
            _style,
            dpi=_style.figure_dpi_lineart,
        )


###############
"""
This function visualizes the results of Canonical Correlation Analysis (CCA) using a graph.
The function takes in several parameters including the CCA results, the save path for the resulting plot, whether or not to save the plot, a p-value threshold, a name for the plot file, and a color palette to use for the nodes.

The function first creates an empty Petersen graph and then iterates over each pair of cell types in the CCA results.
For each pair, it calculates the observed correlation and the correlation for a set of permuted samples.
If the p-value for the observed correlation is less than the specified threshold, it adds an edge to the graph between the two cell types, weighted by the p-value.

The function then uses the graphviz_layout function to position the nodes in the graph and assigns a color to each node based on the specified color palette.
It then iterates over each edge in the graph and sets its alpha and linewidth based on the weight of the edge. Finally, it saves the resulting plot to the specified save path if save_fig is True.

Overall, this function provides a way to visually represent the relationships between cell types in the CCA results, allowing for a better understanding of the underlying patterns and correlations in the data.
"""


def pl_Visulize_CCA_results(
    CCA_results,
    output_dir,
    save_fig=False,
    p_thresh=0.1,
    save_name="CCA_vis.png",
    colors=None,
    style=None,
):
    """
    Visualize the results of Canonical Correlation Analysis (CCA) using a graph.

    Parameters
    ----------
    CCA_results : dict
        Dictionary containing the CCA results, where the keys are cell type pairs and
        the values are tuples of (observed correlation, permuted correlations).
    output_dir : str
        The output directory for saving the plot.
    save_fig : bool, optional
        Whether to save the plot. Defaults to False.
    p_thresh : float, optional
        The p-value threshold for adding edges to the graph. Defaults to 0.1.
    save_name : str, optional
        The name of the plot file. Defaults to "CCA_vis.png".
    colors : list or None, optional
        The color palette for the nodes. If None, a default palette will be used.

    Returns
    -------
    None
    """

    # Visualization of CCA
    g1 = nx.petersen_graph()
    for cn_pair, cc in CCA_results.items():
        s, t = cn_pair
        obs, perms = cc
        p = np.mean(obs > perms)
        if p > p_thresh:
            g1.add_edge(s, t, weight=p)

    _style = _apply_style(style)
    if colors != None:
        pal = colors
    else:
        pal = sns.color_palette("bright", 50)

    pos = _graphviz_layout(g1, prog="neato", seed=1)
    for k, v in pos.items():
        x, y = v
        plt.scatter([x], [y], c=[pal[k]], s=300, zorder=3)
        # plt.text(x,y, k, fontsize = 10, zorder = 10,ha = 'center', va = 'center')
        plt.axis("off")

    for e0, e1 in g1.edges():
        p = g1.get_edge_data(e0, e1, default=0)
        if len(p) == 0:
            p = 0
        else:
            p = p["weight"]
        print(p)

        alpha = 3 * p**1
        if alpha > 1:
            alpha = 1

        plt.plot(
            [pos[e0][0], pos[e1][0]],
            [pos[e0][1], pos[e1][1]],
            c="black",
            alpha=alpha,
            linewidth=3 * p**3,
        )
    if save_fig == True:
        _savefig(None, output_dir, pathlib.Path(save_name).stem, "", _style)


#######


def pl_plot_modules_heatmap(
    data, cns, cts, figsize=(20, 5), num_tissue_modules=2, num_cn_modules=5
):
    """
    Plot the modules and their loadings using heatmaps.

    Parameters
    ----------
    data : array-like
        The input data.
    cns : list
        The names of the copy number alterations (CNs).
    cts : list
        The names of the cell types.
    figsize : tuple, optional
        The figure size. Defaults to (20, 5).
    num_tissue_modules : int, optional
        The number of tissue modules. Defaults to 2.
    num_cn_modules : int, optional
        The number of CN modules. Defaults to 5.

    Returns
    -------
    None
    """

    figsize = figsize
    core, factors = non_negative_tucker(
        data, rank=[num_tissue_modules, num_cn_modules, num_cn_modules], random_state=32
    )
    plt.subplot(1, 2, 1)
    sns.heatmap(pd.DataFrame(factors[1], index=cns))
    plt.ylabel("CN")
    plt.xlabel("CN module")
    plt.title("Loadings onto CN modules")
    plt.subplot(1, 2, 2)
    sns.heatmap(pd.DataFrame(factors[2], index=cts))
    plt.ylabel("CT")
    plt.xlabel("CT module")
    plt.title("Loadings onto CT modules")
    plt.show()

    figsize = (num_tissue_modules * 3, 3)
    for p in range(num_tissue_modules):
        plt.subplot(1, num_tissue_modules, p + 1)
        sns.heatmap(pd.DataFrame(core[p]))
        plt.title("tissue module {}, couplings".format(p))
        plt.ylabel("CN module")
        plt.ylabel("CT module")
    plt.show()


#######

"""
This is a Python function that generates a graphical representation of modules discovered in a dataset using non-negative matrix factorization (NMF).
The function takes as input a dataset ('dat'), lists of tissue types ('cts') and copy number segments ('cns'), and parameters specifying the number of tissue and copy number modules to identify ('num_tissue_modules' and 'num_cn_modules', respectively).
The function then performs NMF on the input dataset and plots a separate graph for each tissue module.

In each graph, the function displays the copy number segments and tissue types as scatter points, with the color of each point representing the degree to which that segment or type belongs to the corresponding module.
The function also draws rectangles and lines to visually separate the different modules and indicate the strength of the relationships between copy number segments and tissue types within each module.
The resulting plots can be saved to a specified file path and name using the 'save_path' and 'save_name' arguments.
"""


def pl_plot_modules_graphical(
    data,
    cts,
    cns,
    num_tissue_modules=2,
    num_cn_modules=4,
    scale=0.4,
    color_dic=None,
    save_name=None,
    save_path=None,
    style=None,
):
    """
    Generate a graphical representation of modules discovered in a dataset using non-negative matrix factorization (NMF).

    Parameters
    ----------
    data : array-like
        The input dataset.
    cts : list
        The list of tissue types.
    cns : list
        The list of copy number segments.
    num_tissue_modules : int, optional
        The number of tissue modules to identify. Defaults to 2.
    num_cn_modules : int, optional
        The number of copy number modules to identify. Defaults to 4.
    scale : float, optional
        The scaling factor for the plot. Defaults to 0.4.
    color_dic : dict, optional
        The color palette dictionary. Defaults to None.
    save_name : str, optional
        The name for saving the plots. Defaults to None.
    save_path : str, optional
        The file path for saving the plots. Defaults to None.

    Returns
    -------
    None
    """

    _style = _apply_style(style)
    core, factors = non_negative_tucker(
        data, rank=[num_tissue_modules, num_cn_modules, num_cn_modules], random_state=32
    )

    if color_dic is None:
        color_dic = sns.color_palette("bright", 10)
    palg = sns.color_palette("Greys", 10)

    figsize = (3.67 * scale, 2.00 * scale)
    cn_scatter_size = scale * scale * 45
    cel_scatter_size = scale * scale * 15

    for p in range(num_tissue_modules):
        for idx in range(num_cn_modules):
            an = (
                float(np.max(core[p][idx, :]) > 0.1)
                + (np.max(core[p][idx, :]) <= 0.1) * 0.05
            )
            ac = (
                float(np.max(core[p][:, idx]) > 0.1)
                + (np.max(core[p][:, idx]) <= 0.1) * 0.05
            )

            cn_fac = factors[1][:, idx]
            cel_fac = factors[2][:, idx]

            cols_alpha = [
                (*color_dic[cn], an * np.minimum(cn_fac, 1.0)[i])
                for i, cn in enumerate(cns)
            ]
            cols = [
                (*color_dic[cn], np.minimum(cn_fac, 1.0)[i]) for i, cn in enumerate(cns)
            ]
            cell_cols_alpha = [
                (0, 0, 0, an * np.minimum(cel_fac, 1.0)[i])
                for i, _ in enumerate(cel_fac)
            ]
            cell_cols = [
                (0, 0, 0, np.minimum(cel_fac, 1.0)[i]) for i, _ in enumerate(cel_fac)
            ]

            plt.scatter(
                0.5 * np.arange(len(cn_fac)),
                5 * idx + np.zeros(len(cn_fac)),
                c=cols_alpha,
                s=cn_scatter_size,
            )
            offset = 9
            for i, k in enumerate(cns):
                plt.text(
                    0.5 * i,
                    5 * idx,
                    k,
                    fontsize=scale * 2,
                    ha="center",
                    va="center",
                    alpha=an,
                )

            plt.scatter(
                -4.2 + 0.25 * np.arange(len(cel_fac)) + offset,
                5 * idx + np.zeros(len(cel_fac)),
                c=cell_cols_alpha,
                s=0.5 * cel_scatter_size,
            )  # ,vmax = 0.5,edgecolors=len(cell_cols_alpha)*[(0,0,0,min(1.0,max(0.1,2*an)))], linewidths= 0.05)

            rect = plt.Rectangle(
                (-0.5, 5 * idx - 2),
                4.5,
                4,
                linewidth=scale * scale * 1,
                edgecolor="black",
                facecolor="none",
                zorder=0,
                alpha=an,
                linestyle="--",
            )
            ax = plt.gca()
            ax.add_artist(rect)
            plt.scatter(
                [offset - 5],
                [5 * idx],
                c="black",
                marker="D",
                s=scale * scale * 5,
                zorder=5,
                alpha=an,
            )
            plt.text(
                offset - 5,
                5 * idx,
                idx,
                color="white",
                alpha=an,
                ha="center",
                va="center",
                zorder=6,
                fontsize=4.5,
            )
            plt.scatter(
                [offset - 4.5],
                [5 * idx],
                c="black",
                marker="D",
                s=scale * scale * 5,
                zorder=5,
                alpha=ac,
            )
            plt.text(
                offset - 4.5,
                5 * idx,
                idx,
                color="white",
                alpha=ac,
                ha="center",
                va="center",
                zorder=6,
                fontsize=4.5,
            )

            rect = plt.Rectangle(
                (offset - 4.5, 5 * idx - 2),
                4.5,
                4,
                linewidth=scale * 1,
                edgecolor="black",
                facecolor="none",
                zorder=0,
                alpha=ac,
                linestyle="-.",
            )
            ax.add_artist(rect)

        for i, ct in enumerate(cts):
            plt.text(
                -4.2 + offset + 0.25 * i,
                27.5,
                ct,
                rotation=45,
                color="black",
                ha="left",
                va="bottom",
                fontsize=scale * 2,
                alpha=1,
            )
        for cn_i in range(num_cn_modules):
            for cel_i in range(num_cn_modules):
                plt.plot(
                    [-3 + offset - 2, -4 + offset - 0.5],
                    [5 * cn_i, 5 * cel_i],
                    color="black",
                    linewidth=2
                    * scale
                    * scale
                    * 1
                    * min(1.0, max(0, -0.00 + core[p][cn_i, cel_i])),
                    alpha=min(1.0, max(0.000, -0.00 + 10 * core[p][cn_i, cel_i])),
                )  # max(an,ac))

        plt.ylim(-5, 30)
        plt.axis("off")

        if save_name:
            _savefig(None, save_path, save_name + "_" + str(p), "_tensor", _style)

        plt.show()


#########


def pl_evaluate_ranks(data, num_tissue_modules=2):
    """
    Evaluate the reconstruction error of different ranks in non-negative matrix factorization (NMF).

    Parameters
    ----------
    data : array-like
        The input dataset.
    num_tissue_modules : int, optional
        The number of tissue modules to evaluate. Defaults to 2.

    Returns
    -------
    None
    """

    num_tissue_modules = num_tissue_modules + 1
    pal = sns.color_palette("bright", 10)
    palg = sns.color_palette("Greys", 10)

    mat1 = np.zeros((num_tissue_modules, 15))
    for i in range(2, 15):
        for j in range(1, num_tissue_modules):
            # we use NNTD as described in the paper
            facs_overall = non_negative_tucker(data, rank=[j, i, i], random_state=2336)
            mat1[j, i] = np.mean(
                (data - tl.tucker_to_tensor((facs_overall[0], facs_overall[1]))) ** 2
            )
    for j in range(1, num_tissue_modules):
        plt.plot(2 + np.arange(13), mat1[j][2:], label="rank = ({},x,x)".format(j))

    plt.xlabel("x")
    plt.ylabel("reconstruction error")
    plt.legend()
    plt.show()


#########


"""
data: the input pandas data frame.
sub_list2: a list of subcategories to be considered.
per_categ: the categorical column in the data frame to be used.
group2: the grouping column in the data frame.
repl: the replicate column in the data frame.
sub_collumn: the subcategory column in the data frame.
cell: the cell type column in the data frame.
thres (optional): the threshold for the correlation, default is 0.9.
normed (optional): if the percentage should be normalized, default is True.
cell2 (optional): the second cell type column in the data frame.
"""


def pl_corr_cell(
    data,
    per_categ,
    group2,
    rep,
    sub_column,
    cell,
    output_dir,
    save_name,
    thres=0.9,
    normed=True,
    cell2=None,
    sub_list2=None,
    style=None,
):
    """
    Perform correlation analysis on a pandas DataFrame.

    Parameters
    ----------
    data : pandas DataFrame
        The input DataFrame.
    per_categ : str
        The categorical column in the DataFrame to be used.
    group2 : str
        The grouping column in the DataFrame.
    rep : str
        The replicate column in the DataFrame.
    sub_column : str
        The subcategory column in the DataFrame.
    cell : str
        The cell type column in the DataFrame.
    output_dir : str
        The directory to save the correlation plot.
    save_name : str
        The name of the saved correlation plot.
    thres : float, optional
        The threshold for correlation. Default is 0.9.
    normed : bool, optional
        If the percentage should be normalized. Default is True.
    cell2 : str, optional
        The second cell type column in the DataFrame. Default is None.
    sub_list2 : list, optional
        A list of subcategories to be considered. Default is None.

    Returns
    -------
    all_pairs : list
        List of all correlated pairs.
    pair2 : list
        List of correlated pairs above the threshold.

    """

    _style = _apply_style(style)
    if sub_list2 != None:
        result = hf_per_only(
            data=data,
            per_cat=per_categ,
            grouping=group2,
            sub_list=sub_list2,
            replicate=rep,
            sub_col=sub_column,
            norm=normed,
        )
    else:
        sub_list2 = data[per_categ].unique()
        result = hf_per_only(
            data=data,
            per_cat=per_categ,
            grouping=group2,
            sub_list=sub_list2,
            replicate=rep,
            sub_col=sub_column,
            norm=normed,
        )

    # Format for correlation function
    mp = pd.pivot_table(
        result, columns=[per_categ], index=[group2, rep], values=["percentage"]
    )
    mp.columns = mp.columns.droplevel(0)
    cc = mp.reset_index()
    cmat = cc.corr()

    # Plot
    sl2, pair2, all_pairs = hf_cor_subset(cor_mat=cmat, threshold=thres, cell_type=cell)

    if cell2:
        sl3 = [cell2, cell]
        pl_cor_subplot(
            mp=cc, sub_list=sl3, output_dir=output_dir, save_name=cell + "_" + cell2
        )
    else:
        pl_cor_subplot(mp=cc, sub_list=sl2, output_dir=output_dir, save_name=cell)

    if save_name:
        _savefig(None, output_dir, save_name, "", _style)

    return all_pairs, pair2, cc


###########


"""
data: Pandas data frame which is used as input for plotting.


group1: Categorical column in data that will be used as the x-axis in the pairplot.

per_cat: Categorical column in data that will be used to calculate the correlation between categories in group1.

sub_col (optional): Categorical column in data that is used to subset the data.

sub_list (optional): List of values that is used to select a subset of data based on the sub_col.

norm (optional): Boolean that determines if the data should be normalized or not.

group2 (optional): Categorical column in data that is used to group the data.

count (optional): Boolean that determines if the count of each category in per_cat should be used instead of the percentage.

plot_scatter (optional): Boolean that determines if the scatterplot should be plotted or not.

cor_mat: Output data frame containing the correlation matrix.

mp: Output data frame containing the pivot table of the count or percentage of each category in per_cat based on group1.


Returns:
cor_mat (pandas dataframe): Correlation matrix.
mp (pandas dataframe): Data after pivoting and grouping.
"""


def pl_cor_plot(
    data,
    group1,
    per_cat,
    sub_col=None,
    sub_list=None,
    norm=False,
    group2=None,
    count=False,
    plot_scatter=True,
):
    """
    Create a correlation plot using a pandas DataFrame.

    Parameters
    ----------
    data : pandas DataFrame
        The input DataFrame.
    group1 : str
        Categorical column in data that will be used as the x-axis in the pairplot.
    per_cat : str
        Categorical column in data that will be used to calculate the correlation between categories in group1.
    sub_col : str, optional
        Categorical column in data that is used to subset the data. Default is None.
    sub_list : list, optional
        List of values that is used to select a subset of data based on the sub_col. Default is None.
    norm : bool, optional
        Boolean that determines if the data should be normalized or not. Default is False.
    group2 : str, optional
        Categorical column in data that is used to group the data. Default is None.
    count : bool, optional
        Boolean that determines if the count of each category in per_cat should be used instead of the percentage. Default is False.
    plot_scatter : bool, optional
        Boolean that determines if the scatterplot should be plotted or not. Default is True.

    Returns
    -------
    cor_mat : pandas DataFrame
        Correlation matrix.
    mp : pandas DataFrame
        Data after pivoting and grouping.
    """

    if group2:
        plt.rcParams["legend.markerscale"] = 1
        tf = (
            data.groupby([group1, group2])
            .apply(lambda x: x[per_cat].value_counts(normalize=True, sort=False) * 100)
            .to_frame()
        )
        tf.columns = tf.columns.astype(str)
        tf.reset_index(inplace=True)
        mp = pd.pivot_table(
            tf, columns=["level_2"], index=[group1, group2], values=[per_cat]
        )
        mp.columns = mp.columns.droplevel(0)
        mp.reset_index(inplace=True)
        mp2 = mp.fillna(0)
        cor_mat = mp2.corr()
        mask = np.triu(np.ones_like(cor_mat, dtype=bool))
        plt.figure(figsize=(len(cor_mat.index), len(cor_mat.columns) * 0.8))
        sns.heatmap(cor_mat, cmap="coolwarm", center=0, vmin=-1, vmax=1, mask=mask)
        if plot_scatter:
            sns.pairplot(
                mp,
                diag_kind="kde",
                plot_kws={"alpha": 0.6, "s": 80, "edgecolor": "k"},
                size=4,
                hue=group2,
            )
    else:
        if count:
            tf = data.groupby([group1, per_cat]).count()["region"].to_frame()
            tf.reset_index(inplace=True)
            mp = pd.pivot_table(
                tf, columns=[per_cat], index=[group1], values=["region"]
            )
            mp.columns = mp.columns.droplevel(0)
            mp.reset_index(inplace=True)
            mp2 = mp.fillna(0)
            cor_mat = mp2.corr()
            mask = np.triu(np.ones_like(cor_mat, dtype=bool))
            plt.figure(figsize=(len(cor_mat.index), len(cor_mat.columns) * 0.8))
            sns.heatmap(cor_mat, cmap="coolwarm", center=0, vmin=-1, vmax=1, mask=mask)
            if plot_scatter:
                sns.pairplot(
                    mp,
                    diag_kind="kde",
                    plot_kws={"scatter_kws": {"alpha": 0.6, "s": 80, "edgecolor": "k"}},
                    size=4,
                    kind="reg",
                )
        else:
            # Find Percentage of cell type
            test = data.copy()

            if sub_list == None:
                sub_list = data[per_cat].unique()

            sub_list1 = sub_list.copy()

            if norm == True:
                test1 = test.loc[test[sub_col].isin(sub_list1)]
                immune_list = list(test1[per_cat].unique())
            else:
                test1 = test.copy()
                immune_list = list(
                    test.loc[test[sub_col].isin(sub_list1)][per_cat].unique()
                )

            test1[per_cat] = test1[per_cat].astype("category")
            tf = test1.groupby([group1]).apply(
                lambda x: x[per_cat].value_counts(normalize=True, sort=False) * 100
            )
            tf.columns = tf.columns.astype(str)
            mp = tf[immune_list]
            mp.reset_index(inplace=True)
            cor_mat = mp.corr()
            mask = np.triu(np.ones_like(cor_mat, dtype=bool))
            plt.figure(figsize=(len(cor_mat.index), len(cor_mat.columns) * 0.8))
            sns.heatmap(cor_mat, cmap="coolwarm", center=0, vmin=-1, vmax=1, mask=mask)
            if plot_scatter:
                sns.pairplot(
                    mp,
                    diag_kind="kde",
                    plot_kws={"scatter_kws": {"alpha": 0.6, "s": 80, "edgecolor": "k"}},
                    size=4,
                    kind="reg",
                )

    return cor_mat, mp


########


"""
mp: A pandas dataframe from which a subset of columns will be selected and plotted.
sub_list: A list of column names from the dataframe mp that will be selected and plotted.
save_name (optional): A string that specifies the file name for saving the plot.
If save_name is not provided, the plot will not be saved.
"""


def pl_cor_subplot(mp, sub_list, output_dir, save_name=None, style=None):
    """
    Create a subplot of pairwise correlation plots using a subset of columns from a pandas DataFrame.

    Parameters
    ----------
    mp : pandas DataFrame
        The input DataFrame from which a subset of columns will be selected and plotted.
    sub_list : list
        A list of column names from the dataframe `mp` that will be selected and plotted.
    output_dir : str
        The output directory where the plot will be saved.
    save_name : str, optional
        A string that specifies the file name for saving the plot. If `save_name` is not provided, the plot will not be saved.
    """

    _style = _apply_style(style)
    sub_cor = mp[sub_list]
    sns.pairplot(
        sub_cor,
        diag_kind="kde",
        plot_kws={"scatter_kws": {"alpha": 0.6, "s": 80, "edgecolor": "k"}},
        size=4,
        kind="reg",
        corner=True,
    )
    if save_name:
        _savefig(None, output_dir, save_name, "_corrplot", _style)


def annotate(data, names, **kws):
    """
    Annotate an axes with the Pearson correlation coefficient and p-value.

    Intended to be passed as a mapping function to seaborn plot grids.  The
    annotation is placed in the upper-middle area of the current axes.

    Parameters
    ----------
    data : pandas.DataFrame
        Data frame containing the two columns to correlate.
    names : list of str
        Two-element list with the column names to correlate.  ``names[0]``
        is used as x and ``names[1]`` as y.
    **kws
        Additional keyword arguments (accepted but ignored, for compatibility
        with seaborn map interfaces).

    Returns
    -------
    None
    """
    r, p = sp.stats.pearsonr(data[names[0]], data[names[1]])
    ax = plt.gca()
    ax.text(
        0.5, 0.8, "r={:.2f}, p={:.2g}".format(r, p), transform=ax.transAxes, fontsize=14
    )


def pl_cor_subplot_new(mp, sub_list, output_dir, save_name=None, style=None):
    """
    Create a subplot of pairwise correlation plots using a subset of columns from a pandas DataFrame.

    Parameters
    ----------
    mp : pandas DataFrame
        The input DataFrame from which a subset of columns will be selected and plotted.
    sub_list : list
        A list of column names from the dataframe `mp` that will be selected and plotted.
    output_dir : str
        The output directory where the plot will be saved.
    save_name : str, optional
        A string that specifies the file name for saving the plot. If `save_name` is not provided, the plot will not be saved.
    """

    _style = _apply_style(style)
    sub_cor = mp[sub_list]

    names = sub_cor.columns.tolist()

    g = sns.lmplot(x=names[0], y=names[1], data=df2, height=5, aspect=1)

    g.map_dataframe(annotate(data=mp, names=names))
    plt.xlabel(names[0], fontsize=14)
    plt.ylabel(names[1], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    if save_name:
        _savefig(None, output_dir, save_name, "_corrplot", _style)


########


def pl_Niche_heatmap(k_centroids, w, n_num, sum_cols):
    """
    Create a heatmap to show the types of cells (ClusterIDs) in different niches.

    Parameters
    ----------
    k_centroids : numpy array
        The centroid values for each niche.
    w : pandas DataFrame
        The input data containing cluster information.
    n_num : int
        The niche number to plot.
    sum_cols : list
        The list of columns to sum for the heatmap.
    """

    # this plot shows the types of cells (ClusterIDs) in the different niches (0-9)
    k_to_plot = n_num
    niche_clusters = k_centroids[k_to_plot]
    values = w[sum_cols].values
    tissue_avgs = values.mean(axis=0)
    fc = np.log2(
        (
            (niche_clusters + tissue_avgs)
            / (niche_clusters + tissue_avgs).sum(axis=1, keepdims=True)
        )
        / tissue_avgs
    )
    fc = pd.DataFrame(fc, columns=sum_cols)
    s = sns.clustermap(fc, cmap="bwr", vmax=-5)


def pl_Barycentric_coordinate_projection(
    w,
    plot_list,
    threshold,
    output_dir,
    save_name,
    col_dic,
    l,
    n_num,
    cluster_col,
    SMALL_SIZE=14,
    MEDIUM_SIZE=16,
    BIGGER_SIZE=18,
    figsize=(14, 14),
    style=None,
):
    """
    Create a barycentric coordinate projection plot.

    Parameters
    ----------
    w : pandas DataFrame
        The input data containing coordinate information.
    plot_list : list
        The list of columns to plot.
    threshold : int
        The threshold value for data selection.
    output_dir : str
        The output directory where the plot will be saved.
    save_name : str
        The file name for saving the plot.
    col_dic : dict
        A dictionary mapping cluster IDs to colors.
    l : list
        A list of cluster IDs.
    n_num : int
        The niche number.
    cluster_col : str
        The column containing cluster information.
    SMALL_SIZE : int, optional
        The font size for small text. Default is 14.
    MEDIUM_SIZE : int, optional
        The font size for medium text. Default is 16.
    BIGGER_SIZE : int, optional
        The font size for large text. Default is 18.
    figsize : tuple, optional
        The figure size. Default is (14, 14).
    """

    _style = _apply_style(style)
    import matplotlib as _mpl

    _mpl.rcParams.update(
        {
            "font.size": SMALL_SIZE,
            "axes.titlesize": SMALL_SIZE,
            "axes.labelsize": MEDIUM_SIZE,
            "xtick.labelsize": SMALL_SIZE,
            "ytick.labelsize": SMALL_SIZE,
            "legend.fontsize": SMALL_SIZE,
            "figure.titlesize": BIGGER_SIZE,
        }
    )

    lmap = {j: i for i, j in enumerate(l)}
    palt = col_dic

    wgc = w.loc[w.loc[:, plot_list].sum(axis=1) > threshold, :]
    idx = wgc.index.values
    xl = wgc.loc[:, plot_list]
    proj = np.array([[0, 0], [np.cos(np.pi / 3), np.sin(np.pi / 3)], [1, 0]])
    coords = np.dot(xl / n_num, proj)  #####window size fraction

    plt.figure(figsize=figsize)
    jit = 0.002
    cols = [palt[a] for a in wgc[cluster_col]]

    plt.scatter(
        coords[:, 0] + jit * np.random.randn(len(coords)),
        coords[:, 1] + jit * np.random.randn(len(coords)),
        s=15,
        alpha=0.5,
        c=cols,
    )
    plt.axis("off")
    plt.show()

    if save_name:
        _savefig(None, output_dir, save_name, "", _style)


########


def pl_get_network(
    ttl_per_thres,
    comb_per_thres,
    color_dic,
    windows,
    n_num,
    l,
    tissue_col=None,
    tissue_subset_list=None,
    sub_col="Tissue Unit",
    neigh_sub=None,
    save_name=None,
    save_path=None,
    figsize=(20, 10),
):
    """
    Generate a network plot based on combination frequencies.

    Parameters
    ----------
    ttl_per_thres : float
        The threshold for the total percentage of combinations.
    comb_per_thres : float
        The threshold for the combination frequency.
    color_dic : dict
        A dictionary mapping cluster IDs to colors.
    windows : dict
        A dictionary containing window data.
    n_num : int
        The window size.
    l : list
        A list of cluster IDs.
    tissue_col : bool or None, optional
        Whether to filter data based on tissue columns. Default is None.
    tissue_subset_list : list or None, optional
        A list of tissue subsets to consider. Default is None.
    sub_col : str, optional
        The name of the column for subsetting. Default is 'Tissue Unit'.
    neigh_sub : None, optional
        Subset neighborhoods based on specified values. Default is None.
    save_name : str or None, optional
        The name for saving the plot. Default is None.
    save_path : str or None, optional
        The path for saving the plot. Default is None.
    figsize : tuple, optional
        The figure size. Default is (20, 10).
    """

    # Choose the windows size to continue with
    w = windows[n_num]
    if tissue_col == True:
        w = w[w.tissue_col.isin(tissue_subset_list)]
    if neigh_sub:
        w = w[w[sub_col].isin(neigh_sub)]
    xm = w.loc[:, l].values / n_num

    # Get the neighborhood combinations based on the threshold
    simps = hf_get_thresh_simps(xm, ttl_per_thres)
    simp_freqs = simps.value_counts(normalize=True)
    simp_sums = np.cumsum(simp_freqs)

    g = nx.DiGraph()
    thresh_cumulative = 0.95
    thresh_freq = comb_per_thres
    # selected_simps = simp_sums[simp_sums<=thresh_cumulative].index.values
    selected_simps = simp_freqs[simp_freqs >= thresh_freq].index.values

    # this builds the graph for the CN combination map
    selected_simps
    for e0 in selected_simps:
        for e1 in selected_simps:
            if (set(list(e0)) < set(list(e1))) and (len(e1) == len(e0) + 1):
                g.add_edge(e0, e1)

    # this plots the CN combination map

    draw = g
    pos = _graphviz_layout(draw, prog="dot", seed=1)
    height = 8

    plt.figure(figsize=figsize)
    for n in draw.nodes():
        col = "black"
        if len(draw.in_edges(n)) < len(n):
            col = "black"
        plt.scatter(
            pos[n][0],
            pos[n][1] - 5,
            s=simp_freqs[list(simp_freqs.index).index(n)] * 10000,
            c=col,
            zorder=-1,
        )
        #         if n in tops:
        #             plt.text(pos[n][0],pos[n][1]-7, '*', fontsize = 25, color = 'white', ha = 'center', va = 'center',zorder = 20)
        delta = 8
        # plot_sim((pos[n][0]+delta, pos[n][1]+delta),n, scale = 20,s = 200,text = True,fontsize = 15)
        plt.scatter(
            [pos[n][0]] * len(n),
            [pos[n][1] + delta * (i + 1) for i in range(len(n))],
            c=[color_dic[l[i]] for i in n],
            marker="^",
            zorder=5,
            s=400,
        )

    j = 0
    for e0, e1 in draw.edges():
        weight = 0.2
        alpha = 0.3
        if len(draw.in_edges(e1)) < len(e1):
            color = "black"
            lw = 1
            weight = 0.4
        color = "black"
        plt.plot(
            [pos[e0][0], pos[e1][0]],
            [pos[e0][1], pos[e1][1]],
            color=color,
            linewidth=weight,
            alpha=alpha,
            zorder=-10,
        )

    plt.axis("off")
    if save_name is not None:
        plt.savefig(save_path + save_name + "_spatial_contexts.pdf")  #'.png', dpi=300)
    plt.show()


#########


def pl_spatial_context_stats_vis(
    neigh_comb,
    simp_df_tissue1,
    simp_df_tissue2,
    pal_tis={"Resection": "blue", "Biopsy": "orange"},
    plot_order=["Resection", "Biopsy"],
    figsize=(5, 5),
):
    """
    Visualise and compare the frequency of a neighbourhood combination across two tissue types.

    Performs a Mann-Whitney U test between the two tissue groups for the
    specified neighbourhood combination and plots a box-swarm plot
    coloured by tissue type.

    Parameters
    ----------
    neigh_comb : tuple
        Neighbourhood combination identifier used to index rows in the
        simplex data frames.
    simp_df_tissue1 : pandas.DataFrame
        Simplex frequency data frame for tissue type 1 (rows = neighbourhood
        combinations, columns = samples).
    simp_df_tissue2 : pandas.DataFrame
        Simplex frequency data frame for tissue type 2 (rows = neighbourhood
        combinations, columns = samples).
    pal_tis : dict, optional
        Colour palette mapping tissue labels to colours.
        Default is ``{"Resection": "blue", "Biopsy": "orange"}``.
    plot_order : list, optional
        Display order of tissue types on the x-axis.
        Default is ``["Resection", "Biopsy"]``.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (5, 5).

    Returns
    -------
    None
    """
    # Set Neigh and make comparison
    neigh_comb = (9,)

    df1 = simp_df_tissue1.loc[[neigh_comb]].T
    df2 = simp_df_tissue2.loc[[neigh_comb]].T
    print(stats.mannwhitneyu(df1[df1.columns[0]], df2[df2.columns[0]]))

    df1.reset_index(inplace=True)
    df1[["donor", "tissue"]] = df1["index"].str.split("_", expand=True)
    df2.reset_index(inplace=True)
    df2[["donor", "tissue"]] = df2["index"].str.split("_", expand=True)
    df_m = pd.concat([df1, df2])
    df_m["combo"] = str(neigh_comb)

    # swarmplot to compare
    plt.figure(figsize=figsize)

    ax = sns.boxplot(
        data=df_m,
        x="combo",
        y=neigh_comb,
        hue="tissue",
        dodge=True,
        hue_order=plot_order,
        palette=pal_tis,
    )
    ax = sns.swarmplot(
        data=df_m,
        x="combo",
        y=neigh_comb,
        hue="tissue",
        dodge=True,
        hue_order=plot_order,
        edgecolor="black",
        linewidth=1,
        palette=pal_tis,
    )
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.3))
    plt.xlabel("")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles[: len(df_m["tissue"].unique())],
        labels[: len(df_m["tissue"].unique())],
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        frameon=False,
    )
    plt.xticks(rotation=90)
    sns.despine(trim=True)

    # pt.savefig(save_path+save_name+'_swarm_boxplot.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


##########


def pl_conplot(
    df,
    feature,
    exp="Exp",
    X="X",
    Y="Y",
    invert_y=False,
    cmap="RdBu",
    size=5,
    alpha=1,
    figsize=10,
    exps=None,
    fig=None,
    **kwargs,
):
    """
    Plot continuous variable with a colormap:

    df:  dataframe of cells with spatial location and feature to color.  Must have columns ['X','Y','Exp',feature]
    feature:  feature in df to color points by
    cmap:  matplotlib colormap
    size:  point size
    thresh_val: only include points below this value
    """
    if invert_y:
        y_orig = df[Y].values.copy()
        df[Y] *= -1

    if exps is None:
        exps = list(df[exp].unique())  # display all experiments
    elif type(exps) != list:
        exps = [exps]

    if fig is None:
        f, ax = plt.subplots(len(exps), 1, figsize=(figsize, len(exps) * figsize))
        if len(exps) == 1:
            ax = [ax]
    else:
        f, ax = fig

    for i, name in enumerate(exps):
        data = df[df[exp] == name]

        ax[i].scatter(
            data[X], -data[Y], c=data[feature], cmap=cmap, s=size, alpha=alpha, **kwargs
        )
        ax[i].set_title(name + "_" + str(feature) + "_" + str(len(data)))
        ax[i].axis("off")

    if invert_y:
        df[Y] = y_orig
    return f, ax


##############


def pl_catplot(
    df,
    hue,
    exp="Exp",
    X="X",
    Y="Y",
    invert_y=False,
    size=3,
    legend=True,
    palette="bright",
    figsize=5,
    background="white",
    style=None,
    exps=None,
    axis="on",
    ticks_fontsize=15,
    scatter_kws={},
    **kwargs,
):
    """
    Plots cells in tissue section color coded by either cell type or node allocation.
    df:  dataframe with cell information
    size:  size of point to plot for each cell.
    hue:  color by "Clusterid" or "Node" respectively.
    legend:  to include legend in plot.
    """
    import warnings

    if isinstance(style, str) and style not in (None,):
        import pathlib as _pl

        if not _pl.Path(str(style)).exists():
            warnings.warn(
                "Passing a background color string as 'style=' is deprecated. "
                "Use 'background=' for the axes background color.",
                DeprecationWarning,
                stacklevel=2,
            )
            background, style = style, None
    _style = _apply_style(style)
    scatter_kws_ = {"s": size, "alpha": 1}
    scatter_kws_.update(scatter_kws)

    figures = []
    df = df.rename(columns=lambda x: str(x))

    df[hue] = df[hue].astype("category")
    if invert_y:
        y_orig = df[Y].values.copy()
        df[Y] *= -1

    sns.set_style({"axes.facecolor": background})
    if exps == None:
        exps = list(df[exp].unique())  # display all experiments
    elif type(exps) != list:
        exps = [exps]

    for name in exps:
        data = df[df[exp] == name]

        data[X] = data[X] - data[X].min()
        data[Y] = data[Y] - data[Y].min()

        print(name)
        xrange = data[X].max() - data[X].min()
        yrange = data[Y].max() - data[Y].min()
        #        if 'aspect' not in kwargs:
        #            kwargs['aspect'] = xrange/yrange
        f = sns.lmplot(
            x=X,
            y=Y,
            data=data,
            hue=hue,
            legend=legend,
            fit_reg=False,
            markers=".",
            height=yrange / 400,
            palette=palette,
            scatter=True,
            scatter_kws=scatter_kws_,
            aspect=xrange / yrange,
            **kwargs,
        )

        if axis == "off":
            sns.despine(top=True, right=True, left=True, bottom=True)
            f = f.set(xticks=[], yticks=[]).set_xlabels("").set_ylabels("")
        # plt.legend(frameon=True)

        plt.title(name)

        plt.xticks(fontsize=_style.font_size_tiny)  # Increase x-axis label size
        plt.yticks(fontsize=_style.font_size_tiny)  # Increase y-axis label size

        plt.show()
        figures += [f]

    if invert_y:
        df[Y] = y_orig

    return figures


##########


def pl_comb_num_freq(data_list, plot_order=None, pal_tis=None, figsize=(5, 5)):
    """
    Plot the frequency distribution of neighbourhood combination sizes.

    For each data frame in ``data_list`` the function counts how many
    neighbourhood combinations contain 1, 2, 3, … members and then
    shows a box-swarm plot of those frequencies stratified by tissue type.

    Parameters
    ----------
    data_list : list of pandas.DataFrame
        List of simplex-frequency data frames.  Each frame must have a
        ``merge`` column (renamed internally to ``combination``) and
        sample-level columns whose names follow the pattern
        ``<donor>_<tissue>``.
    plot_order : list, optional
        Display order for tissue types on the x-axis. Default is None.
    pal_tis : dict or str, optional
        Seaborn colour palette or dictionary mapping tissue labels to
        colours. Default is None.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (5, 5).

    Returns
    -------
    pandas.DataFrame
        Melted data frame with columns ``count``, ``unique_cond``,
        ``fraction``, ``donor``, and ``tissue``.
    """
    df_new = []
    for df in data_list:
        df.reset_index(inplace=True)
        df.rename(columns={"merge": "combination"}, inplace=True)
        df["count"] = df["combination"].apply(len)
        sum_df = df.groupby("count").sum()

        tbt = sum_df.reset_index()
        ttt = tbt.melt(id_vars=["count"])
        ttt.rename(
            columns={"variable": "unique_cond", "value": "fraction"}, inplace=True
        )
        df_new.append(ttt)
    df_exp = pd.concat(df_new)

    df_exp[["donor", "tissue"]] = df_exp["unique_cond"].str.split("_", expand=True)

    # swarmplot to compare
    plt.figure(figsize=figsize)

    ax = sns.boxplot(
        data=df_exp,
        x="count",
        y="fraction",
        hue="tissue",
        dodge=True,
        hue_order=plot_order,
        palette=pal_tis,
    )
    ax = sns.swarmplot(
        data=df_exp,
        x="count",
        y="fraction",
        hue="tissue",
        dodge=True,
        hue_order=plot_order,
        edgecolor="black",
        linewidth=1,
        palette=pal_tis,
    )
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.3))
    plt.xlabel("")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles[: len(df_exp["tissue"].unique())],
        labels[: len(df_exp["tissue"].unique())],
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        frameon=False,
    )
    plt.xticks(rotation=90)
    sns.despine(trim=True)

    return df_exp


##########
# this function helps to determine what threshold to use for remove noises
# default cut off is top 1%
def classification_report_heatmap(
    y_true=None,
    y_pred=None,
    report=None,
    target_names=None,
    figsize=None,
    savefig=False,
    output_fname="",
    output_dir="./",
    return_data=False,
    style=None,
):
    """
    Plot a per-class precision/recall/F1 classification report as a heatmap.

    Evaluates a classifier's predictions — e.g. the output of
    :func:`spacec.tools.ml_train`/:func:`spacec.tools.ml_predict`, or
    :func:`spacec.tools.adata_stellar` predictions compared against ground
    truth — as an annotated heatmap. Either pass a precomputed
    ``report`` dict (``sklearn.metrics.classification_report(...,
    output_dict=True)``) or raw ``y_true``/``y_pred`` labels to compute it.

    Parameters
    ----------
    y_true, y_pred : array-like, optional
        Ground-truth and predicted labels. Required if ``report`` is not
        given.
    report : dict, optional
        A precomputed ``sklearn.metrics.classification_report`` dict
        (``output_dict=True``). If given, ``y_true``/``y_pred`` are ignored.
    target_names : list of str, optional
        Class labels to report, in display order, when computing the report
        from ``y_true``/``y_pred``. Must match the label values in
        ``y_true``/``y_pred``; also passed as ``labels`` so row order follows
        this list. Defaults to all labels, sorted.
    figsize : tuple, optional
        Figure size. Defaults to a height scaled to the number of rows so
        every class label is shown.
    savefig : bool, optional
        Whether to save the figure. Defaults to False.
    output_fname : str, optional
        Base filename (without extension) used when ``savefig=True``.
    output_dir : str, optional
        Directory to save into when ``savefig=True``. Defaults to ``"./"``.
    return_data : bool, optional
        If True, also return the report as a DataFrame (source data for
        publication). Defaults to False.
    style : SPACEcStyle, str, or pathlib.Path, optional
        Style to apply for this call. Defaults to the active style.

    Returns
    -------
    matplotlib.axes.Axes or (matplotlib.axes.Axes, pandas.DataFrame)
        The heatmap axes, and — if ``return_data=True`` — the report table.
    """
    from sklearn.metrics import classification_report

    _style = _apply_style(style)

    if report is None:
        if y_true is None or y_pred is None:
            raise ValueError("Provide either `report`, or both `y_true` and `y_pred`.")
        # Pass labels=target_names so sklearn orders rows by target_names instead
        # of renaming its sorted label set positionally (silent mislabeling).
        report = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            labels=target_names,
            target_names=target_names,
            output_dict=True,
        )

    report_df = pd.DataFrame(report).iloc[:-1, :].T

    if figsize is None:
        figsize = (6, max(4, 0.35 * len(report_df) + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        report_df,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        xticklabels=True,
        yticklabels=True,  # "auto" drops every other label on tall heatmaps
        cmap=_style.sequential_cmap,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "score"},
    )
    ax.set_title("Classification Report")
    _apply_spines(ax, _style)

    if savefig:
        _savefig(fig, output_dir, output_fname, "_classification_report", _style)

    if return_data:
        return ax, report_df
    return ax


def confusion_matrix_heatmap(
    y_true,
    y_pred,
    labels=None,
    normalize=None,
    figsize=None,
    annot=True,
    savefig=False,
    output_fname="",
    output_dir="./",
    return_data=False,
    style=None,
):
    """
    Plot a confusion matrix between ground-truth and predicted labels.

    Complements :func:`classification_report_heatmap` — useful for spotting
    which cell types a classifier (:func:`spacec.tools.ml_predict`) or label
    transfer (:func:`spacec.tools.adata_stellar`) confuses with one another.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted labels.
    labels : list, optional
        Label order for the matrix rows/columns. Defaults to the sorted
        union of labels present in ``y_true``/``y_pred``.
    normalize : {None, 'true', 'pred', 'all'}, optional
        Normalization mode passed to ``sklearn.metrics.confusion_matrix``.
        Defaults to None (raw counts).
    figsize : tuple, optional
        Figure size. Defaults to a size scaled to the number of labels.
    annot : bool, optional
        Print the value in every cell. Set to False for large matrices whose
        cells are too small for readable numbers. Defaults to True.
    savefig : bool, optional
        Whether to save the figure. Defaults to False.
    output_fname : str, optional
        Base filename (without extension) used when ``savefig=True``.
    output_dir : str, optional
        Directory to save into when ``savefig=True``. Defaults to ``"./"``.
    return_data : bool, optional
        If True, also return the confusion matrix as a DataFrame (source
        data for publication). Defaults to False.
    style : SPACEcStyle, str, or pathlib.Path, optional
        Style to apply for this call. Defaults to the active style.

    Returns
    -------
    matplotlib.axes.Axes or (matplotlib.axes.Axes, pandas.DataFrame)
        The heatmap axes, and — if ``return_data=True`` — the confusion
        matrix table.
    """
    from sklearn.metrics import confusion_matrix

    _style = _apply_style(style)

    if len(y_true) != len(y_pred):
        raise ValueError("`y_true` and `y_pred` must have the same length.")
    if len(y_true) == 0:
        raise ValueError("`y_true` and `y_pred` are empty.")
    # check on object arrays: np.asarray(["a", nan]) would stringify nan to "nan"
    if (
        pd.isna(np.asarray(y_true, dtype=object)).any()
        or pd.isna(np.asarray(y_pred, dtype=object)).any()
    ):
        raise ValueError(
            "`y_true`/`y_pred` contain missing values; drop them first "
            "(e.g. `mask = ~pd.isna(y_true) & ~pd.isna(y_pred)`)."
        )

    if labels is None:
        labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    n = len(labels)
    if figsize is None:
        figsize = (0.45 * n + 3, 0.45 * n + 2)

    # blank out zero cells so the diagonal and real confusions stand out
    annot_values = np.where(cm > 0, np.char.mod("%.2f" if normalize else "%d", cm), "")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_df,
        annot=annot_values if annot else False,
        fmt="",
        annot_kws={"size": max(5, 11 - n // 3)},
        xticklabels=True,
        yticklabels=True,  # "auto" drops labels on large matrices
        square=True,
        cmap=_style.sequential_cmap,
        ax=ax,
        cbar_kws={"label": "fraction" if normalize else "cell count"},
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    ax.tick_params(axis="x", rotation=90)
    _apply_spines(ax, _style)

    if savefig:
        _savefig(fig, output_dir, output_fname, "_confusion_matrix", _style)

    if return_data:
        return ax, cm_df
    return ax


def compensation_qc(
    data,
    markers,
    x_col="x",
    y_col="y",
    compensated_suffix="_compensated",
    point_size=7,
    n_columns=2,
    figsize=None,
    savefig=False,
    output_fname="",
    output_dir="./",
    style=None,
    masks=None,
    label_col="label",
    mask_outline=True,
):
    """
    Plot before/after spatial panels for spillover compensation QC.

    For each marker, draws two side-by-side spatial panels — raw intensity
    vs. compensated intensity — sharing a color scale, so the effect of
    :func:`spacec.pp.compensate_cell_matrix` can be visually inspected.
    Expects ``data`` to contain the raw marker column and a
    ``marker + compensated_suffix`` column, matching the output produced
    when ``compensate_cell_matrix(..., overwrite=False)`` is used.

    By default each cell is drawn as a small dot at its centroid. Passing
    ``masks`` switches to filling each cell's actual segmentation-mask
    footprint instead, which makes spillover between touching cells much
    easier to see than overlapping dots at high cell density.

    Parameters
    ----------
    data : pandas.DataFrame
        Cell table containing spatial coordinates, raw marker columns, and
        their compensated counterparts.
    markers : str or list of str
        Marker column name(s) to plot.
    x_col, y_col : str, optional
        Spatial coordinate columns. Default ``"x"``/``"y"``.
    compensated_suffix : str, optional
        Suffix appended to a marker name to find its compensated column.
        Defaults to ``"_compensated"``.
    point_size : float, optional
        Scatter point size. Defaults to 7. Ignored when ``masks`` is given.
    n_columns : int, optional
        Unused for a single marker; reserved for future multi-marker grid
        layouts. Each marker currently renders as one before/after row.
    figsize : tuple, optional
        Figure size. Defaults to a size scaled to the number of markers.
    savefig : bool, optional
        Whether to save the figure. Defaults to False.
    output_fname : str, optional
        Base filename (without extension) used when ``savefig=True``.
    output_dir : str, optional
        Directory to save into when ``savefig=True``. Defaults to ``"./"``.
    style : SPACEcStyle, str, or pathlib.Path, optional
        Style to apply for this call. Defaults to the active style.
    masks : numpy.ndarray, optional
        2D integer-labeled segmentation mask for this call's single region
        (0 = background, each cell = a unique positive label) — the same
        array passed to :func:`spacec.pp.compensate_cell_matrix`. When
        given, each cell is rendered as its filled mask footprint instead
        of a dot, colored by marker intensity. ``data[label_col]`` must
        hold valid labels into this array. The full array is drawn every
        call (no cropping), with the axis limits set to ``data``'s
        ``x_col``/``y_col`` range — for a spatially-cropped ``data``
        subset this zooms in "for free" without needing to slice
        ``masks`` to match. Cost scales with the full mask's pixel count
        regardless of zoom; pre-crop ``masks`` yourself if this becomes a
        bottleneck on very large whole-slide masks. Default ``None`` (dots).
    label_col : str, optional
        Column in ``data`` holding each row's exact mask label, matching
        ``compensate_cell_matrix``'s own ``label_col`` default. Only used
        when ``masks`` is given. Defaults to ``"label"``.
    mask_outline : bool, optional
        Whether to draw a thin outline around each filled cell mask.
        Only used when ``masks`` is given. Defaults to True.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing one before/after row per marker.
    """
    _style = _apply_style(style)
    if isinstance(markers, str):
        markers = [markers]

    if masks is not None and label_col not in data.columns:
        raise ValueError(
            f"label_col={label_col!r} is not a column in `data`. Pass the "
            "column holding each row's exact mask label (matching "
            "compensate_cell_matrix's label_col), or omit `masks` to fall "
            "back to a dot scatter."
        )

    if figsize is None:
        figsize = (
            2 * _style.figure_width_single,
            _style.figure_height_default * len(markers),
        )

    fig, axes = plt.subplots(len(markers), 2, figsize=figsize, squeeze=False)

    if masks is not None:
        labels_arr = data[label_col].to_numpy().astype(int)
        if labels_arr.min() < 0 or labels_arr.max() > masks.max():
            raise ValueError(
                f"`data[{label_col!r}]` contains labels outside the range of "
                "`masks` — check that `data` and `masks` come from the same "
                "region."
            )

    def _plot_masks_panel(ax, values):
        cmap_obj = plt.get_cmap(_style.sequential_cmap)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        color_map = np.zeros((masks.max() + 1, 4))
        color_map[labels_arr] = cmap_obj(norm(values.to_numpy()))
        colored_masks = color_map[masks]

        if mask_outline:
            boundaries = find_boundaries(masks, mode="outer") & np.isin(
                masks, labels_arr
            )
            colored_masks[boundaries, :3] = mcolors.to_rgb(_style.edge_color_default)
            colored_masks[boundaries, 3] = 1

        ax.imshow(colored_masks, origin="lower")
        ax.set_xlim(data[x_col].min(), data[x_col].max())
        ax.set_ylim(data[y_col].min(), data[y_col].max())

        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        return sm

    for row, marker in enumerate(markers):
        comp_col = marker + compensated_suffix
        raw_vals = data[marker]
        comp_vals = data[comp_col]
        vmin = min(raw_vals.min(), comp_vals.min())
        vmax = max(raw_vals.max(), comp_vals.max())

        ax_before, ax_after = axes[row]

        if masks is not None:
            sc1 = _plot_masks_panel(ax_before, raw_vals)
        else:
            sc1 = ax_before.scatter(
                data[x_col],
                data[y_col],
                c=raw_vals,
                cmap=_style.sequential_cmap,
                s=point_size,
                alpha=_style.data_alpha_spatial,
                vmin=vmin,
                vmax=vmax,
            )
        ax_before.set_title(
            f"Before compensation — {marker}", fontsize=_style.font_size_medium
        )
        ax_before.set_aspect("equal")
        _style_colorbar(
            fig.colorbar(sc1, ax=ax_before, label=f"{marker} intensity"), _style
        )

        if masks is not None:
            sc2 = _plot_masks_panel(ax_after, comp_vals)
        else:
            sc2 = ax_after.scatter(
                data[x_col],
                data[y_col],
                c=comp_vals,
                cmap=_style.sequential_cmap,
                s=point_size,
                alpha=_style.data_alpha_spatial,
                vmin=vmin,
                vmax=vmax,
            )
        ax_after.set_title(
            f"After compensation — {marker}", fontsize=_style.font_size_medium
        )
        ax_after.set_aspect("equal")
        _style_colorbar(
            fig.colorbar(sc2, ax=ax_after, label=f"{marker} intensity"), _style
        )

        for ax in (ax_before, ax_after):
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            _apply_spines(ax, _style)

    fig.tight_layout()

    if savefig:
        _savefig(fig, output_dir, output_fname, "_compensation_qc", _style)

    plt.show()

    return fig


def pl_mono_cluster_spatial(
    df,
    sample_col="Sample",
    cluster_col="Cell Type",
    x="x",
    y="y",
    color_dict=None,
    s=None,
    alpha=None,
    figsize=(15, 12),
    style=None,
):
    """
    Plot spatial scatter plots for each cell type, one sub-plot per cell type per sample.

    For every unique sample the function creates a grid of scatter plots
    where each panel shows only the cells belonging to one cluster/cell type,
    with the y-axis inverted to match image coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with spatial coordinates and cluster/cell-type annotations.
    sample_col : str, optional
        Column identifying each sample. Default is ``"Sample"``.
    cluster_col : str, optional
        Column containing cell-type or cluster labels. Default is
        ``"Cell Type"``.
    x : str, optional
        Column name for x-coordinates. Default is ``"x"``.
    y : str, optional
        Column name for y-coordinates. Default is ``"y"``.
    color_dict : dict, optional
        Dictionary mapping cell-type labels to colours.  When None, seaborn
        default colours are used. Default is None.
    s : float, optional
        Marker size. When None the style default is used. Default is None.
    alpha : float, optional
        Marker transparency. When None the style default is used.
        Default is None.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (15, 12).
    style : SPACEcStyle or None, optional
        SPACEc style configuration object. Default is None.

    Returns
    -------
    None
    """
    _style = _apply_style(style)
    if s is None:
        s = _style.scatter_size_cell
    if alpha is None:
        alpha = _style.data_alpha_spatial
    for i in df[sample_col].unique():
        df_sub = df[df[sample_col] == i]
        print(i)
        celltype = list(df_sub[cluster_col].unique())
        ncols = 4
        nrows = len(celltype) // ncols + (len(celltype) % ncols > 0)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        plt.subplots_adjust(hspace=0.5)
        for ct, ax in zip(celltype, axs.ravel()):
            df_tmp = df_sub[df_sub[cluster_col] == ct]
            if color_dict == None:
                sns.scatterplot(
                    x=x, y=y, data=df_tmp, hue=cluster_col, s=s, alpha=alpha, ax=ax
                )
            else:
                sns.scatterplot(
                    x=x,
                    y=y,
                    data=df_tmp,
                    hue=cluster_col,
                    s=s,
                    alpha=alpha,
                    ax=ax,
                    palette=color_dict,
                )
            ax.set_title(ct.upper())
            ax.invert_yaxis()
            ax.get_legend().remove()
            ax.set_xlabel("")
        plt.show()


#########


def pl_visualize_2D_density_plot(
    df,
    region_column,
    selected_region,
    subsetting_column,
    values_list,
    x_column,
    y_column,
    style=None,
):
    """
    Visualise a 2-D kernel-density estimate overlaid with a scatter plot.

    Subsets the data to a single region and a subset of rows defined by
    ``subsetting_column``/``values_list``, then plots a KDE density surface
    with individual cell positions overlaid as a scatter plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame containing spatial coordinates, region labels and
        the subsetting column.
    region_column : str
        Column used to select a single tissue region.
    selected_region : str
        Value in ``region_column`` that identifies the region to display.
    subsetting_column : str
        Column used to further filter the cells shown in the density layer.
    values_list : list
        Values in ``subsetting_column`` that are included in the density
        layer.
    x_column : str
        Column name for the x-coordinate.
    y_column : str
        Column name for the y-coordinate.
    style : SPACEcStyle or None, optional
        SPACEc style configuration object. Default is None.

    Returns
    -------
    None
    """
    _style = _apply_style(style)
    # Subset the DataFrame based on region_column and selected_region
    subset_df1 = df[df[region_column] == selected_region]

    # Subset the DataFrame based on subsetting_column and values_list
    subset_df2 = subset_df1[subset_df[subsetting_column].isin(values_list)]

    # Create a 2D density plot
    sns.kdeplot(data=subset_df2, x=x_column, y=y_column, fill=True)

    # Overlay the individual data points as a scatter plot
    sns.scatterplot(
        data=subset_df1,
        x=x_column,
        y=y_column,
        color=_style.network_node_default_color,
        alpha=_style.data_alpha_background,
    )

    # Add labels and title to the plot
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title("2D Density Plot with Overlay")

    # Display the plot
    plt.show()


#######


def cluster_composition_heatmap(
    data,
    cluster_column,
    celltype_column,
    normalize=False,
    figsize=(20, 6),
    annot=True,
    fmt=None,
    savefig=False,
    output_fname="",
    output_dir="./",
    return_data=False,
    style=None,
):
    """
    Plot a heatmap of cell-type composition per cluster.

    Computes a cross-tabulation of cluster labels against cell-type labels
    and displays it as an annotated heatmap — the canonical QC figure for
    checking whether unsupervised clusters (e.g. from
    :func:`spacec.tools.clustering`) align with known cell types.

    Parameters
    ----------
    data : pandas.DataFrame or anndata.AnnData
        Input data containing both cluster and cell-type annotations. If an
        ``AnnData`` is passed, ``adata.obs`` is used.
    cluster_column : str
        Column name containing cluster identifiers (rows of the heatmap).
    celltype_column : str
        Column name containing cell-type labels (columns of the heatmap).
    normalize : bool, optional
        If True, show each cluster's row as a percentage of that cluster's
        total cell count instead of raw counts. Defaults to False.
    figsize : tuple, optional
        Figure size. Defaults to ``(20, 6)``.
    annot : bool, optional
        Print the value in every cell (text size = style annotation size). Set to
        False for large matrices whose cells are too small. Defaults to True.
    fmt : str, optional
        Number format of the cell values, e.g. ``".0f"`` for integers (shorter numbers fit
        small cells). Defaults to ``".1f"`` (normalized) or ``"d"`` (counts).
    savefig : bool, optional
        Whether to save the figure. Defaults to False.
    output_fname : str, optional
        Base filename (without extension) used when ``savefig=True``.
    output_dir : str, optional
        Directory to save into when ``savefig=True``. Defaults to ``"./"``.
    return_data : bool, optional
        If True, also return the underlying cross-tabulation DataFrame
        (source data for publication). Defaults to False.
    style : SPACEcStyle, str, or pathlib.Path, optional
        Style to apply for this call. Defaults to the active style.

    Returns
    -------
    matplotlib.axes.Axes or (matplotlib.axes.Axes, pandas.DataFrame)
        The heatmap axes, and — if ``return_data=True`` — the frequency
        table used to draw it.
    """
    _style = _apply_style(style)
    df = data.obs if hasattr(data, "obs") else data

    frequency_table = pd.crosstab(df[cluster_column], df[celltype_column])
    if normalize:
        frequency_table = frequency_table.div(frequency_table.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        frequency_table,
        cmap=_style.sequential_cmap,
        annot=annot,
        annot_kws={"size": _style.font_size_annotation},
        fmt=fmt if fmt is not None else (".1f" if normalize else "d"),
        ax=ax,
        cbar_kws={"label": "% of cluster" if normalize else "cell count"},
    )
    ax.set_title("Cluster × Cell Type Composition")
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Cluster")
    _apply_spines(ax, _style)

    if savefig:
        _savefig(fig, output_dir, output_fname, "_cluster_composition_heatmap", _style)

    if return_data:
        return ax, frequency_table
    return ax


def _categorical_mask_rgba(masks, labels_arr, colors_arr, mask_outline, _style):
    """Build an RGBA image filling each mask label with its row's exact color.

    ``labels_arr``/``colors_arr`` come from an exact ``label`` column join (each
    row already names its own mask-pixel label), not proximity/centroid matching.
    """
    color_map = np.zeros((masks.max() + 1, 4))
    color_map[labels_arr, :3] = colors_arr
    color_map[labels_arr, 3] = 1
    colored_masks = color_map[masks]

    if mask_outline:
        boundaries = find_boundaries(masks, mode="outer") & np.isin(masks, labels_arr)
        colored_masks[boundaries, :3] = mcolors.to_rgb(_style.edge_color_default)
        colored_masks[boundaries, 3] = 1

    return colored_masks


def _continuous_mask_image(masks, labels_arr, values_arr):
    """Build a masked scalar image filling each mask label with its row's value.

    The scalar counterpart to :func:`_categorical_mask_rgba`: ``labels_arr``
    names each row's own mask-pixel label (an exact ``label`` column join, not
    proximity matching), so ``values_arr[i]`` is painted over every pixel of
    label ``labels_arr[i]``.

    Labels with no matching row -- background (label 0) and cells removed
    upstream by ``filter_data``/``remove_noise`` -- stay masked, so a colormap's
    "bad" color shows through for them.
    """
    if len(labels_arr) and (labels_arr.min() < 0 or labels_arr.max() > masks.max()):
        raise ValueError(
            f"label values out of range for this mask (labels "
            f"{labels_arr.min()}--{labels_arr.max()}, mask max {masks.max()}) "
            "-- check that the observations and the mask come from the same "
            "region."
        )

    value_by_label = np.full(masks.max() + 1, np.nan, dtype=float)
    value_by_label[labels_arr] = values_arr
    value_by_label[0] = np.nan  # label 0 is background, never a cell
    return np.ma.masked_invalid(value_by_label[masks])


def _render_catplot_panel(
    ax,
    panel_data,
    masks_dict,
    region_col,
    label_col,
    X,
    Y,
    color,
    palette,
    mask_outline,
    size,
    alpha,
    region_gap,
    _style,
):
    """Render one catplot panel, tiling each real region side by side.

    Each region present in ``masks_dict`` (keyed by ``region_col``, matching
    :func:`spacec.hf.hf_attach_masks`'s convention) is filled via an exact
    ``label_col`` join; regions with no matching mask fall back to a dot
    scatter. Regions are laid out left-to-right so multiple real regions
    sharing one panel value (e.g. several images under one "condition") don't
    overlap. Returns False if masks aren't usable at all here (caller should
    fall back to a plain scatterplot for the whole panel), True otherwise.
    """
    if (
        masks_dict is None
        or region_col not in panel_data.columns
        or label_col not in panel_data.columns
    ):
        return False

    # Every lookup below matches against `color`'s values cast to str (region/label joins
    # are string-keyed too) -- str-key the palette once here so this still matches even
    # when `color`'s native dtype isn't str (e.g. an integer cluster id), without forcing
    # every OTHER `catplot` palette consumer (the plain sns.scatterplot fallback below,
    # which matches on `color`'s own native dtype) onto the same convention. Also normalize
    # every value to an RGB tuple via mcolors.to_rgb -- the mask-fill branch below does
    # `[:3]` on each color to drop an alpha channel, which silently mis-slices a hex string
    # (e.g. "#1f77b4"[:3] == "#1f", 3 characters, not 3 numbers) or a plain color name (e.g.
    # the "grey" _resolve_categorical_palette falls back to for an uncovered category) --
    # both real inputs (a user's own hex-coded palette dict, or an incomplete one), not just
    # a hypothetical. to_rgb already accepts a tuple as-is, so this is a no-op for those.
    palette = {str(k): mcolors.to_rgb(v) for k, v in palette.items()}
    region_values = sorted(panel_data[region_col].astype(str).unique())
    offset = 0.0
    max_height = 0.0

    for region_value in region_values:
        group = panel_data[panel_data[region_col].astype(str) == region_value]

        if region_value in masks_dict:
            masks = masks_dict[region_value]
            labels_arr = group[label_col].to_numpy().astype(int)
            colors_arr = np.array(
                [
                    mcolors.to_rgb(palette.get(ct, (0, 0, 0)))
                    for ct in group[color].astype(str)
                ]
            )
            colored_masks = _categorical_mask_rgba(
                masks, labels_arr, colors_arr, mask_outline, _style
            )
            H, W = masks.shape
            ax.imshow(colored_masks, origin="lower", extent=(offset, offset + W, 0, H))
            offset += W * (1 + region_gap)
            max_height = max(max_height, H)
        else:
            # group[Y] already reflects invert_y, applied once upstream on the
            # whole df -- do not flip again here.
            x_vals = group[X].to_numpy(dtype=float)
            y_vals = group[Y].to_numpy(dtype=float)
            width = (x_vals.max() - x_vals.min()) if len(x_vals) else 0.0
            height = (y_vals.max() - y_vals.min()) if len(y_vals) else 0.0
            x_shift = offset - (x_vals.min() if len(x_vals) else 0.0)
            y_shift = -(y_vals.min() if len(y_vals) else 0.0)
            colors_arr = [palette.get(ct, (0, 0, 0)) for ct in group[color].astype(str)]
            ax.scatter(
                x_vals + x_shift, y_vals + y_shift, c=colors_arr, s=size, alpha=alpha
            )
            offset += width * (1 + region_gap)
            max_height = max(max_height, height)

    ax.set_xlim(0, offset if offset > 0 else 1)
    ax.set_ylim(0, max_height if max_height > 0 else 1)
    # the legend is drawn once per figure by catplot
    return True


def catplot(
    adata,
    color,
    unique_region,
    subset=None,
    X="x",
    Y="y",
    invert_y=False,
    size=6,
    alpha=1,
    palette=None,  # default is None which means the color comes from the anndata object
    savefig=False,
    output_dir="./",
    output_fname="",
    figsize=5,
    background=None,
    style=None,
    axis="on",
    scatter_kws={},
    n_columns=4,
    mask_outline=True,
    region_col="unique_region",
    label_col="label",
    region_gap=0.05,
):
    """
    Plots cells in tissue section color coded by either cell type or node allocation.

    ``unique_region`` only controls how cells are split into subplot panels (it is
    often a coarse grouping like "condition", spanning several real images). Mask
    lookup is resolved separately via ``region_col``, matching
    :func:`spacec.hf.hf_attach_masks`'s own convention: for each real region named
    in ``adata.obs[region_col]`` that has a matching key in ``adata.uns["masks"]``,
    cells are rendered as their filled segmentation-mask shape (via an exact
    ``label_col`` join, not proximity/centroid matching) instead of a dot, colored
    by ``color``. If a panel contains more than one real region, each is tiled
    side by side (never overlaid) so they don't collide. Real regions with no
    matching mask fall back to a dot scatter, tiled the same way. If masks aren't
    available at all (``adata.uns`` has no ``"masks"`` key, or ``region_col``/
    ``label_col`` aren't present), the whole panel falls back to today's plain
    dot scatter. ``invert_y`` only affects dot rendering -- mask fill always uses
    the mask array's true pixel orientation, since the mask itself is never
    flipped or shifted (only its on-screen placement, via imshow's extent).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    color : str
        Color by "Clusterid" or "Node" respectively.
    unique_region : str
        Column used to split cells into subplot panels. Each panel is one
        independent CODEX image, or a coarser grouping (e.g. "condition") that
        may span several real images -- see ``region_col`` for mask lookup.
    subset : str, optional
        Subset of data to plot. If None, all data is plotted.
    X : str, optional
        Column name for x-axis in the DataFrame.
    Y : str, optional
        Column name for y-axis in the DataFrame.
    invert_y : bool, optional
        If True, invert y-axis. Only applies to regions without a mask (see above).
    size : int, optional
        Size of point to plot for each cell.
    alpha : float, optional
        Transparency of points.
    palette : dict or str, optional
        A ``{category: color}`` dict -- any category of `color` it doesn't cover is colored
        grey, with a warning, rather than requiring an exhaustive mapping -- or a matplotlib
        colormap name to sample evenly across `color`'s categories. If left as the default
        `None`, resolved via the shared `_resolve_categorical_palette` system (same as
        `extended_dotplot`): reuse/create a persistent mapping for every value of
        `adata.obs[color]`, cached in `adata.uns[color + "_colors"]`, so this matches whatever
        another function already established -- or will later reuse -- for the same column
        (e.g. a prior `sc.pl.umap` call, or `extended_dotplot`'s own `palette="auto"`).
    savefig : bool, optional
        If True, save figure.
    output_dir : str, optional
        Directory to save figure.
    output_fname : str, optional
        Filename to save figure.
    figsize : int, optional
        Size of the figure.
    background : str, optional
        Facecolor for each panel's axes. Defaults to `None`, which resolves to the active
        style's `_style.background_color` (so it follows `style=` -- e.g. dark under
        `SPACEcStyle.tissue()` -- instead of always being white regardless of style).
    style : str, optional
        Style of the plot.
    axis : str, optional
        If "off", axis is not displayed.
    scatter_kws : dict, optional
        Additional keyword arguments to pass to `scatterplot()`.
    n_columns : int, optional
        Number of columns in the figure.
    mask_outline : bool, optional
        Whether to draw a thin outline around each filled cell mask, for regions
        rendered via ``adata.uns["masks"]``. Defaults to True.
    region_col : str, optional
        Column in ``adata.obs`` naming each row's real per-image region -- the
        key into ``adata.uns["masks"]`` and the grouping used for the exact
        ``label_col`` join, independent of ``unique_region``. Matches
        :func:`spacec.hf.hf_attach_masks`'s default. Defaults to ``"unique_region"``.
    label_col : str, optional
        Column in ``adata.obs`` holding each row's exact mask-pixel label.
        Defaults to ``"label"``.
    region_gap : float, optional
        Horizontal gap between adjacent region tiles within one panel, as a
        fraction of the preceding tile's own width. Defaults to 0.05.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (also saved to `output_dir` if `savefig=True`).
    """
    import warnings as _warnings

    # backwards compat: old API passed a colour string as style= for background
    if isinstance(style, str) and not pathlib.Path(str(style)).exists():
        _warnings.warn(
            "Passing a string to 'style' for background colour is deprecated. "
            "Use 'background=' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        background = style
        style = None
    _style = _apply_style(style)
    if background is None:
        background = _style.background_color

    scatter_kws_ = {"s": size, "alpha": alpha}
    scatter_kws_.update(scatter_kws)

    # de-duplicate column selection: color and unique_region may reference
    # the same obs column, and duplicate labels would make df[color] /
    # df[unique_region] return a DataFrame instead of a Series below.
    cols = [X, Y, color, unique_region]
    # region_col/label_col are only needed for mask lookup -- include them
    # when present so the mask branch can see them; their absence is exactly
    # the signal _render_catplot_panel uses to fall back to plain dots.
    if region_col in adata.obs.columns:
        cols.append(region_col)
    if label_col in adata.obs.columns:
        cols.append(label_col)
    cols = list(dict.fromkeys(cols))
    df = pd.DataFrame(adata.obs[cols])

    df[color] = df[color].astype("category")
    if invert_y:
        df[Y] *= -1

    # df[color].cat.categories is sorted and NaN-free by construction (a pandas categorical
    # never stores NaN as a category) -- unlike the old np.sort(adata.obs[color].unique()),
    # which raised "'<' not supported between instances of 'float' and 'str'" the moment
    # `color` had any missing values (e.g. an incompletely-mapped annotation dict).
    categories = list(df[color].cat.categories)
    # _resolve_categorical_palette keys the None-palette case to adata.obs[color]'s FULL
    # value set (sorted), cached in adata.uns[color + "_colors"] -- so this matches whatever
    # a prior extended_dotplot/patch_QC call on the same `color` already established for this
    # annotation round, by name, regardless of any dendrogram/display reordering those use
    # internally. A dict palette gets any category it doesn't cover colored grey, with a
    # warning, instead of silently requiring an exhaustive mapping.
    palette = _resolve_categorical_palette(
        _style, adata, color, categories, palette=palette
    )

    if subset is None:
        region_list = list(
            pd.Series(df[unique_region].unique()).sort_values()
        )  # display all experiments
    else:
        if subset not in list(pd.Series(df[unique_region].unique()).sort_values()):
            print(subset + " is not in unique_region!")
            return
        else:
            region_list = [subset]

    n_rows = int(np.ceil(len(region_list) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(figsize * n_columns, figsize * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 0.25, "hspace": 0.4},
    )

    masks_dict = adata.uns.get("masks")
    if masks_dict is not None:
        masks_dict = {str(k): v for k, v in masks_dict.items()}

    for i_ax, (name, ax) in enumerate(zip(region_list, axes.flatten())):
        data = df[df[unique_region] == name]

        rendered = _render_catplot_panel(
            ax,
            data,
            masks_dict,
            region_col,
            label_col,
            X,
            Y,
            color,
            palette,
            mask_outline,
            size,
            alpha,
            region_gap,
            _style,
        )
        if not rendered:
            sns.scatterplot(
                x=X,
                y=Y,
                data=data,
                hue=color,
                palette=palette,
                ax=ax,
                legend=False,
                **scatter_kws_,  # s, alpha + caller's scatter_kws
            )

        ax.set_facecolor(background)
        ax.grid(False)
        if axis == "off":
            ax.axis("off")

        ax.set_title(name, fontsize=_style.font_size_medium)
        ax.set_aspect("equal")

    for i in range(i_ax + 1, n_rows * n_columns):
        axes.flatten()[i].axis("off")

    # one legend for the whole figure: colour patches stay visible at any point size
    # and the legend never runs into a neighbouring panel
    if isinstance(palette, dict):
        present = set(df.loc[df[unique_region].isin(region_list), color].astype(str))
        fig.legend(
            handles=[
                mpatches.Patch(color=c, label=str(k))
                for k, c in palette.items()
                if str(k) in present
            ],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            title=color,
            fontsize=_style.font_size_small,
            frameon=_style.legend_frameon,
        )

    if savefig:
        _savefig(fig, output_dir, output_fname, "_spatial_plot", _style)

    return fig


def _outline_umap_cluster(adata, ax, col, cluster, edgecolor, linewidth, base_size):
    """Draw an edge-only scatter over `adata.obs[col] == cluster` cells, on top of an
    already-drawn UMAP axes.

    Leaves every point's fill color untouched -- this only adds a ring around the
    highlighted cluster's cells (matching `sc.pl.umap`'s own default point size, so the ring
    sits right at each dot's edge), it doesn't greyscale or recolor anything else. No-op if
    `col` is missing/None or nothing matches `cluster`.
    """
    if col is None or col not in adata.obs or "X_umap" not in adata.obsm:
        return
    mask = (adata.obs[col].astype(str) == str(cluster)).to_numpy()
    if not mask.any():
        return
    ax_obj = ax[0] if isinstance(ax, list) else ax
    xy = adata.obsm["X_umap"][mask, :2]
    size = base_size if base_size is not None else 120000 / adata.n_obs
    ax_obj.scatter(
        xy[:, 0],
        xy[:, 1],
        s=size,
        facecolors="none",
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=5,
    )


def umap(
    adata,
    color=None,
    palette=None,
    figsize=None,
    savefig=False,
    output_fname="",
    output_dir="./",
    style=None,
    highlight_cluster=None,
    highlight_col=None,
    highlight_color=None,
    highlight_linewidth=2.0,
    **kwargs,
):
    """
    Plot a UMAP embedding using the active SPACEc style.

    A thin, style-aware wrapper around :func:`scanpy.pl.umap`. Reads the
    embedding from ``adata.obsm["X_umap"]`` (written by
    :func:`spacec.tools.clustering`) and colors cells by ``color`` using the
    cached palette in ``adata.uns[color + "_colors"]`` when available,
    falling back to the active style's categorical palette.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with a precomputed UMAP embedding.
    color : str or list of str, optional
        Column(s) in ``adata.obs`` (or var name) to color cells by.
    palette : list or dict, optional
        Explicit palette to use. Defaults to the cached/style palette.
    figsize : tuple, optional
        Figure size. Defaults to the style's single-column figure size.
    savefig : bool, optional
        Whether to save the figure. Defaults to False.
    output_fname : str, optional
        Base filename (without extension) used when ``savefig=True``.
    output_dir : str, optional
        Directory to save into when ``savefig=True``. Defaults to ``"./"``.
    style : SPACEcStyle, str, or pathlib.Path, optional
        Style to apply for this call. Defaults to the active style.
    highlight_cluster : str, optional
        A value of ``highlight_col`` (or ``color``, if ``highlight_col`` is None) whose cells
        get an open-circle outline drawn on top of the normal scatter, without changing their
        fill color -- e.g. to spot-check where one cluster sits on the embedding. No-op if
        None (default) or if nothing matches. By default None.
    highlight_col : str, optional
        obs column to match ``highlight_cluster`` against. Defaults to ``color``.
    highlight_color : str, optional
        Outline color. Defaults to the active style's ``accent_color_negative``.
    highlight_linewidth : float, optional
        Outline width in points. By default 2.0.
    **kwargs
        Passed through to :func:`scanpy.pl.umap`.

    Returns
    -------
    matplotlib.axes.Axes or list of matplotlib.axes.Axes
        The axes produced by ``scanpy.pl.umap``.
    """
    import scanpy as sc

    _style = _apply_style(style)
    if figsize is None:
        figsize = (_style.figure_width_single, _style.figure_width_single)

    if palette is None and color is not None and isinstance(color, str):
        n = adata.obs[color].nunique() if color in adata.obs else None
        if n is not None:
            palette = _get_palette(_style, n, adata=adata, col=color)

    with _style.context():
        plt.rcParams["figure.figsize"] = figsize
        ax = sc.pl.umap(adata, color=color, palette=palette, show=False, **kwargs)
        fig = ax[0].figure if isinstance(ax, list) else ax.figure
        for a in ax if isinstance(ax, list) else [ax]:
            _apply_spines(a, _style)
        if highlight_cluster is not None:
            _outline_umap_cluster(
                adata,
                ax,
                highlight_col or color,
                highlight_cluster,
                highlight_color or _style.accent_color_negative,
                highlight_linewidth,
                base_size=kwargs.get("size"),
            )
        if savefig:
            _savefig(fig, output_dir, output_fname, "_umap", _style)
        return ax


def marker_dotplot(
    adata,
    markers,
    groupby,
    dendrogram=True,
    expr_threshold=0.0,
    show_cluster_size=True,
    min_cluster_frac=0.01,
    highlight_cluster=None,
    spatial_region=None,
    region_col="unique_region",
    x_col="x",
    y_col="y",
    spatial_point_size=5.0,
    spatial_grey_alpha=0.2,
    spatial_color_alpha=0.8,
    figsize=None,
    savefig=False,
    output_fname="",
    output_dir="./",
    style=None,
):
    """
    Plot a marker-expression dot plot with per-cluster size context.

    ``highlight_cluster`` may be either a single cluster label, e.g. ``"10"``,
    or a list/tuple/set of labels, e.g. ``["10", "2"]``. If ``spatial_region``
    is also given, all highlighted clusters are shown in the spatial panel,
    each in its own cluster color.
    """
    from scipy.cluster.hierarchy import dendrogram as _sch_dendrogram
    from scipy.cluster.hierarchy import linkage as _sch_linkage

    # Normalize highlight_cluster to a list while preserving backwards compatibility.
    if highlight_cluster is None:
        highlight_clusters = []
    elif isinstance(highlight_cluster, (list, tuple, set, np.ndarray, pd.Index)):
        highlight_clusters = list(highlight_cluster)
    else:
        highlight_clusters = [highlight_cluster]

    if spatial_region is not None and len(highlight_clusters) == 0:
        raise ValueError("spatial_region requires highlight_cluster to also be set.")

    _style = _apply_style(style)

    markers = list(markers)
    X = adata[:, markers].X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    expr = pd.DataFrame(X, index=adata.obs_names, columns=markers)
    groups = adata.obs[groupby]

    mean_expr = expr.groupby(groups, observed=True).mean()
    frac_expr = expr.groupby(groups, observed=True).apply(
        lambda g: (g > expr_threshold).mean()
    )

    sizes = groups.value_counts().reindex(mean_expr.index)
    frac_of_total = sizes / sizes.sum()

    # z-score each marker across groups for the dot color scale
    col_std = mean_expr.std(axis=0).replace(0, 1)
    z_expr = (mean_expr - mean_expr.mean(axis=0)) / col_std
    z_expr = z_expr.clip(lower=_style.heatmap_vmin, upper=_style.heatmap_vmax)

    group_order = list(mean_expr.index)
    linkage_matrix = None

    if dendrogram and len(group_order) > 2:
        linkage_matrix = _sch_linkage(z_expr.values, method="average")
        leaves = _sch_dendrogram(linkage_matrix, no_plot=True)["leaves"]
        group_order = [mean_expr.index[i] for i in leaves]

    mean_expr = mean_expr.loc[group_order]
    frac_expr = frac_expr.loc[group_order]
    z_expr = z_expr.loc[group_order]
    sizes = sizes.loc[group_order]
    frac_of_total = frac_of_total.loc[group_order]

    n_groups = len(group_order)
    n_markers = len(markers)

    if isinstance(groups.dtype, pd.CategoricalDtype):
        categories = list(groups.cat.categories)
    else:
        categories = sorted(groups.unique().tolist())

    palette = _get_palette(_style, len(categories), adata=adata, col=groupby)
    color_map = dict(zip(categories, palette))

    # String-based color lookup is important because cluster labels are often
    # passed as strings, while adata.obs[groupby] may store them as categorical,
    # integer, or mixed labels.
    color_map_str = {str(k): v for k, v in color_map.items()}

    def _cluster_color(label, fallback=None):
        if label in color_map:
            return color_map[label]
        if str(label) in color_map_str:
            return color_map_str[str(label)]
        if fallback is not None:
            return fallback
        return _style.accent_color_positive

    if figsize is None:
        base_width = max(_style.figure_width_double, 0.3 * n_markers + 2)

        if spatial_region is not None:
            base_width += 0.6 * _style.figure_width_single

        main_height_in = max(2.0, 0.25 * n_groups + 1)
    else:
        base_width, main_height_in = figsize

    # Dedicated legend row.
    legend_height_in = 0.9

    # Dedicated spacer row for rotated x-axis labels.
    max_marker_len = max((len(str(m)) for m in markers), default=0)
    xtick_pad_in = max(
        0.7,
        min(
            2.2,
            0.008 * _style.font_size_small * max_marker_len + 0.35,
        ),
    )

    figsize = (base_width, main_height_in + xtick_pad_in + legend_height_in)

    layout = []

    if linkage_matrix is not None:
        layout.append(("dendrogram", 0.15))
        layout.append((None, 0.09))

    layout.append(("dots", 1.0))
    layout.append((None, 0.03))
    layout.append(("colorbar", 0.05))
    layout.append((None, 0.12))

    if show_cluster_size:
        layout.append(("size", 0.25))

    if spatial_region is not None:
        layout.append((None, 0.1))
        layout.append(("spatial", 0.6))

    width_ratios = [w for _, w in layout]

    col_of = {}
    for i, (name, _) in enumerate(layout):
        if name is not None:
            col_of[name] = i

    with _style.context():
        fig = plt.figure(figsize=figsize)

        gs = fig.add_gridspec(
            3,
            len(layout),
            width_ratios=width_ratios,
            height_ratios=[main_height_in, xtick_pad_in, legend_height_in],
            wspace=0.02,
            hspace=0.0,
        )

        ax_dots = fig.add_subplot(gs[0, col_of["dots"]])
        axes_by_name = {"dots": ax_dots}

        if "dendrogram" in col_of:
            axes_by_name["dendrogram"] = fig.add_subplot(gs[0, col_of["dendrogram"]])

        if "size" in col_of:
            axes_by_name["size"] = fig.add_subplot(
                gs[0, col_of["size"]],
                sharey=ax_dots,
            )

        if "spatial" in col_of:
            axes_by_name["spatial"] = fig.add_subplot(gs[0, col_of["spatial"]])

        ax_cbar = fig.add_subplot(gs[0, col_of["colorbar"]])
        ax_legend = fig.add_subplot(gs[2, col_of["dots"]])
        ax_legend.axis("off")

        y_pos = np.arange(n_groups)

        if "dendrogram" in axes_by_name:
            ax_d = axes_by_name["dendrogram"]

            _sch_dendrogram(
                linkage_matrix,
                orientation="left",
                no_labels=True,
                ax=ax_d,
                color_threshold=0,
                above_threshold_color=_style.accent_color_neutral,
            )

            ax_d.set_ylim(0, 10 * n_groups)
            ax_d.axis("off")

        ax_dots = axes_by_name["dots"]

        cmap_obj = plt.get_cmap(_style.diverging_cmap)
        norm = mcolors.Normalize(
            vmin=_style.heatmap_vmin,
            vmax=_style.heatmap_vmax,
        )

        max_dot_area = 200.0

        for yi, group in enumerate(group_order):
            for xi, marker in enumerate(markers):
                frac = frac_expr.loc[group, marker]
                val = z_expr.loc[group, marker]

                ax_dots.scatter(
                    xi,
                    yi,
                    s=max(frac, 0.02) * max_dot_area,
                    color=cmap_obj(norm(val)),
                    edgecolors=_style.edge_color_default,
                    linewidths=0.3,
                )

        ax_dots.set_xlim(-0.5, n_markers - 0.5)
        ax_dots.set_ylim(-0.5, n_groups - 0.5)

        ax_dots.set_xticks(range(n_markers))
        ax_dots.set_xticklabels(
            markers,
            rotation=90,
            fontsize=_style.font_size_small,
            ha="center",
            va="top",
        )
        ax_dots.tick_params(axis="x", pad=1)

        ax_dots.set_yticks(y_pos)
        ax_dots.tick_params(axis="y", labelleft=True)
        ax_dots.set_yticklabels(
            [str(g) for g in group_order],
            fontsize=_style.font_size_small,
        )

        for tick, group in zip(ax_dots.get_yticklabels(), group_order):
            tick.set_color(_cluster_color(group, _style.accent_color_neutral))

        _apply_spines(ax_dots, _style)

        # Highlight one or more cluster rows in their own colors.
        group_order_str = [str(g) for g in group_order]

        for cluster in highlight_clusters:
            cluster_str = str(cluster)

            if cluster_str in group_order_str:
                y_idx = group_order_str.index(cluster_str)
                highlight_color = _cluster_color(
                    cluster,
                    fallback=_style.accent_color_positive,
                )

                ax_dots.axhspan(
                    y_idx - 0.5,
                    y_idx + 0.5,
                    color=highlight_color,
                    alpha=0.15,
                    zorder=0,
                )

        # Dot-size legend.
        legend_fracs = [0.25, 0.5, 0.75, 1.0]

        legend_handles = [
            ax_legend.scatter(
                [],
                [],
                s=f * max_dot_area,
                color=_style.accent_color_neutral,
                edgecolors=_style.edge_color_default,
                linewidths=0.3,
            )
            for f in legend_fracs
        ]

        ax_legend.legend(
            legend_handles,
            [f"{int(f * 100)}%" for f in legend_fracs],
            title="fraction expressing",
            loc="center",
            ncol=len(legend_fracs),
        )

        _style_legend(ax_legend, _style)

        # Colorbar.
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])

        _style_colorbar(
            fig.colorbar(
                sm,
                cax=ax_cbar,
                label="mean expression (z-score)",
            ),
            _style,
        )

        ax_size = axes_by_name.get("size")

        if ax_size is not None:
            bar_colors = [
                (
                    _style.accent_color_negative
                    if f < min_cluster_frac
                    else _style.accent_color_neutral
                )
                for f in frac_of_total
            ]

            ax_size.barh(
                y_pos,
                sizes.values,
                color=bar_colors,
                height=0.6,
            )

            ax_size.set_xlabel(
                "n cells",
                fontsize=_style.font_size_small,
            )

            ax_size.tick_params(
                axis="both",
                labelsize=_style.font_size_tiny,
            )

            ax_size.tick_params(
                axis="y",
                labelleft=False,
            )

            _apply_spines(ax_size, _style)

        ax_spatial = axes_by_name.get("spatial")

        if ax_spatial is not None:
            region_mask = adata.obs[region_col].astype(str) == str(spatial_region)
            region_df = adata.obs.loc[region_mask]

            cluster_labels_str = [str(c) for c in highlight_clusters]
            region_group_str = region_df[groupby].astype(str)
            highlighted_mask = region_group_str.isin(cluster_labels_str)

            # Grey background: all cells in the region except highlighted clusters.
            ax_spatial.scatter(
                region_df.loc[~highlighted_mask, x_col],
                region_df.loc[~highlighted_mask, y_col],
                c="#CCCCCC",
                s=spatial_point_size,
                alpha=spatial_grey_alpha,
                linewidths=0,
                rasterized=True,
            )

            # Draw each highlighted cluster separately in its own color.
            spatial_counts = {}

            for cluster in highlight_clusters:
                cluster_str = str(cluster)
                cluster_mask = region_group_str == cluster_str
                spatial_counts[cluster_str] = int(cluster_mask.sum())

                if cluster_mask.any():
                    ax_spatial.scatter(
                        region_df.loc[cluster_mask, x_col],
                        region_df.loc[cluster_mask, y_col],
                        color=[_cluster_color(cluster, _style.accent_color_positive)],
                        s=spatial_point_size,
                        alpha=spatial_color_alpha,
                        linewidths=0,
                        rasterized=True,
                        label=cluster_str,
                    )

            if len(highlight_clusters) == 1:
                title_cluster = str(highlight_clusters[0])
                title_n = spatial_counts.get(title_cluster, 0)
                spatial_title = f"{title_cluster} — {spatial_region} " f"(n={title_n})"
            else:
                count_text = ", ".join(
                    f"{c}: n={spatial_counts.get(str(c), 0)}"
                    for c in highlight_clusters
                )
                spatial_title = (
                    f"{', '.join(map(str, highlight_clusters))} — "
                    f"{spatial_region} ({count_text})"
                )

            ax_spatial.set_title(
                spatial_title,
                fontsize=_style.font_size_small,
            )

            ax_spatial.set_xlabel(
                x_col,
                fontsize=_style.font_size_small,
            )

            ax_spatial.set_ylabel(
                y_col,
                fontsize=_style.font_size_small,
            )

            ax_spatial.invert_yaxis()
            ax_spatial.set_aspect("equal", adjustable="datalim")

            ax_spatial.tick_params(
                axis="both",
                labelsize=_style.font_size_tiny,
            )

            if len(highlight_clusters) > 1:
                ax_spatial.legend(
                    title=groupby,
                    loc="best",
                    fontsize=_style.font_size_tiny,
                    title_fontsize=_style.font_size_tiny,
                    frameon=False,
                )

            _apply_spines(ax_spatial, _style)

        if savefig:
            _savefig(
                fig,
                output_dir,
                output_fname,
                "_marker_dotplot",
                _style,
            )

        plt.show()

        return {
            "dots": ax_dots,
            "size": axes_by_name.get("size"),
            "dendrogram": axes_by_name.get("dendrogram"),
            "spatial": ax_spatial,
            "colorbar": ax_cbar,
            "legend": ax_legend,
        }


def _to_dense_array(arr):
    """Convert sparse or array-like to a dense numpy array."""
    if hasattr(arr, "toarray"):
        return arr.toarray()
    return np.asarray(arr)


def _measure_text_width_inches(text, fontsize, dpi):
    """Actual rendered width of `text` at `fontsize`, in inches -- via a throwaway figure, since
    per-character width estimates vary too much between strings (measured 0.48-0.63 * fontsize/72
    per char across a handful of real title strings) to use a single fixed factor reliably.
    """
    _fig = plt.figure(dpi=dpi)
    _txt = _fig.text(0, 0, text, fontsize=fontsize)
    _fig.canvas.draw()
    _bbox = _txt.get_window_extent(renderer=_fig.canvas.get_renderer())
    _width_in = (_bbox.x1 - _bbox.x0) / dpi
    plt.close(_fig)
    return _width_in


def _is_light_color(color, threshold=0.7):
    """True if `color` is light enough that a black outline is needed to stay visible against
    white (e.g. a pale palette swatch used as a bar/marker fill). Perceived-brightness luminance
    (``0.299 R + 0.587 G + 0.114 B``, matching the standard weights for how sensitive the eye is
    to each channel), not a straight RGB average."""
    r, g, b = mcolors.to_rgb(color)
    return (0.299 * r + 0.587 * g + 0.114 * b) > threshold


def _wrap_label(label, max_chars=10):
    """Split a label longer than `max_chars` into two lines at the underscore/space closest to
    its middle (e.g. "qc_flag_low_dapi" -> "qc_flag\\nlow_dapi") -- under a 90-degree tick-label
    rotation the two lines render side by side rather than stacked, so this trades a modest bit
    of extra column width for a much shorter required label margin. Labels already short enough,
    or with no underscore/space to split on, are returned unchanged."""
    if max_chars is None or len(label) <= max_chars:
        return label
    split_positions = [i for i, ch in enumerate(label) if ch in "_ "]
    if not split_positions:
        return label
    mid = len(label) / 2
    best = min(split_positions, key=lambda i: abs(i - mid))
    return label[:best] + "\n" + label[best + 1 :]


def _get_clustering_report_markers(adata, groupby):
    """Look up the marker list `sp.tl.clustering` actually used to produce `adata.obs[groupby]`,
    per the most recent matching entry in `adata.uns["clustering_report"]` (see
    `spacec.tl.clustering`/`clustering_report`).

    Returns `None` if `adata.uns["clustering_report"]` is missing/unparseable, if no recorded
    step's `key_added` matches `groupby`, or if the matching step is a `subclustering` action
    (which doesn't record a marker list at all). `None` means "no record found" -- callers must
    not treat it as "recorded as empty".

    Never raises -- a failure here must not break whatever it's attached to, matching
    `_append_clustering_report`'s own defensive convention.
    """
    try:
        raw = adata.uns.get("clustering_report")
        if not raw:
            return None
        steps = json.loads(raw) if isinstance(raw, str) else list(raw)

        matching = [s for s in steps if s.get("key_added") == groupby]
        if not matching:
            return None
        step = matching[-1]  # most recent step that wrote this column
        if step.get("action") != "clustering":
            return None  # e.g. "subclustering" -- no marker_list recorded, nothing to look up

        recorded = step.get("marker_list")
        return list(recorded) if recorded else list(adata.var_names.astype(str))
    except Exception:
        return None


def _warn_on_clustering_marker_mismatch(adata, groupby, clustering_markers):
    """Warn if `clustering_markers` doesn't exactly match the marker panel `sp.tl.clustering`
    actually used to produce `adata.obs[groupby]` (via `_get_clustering_report_markers`).

    Only fires when a real, comparable record is found -- silently does nothing otherwise. A
    missing record is not evidence of a mismatch, so it must never be reported as one.

    Never raises -- a failure here must not break the plot it's attached to, matching
    `_append_clustering_report`'s own defensive convention.
    """
    try:
        recorded = _get_clustering_report_markers(adata, groupby)
        if recorded is None:
            return

        recorded_set = set(map(str, recorded))
        given_set = set(map(str, clustering_markers))

        if given_set == recorded_set:
            return
        if given_set < recorded_set:
            warnings.warn(
                f"clustering_markers ({len(given_set)} markers) is a subset of the "
                f"{len(recorded_set)} markers sp.tl.clustering actually used to build "
                f"adata.obs['{groupby}']. The dot pattern shown reflects only this narrower "
                "subset, and the dendrogram (if drawn) summarizes cluster similarity over just "
                "these markers, not the full panel actually used for clustering.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"clustering_markers does not match the {len(recorded_set)} markers "
                f"sp.tl.clustering actually used to build adata.obs['{groupby}']. The dot "
                "pattern shown reflects only the markers passed here, and the dendrogram (if "
                "drawn) summarizes cluster similarity over a different marker set than what "
                "actually produced this clustering.",
                stacklevel=2,
            )
    except Exception:
        pass


def extended_dotplot(
    adata,
    groupby,
    *,
    clustering_markers,
    extra_markers=None,
    obs_continuous=None,
    obs_flags=None,
    layer=None,
    expression_threshold=0.0,
    color_scale="zscore",
    obs_continuous_cmap="YlOrRd",
    obs_flags_cmap="Reds",
    max_dot_size=220.0,
    label_max_chars=10,
    annotate_obs=True,
    show_totals=True,
    flag_small_clusters=False,
    min_cluster_frac=0.01,
    cell_w_inches=0.52,
    row_h_inches=0.40,
    palette="auto",
    cluster_order=None,
    label_overrides=None,
    dendrogram=False,
    dendrogram_position="after_markers",
    dendrogram_linkage_method="complete",
    dendrogram_width_inches=0.9,
    highlight_cluster=None,
    spatial_region=None,
    region_col="unique_region",
    x_col="x",
    y_col="y",
    spatial_point_size=15.0,
    spatial_grey_alpha=0.2,
    spatial_color_alpha=0.8,
    spatial_width_inches=10.0,
    title=None,
    style=None,
    transpose=False,
    margin_left_inches=None,
    section_gap_inches=0.35,
    totals_width_inches=1.5,
    label_block_min_inches=1.3,
    label_block_inches=None,
):
    """
    Multi-section dotplot for cluster annotation, combining marker expression with QC signal.

    Extends a standard scanpy dotplot (marker dot color = per-marker min-max scaled mean
    expression, dot size = fraction of cells above `expression_threshold`) with independently
    colour-scaled heatmap sections for continuous obs features (e.g. morphology from
    `spacec.tl.add_morphology_features`) and binary flag obs features (e.g. QC flags from
    `spacec.pp.filter_data`/`spacec.pp.remove_noise`), plus an optional per-cluster cell-count bar
    chart -- so marker identity, shape/contact context, and QC flags can be read off in one view
    instead of switching between separate plots.

    Parameters
    ----------
    adata : AnnData
    groupby : str
        obs column containing cluster labels.
    clustering_markers : Sequence[str]
        var_names used for clustering -- shown in the left dot section. The dot pattern (and
        dendrogram, if drawn) always reflect exactly this marker list, whatever it is -- if it
        doesn't match the marker panel actually used to produce `adata.obs[groupby]`,
        `sp.tl.clustering`'s own record in `adata.uns["clustering_report"]` is checked and a
        `UserWarning` is raised on any mismatch (including a partial/subset match). Silent if
        no matching record exists (e.g. `groupby` came from `subclustering`, or wasn't produced
        via `sp.tl.clustering` at all) -- there's nothing to check it against in that case.
    extra_markers : Sequence[str], optional
        Additional var_names displayed in a second dot section (e.g. markers excluded from
        clustering this round).
    obs_continuous : Sequence[str], optional
        obs columns with continuous values (e.g. morphology features). Displayed as a
        mean-per-cluster heatmap with text annotations.
    obs_flags : Sequence[str], optional
        obs columns with boolean/binary values (e.g. QC flags). Displayed as a
        fraction-flagged-per-cluster heatmap.
    layer : str, optional
        adata.layers key for expression; defaults to adata.X.
    expression_threshold : float, optional
        Threshold for dot-size calculation: fraction of cells with value strictly above this
        value determines dot area. By default 0.0.
    color_scale : {"minmax", "zscore"}, optional
        How each `clustering_markers`/`extra_markers` column's per-cluster mean expression is
        turned into dot color:

        - ``"zscore"`` (default): standardize each marker's cluster means (subtract their mean,
          divide by their std across clusters), clipped to `style.heatmap_vmin`/`heatmap_vmax` and
          drawn on `style.diverging_cmap` -- matching `marker_dotplot`'s own coloring. The default
          because `sp.pp.format`'s own default `method="zscore"` already per-marker z-scores
          `adata.X` before clustering (see note below), so a cluster's raw mean here is already in
          per-cell-standardized units, comparable *across markers* without any further rescaling
          -- this re-standardizes across *clusters* on top of that, rather than discarding it.
        - ``"minmax"``: rescale each marker's cluster means to ``[0, 1]`` instead (its lowest
          cluster -> 0, its highest -> 1). Matches `scanpy.pl.dotplot(..., standard_scale="var")`.
          Every annotation call written before the default changed was built and read
          against this (`extended_dotplot`'s prior default), so pass this explicitly to reproduce
          an annotation dictionary written before `color_scale` defaulted to `"zscore"` -- switching
          an existing call to the new default may shift which cluster reads as "most positive" for
          a marker where no cluster is strongly enriched.

        Note for this pipeline specifically: min-max always stretches *some* cluster to full
        saturation for every marker, even if every cluster's mean is close to zero, so it can't
        distinguish "genuinely enriched" from "merely the highest of a uniformly weak field" the
        way a z-score can -- on input that's already per-cell z-scored, that's a real cost, not
        just a cosmetic one, which is why `"zscore"` is the default here rather than the more
        common scanpy convention.
    obs_continuous_cmap : str, optional
        Colormap for the `obs_continuous` heatmap section. By default "YlOrRd" -- kept as its
        own argument (unlike `clustering_markers`/`extra_markers`, which now use the active
        style's `sequential_cmap`) since the right colormap here depends on what the feature
        means, not on house style.
    obs_flags_cmap : str, optional
        Colormap for the `obs_flags` heatmap section. By default "Reds", for the same reason.
    label_max_chars : int or None, optional
        A feature/marker display label (after `label_overrides`) longer than this is split onto
        two lines at the underscore/space nearest its middle (e.g. a QC flag name like
        "qc_flag_low_dapi" -> "qc_flag" / "low_dapi") -- under the 90-degree tick-label rotation
        this renders as two shorter lines side by side rather than one long vertical run, keeping
        the bottom margin from growing with every long name. Pass None to disable. By default 10.
    annotate_obs : bool, optional
        Annotate each obs heatmap cell with its numeric value. By default True.
    show_totals : bool, optional
        Add a horizontal bar chart of cluster cell counts on the right. By default True.
    flag_small_clusters : bool, optional
        In the `show_totals` bar, color a cluster below `min_cluster_frac` of all cells red
        (`style.accent_color_negative`) -- a quick "this cluster might be too small/noisy to
        annotate confidently" signal, matching `marker_dotplot`'s `show_cluster_size` flag. A
        cluster that's also in `highlight_cluster` shows its own color instead (see
        `highlight_cluster`) -- the two signals are independent and don't share a color. Has no
        effect if `show_totals=False`. By default False (opt-in, since it changes an existing
        call's bar colors).
    min_cluster_frac : float, optional
        Threshold for `flag_small_clusters`, as a fraction of all cells shown (0-1). By default
        0.01 (1%), matching `marker_dotplot`'s default.
    cell_w_inches : float, optional
        Width (inches) allocated per feature column. By default 0.52.
    row_h_inches : float, optional
        Height (inches) per cluster row. By default 0.40.
    margin_left_inches : float or None, optional
        Space (inches) left of the plot for the cluster labels (not transposed). None (default)
        keeps the fixed 2.0" (+ colour strip); set it smaller for print-sized figures.
    section_gap_inches : float, optional
        Gap (inches) after each section (not transposed). By default 0.35.
    totals_width_inches : float, optional
        Width (inches) of the "Cluster size" bar panel (not transposed). By default 1.5.
    label_block_min_inches : float, optional
        Minimum height (inches) reserved below the plot for the rotated feature labels
        (not transposed). By default 1.3.
    label_block_inches : float or None, optional
        Exact height (inches) reserved for the rotated feature labels (not transposed), instead of
        the estimate from the longest label. None (default) uses the estimate.
    palette : dict or None or "auto", optional
        Maps cluster label to a color, drawn as a strip to the left of the cluster labels.
        "auto" (default) reuses/creates a persistent mapping for every value of
        `adata.obs[groupby]` (cached in `adata.uns[groupby + "_colors"]`), so this always
        matches whatever another function (e.g. `catplot`) already established -- or will
        later reuse -- for the same `groupby` column; pass None to omit the strip entirely;
        pass a dict to use your own (any cluster it doesn't cover is colored grey, with a
        warning, rather than requiring an exhaustive mapping).
    cluster_order : Sequence[str], optional
        Clusters listed top to bottom; any present-but-unlisted clusters are appended at the
        bottom in natural sort order.
    label_overrides : dict, optional
        Maps a feature name to a display label, for prettier axis text.
    dendrogram : bool, optional
        Draw a hierarchical-clustering dendrogram alongside the plot, matching
        `scanpy.tl.dendrogram`'s own approach: correlation distance (1 - Pearson r) between
        per-cluster mean `clustering_markers` profiles, with complete linkage by default -- so this
        agrees with what `sc.pl.dotplot(..., dendrogram=True)` would draw for the same
        clusters/markers. When True, clusters are reordered to the dendrogram's own leaf order --
        `cluster_order` is ignored, since a fixed custom order would otherwise force crossing
        branches in the drawn tree. Skipped (with no error) if there are fewer than 2 clusters. By
        default False.
    dendrogram_position : {"after_markers", "end"}, optional
        Where to place the dendrogram panel: immediately right of the clustering-markers section,
        or as the very last panel (right of the cell-count bars). By default "after_markers".
        Under `transpose=True`, "right of"/"last panel" become "directly below"/"at the very
        bottom" -- the dendrogram is never rotated, just relocated to the axis that's now
        shared across columns instead of rows.
    dendrogram_linkage_method : str, optional
        Passed to `scipy.cluster.hierarchy.linkage`. By default "complete" (matching
        `scanpy.tl.dendrogram`'s default).
    dendrogram_width_inches : float, optional
        Width (inches) of the dendrogram panel. By default 0.9. Under `transpose=True` this is
        the panel's *height* instead (it now sits below the clustering-markers section, spanning
        the cluster columns), everything else about the argument unchanged.
    highlight_cluster : str, int, or list/tuple/set of str/int, optional
        One or more cluster labels (within `groupby`) to highlight throughout the figure: a blue
        (`_style.accent_color_positive`) band in every dot section (`clustering_markers`/
        `extra_markers`) -- a row band normally, a column band under `transpose=True` -- bold
        annotation text in every heatmap section (`obs_continuous`/`obs_flags`), and -- in the
        cluster-size panel -- that cluster's own established color (the same one its color-strip
        swatch uses, via `palette`; outlined in black if that color is too pale to read against
        white). A label not found in `adata.obs[groupby]` is silently skipped (no error) rather
        than failing the whole call. Independent of `spatial_region` -- set this alone to
        highlight the cluster(s) without adding the spatial panel below.
    spatial_region : str, optional
        Add a scatter panel at the very end of the figure (the bottom-most panel under
        `transpose=True`, the rightmost one otherwise -- the scatter itself is never rotated,
        only its position in the figure changes) showing this one `region_col` value's cells,
        colored by whether they belong to `highlight_cluster` (grey background for everything
        else) -- each highlighted cluster in its own established color (the same one used for
        its row's color strip, via `palette`), with its own count in the panel title. Requires
        `highlight_cluster` to also be set.
    region_col : str, optional
        obs column identifying each cell's region, used to select `spatial_region`. By default
        "unique_region".
    x_col, y_col : str, optional
        obs columns with each cell's spatial coordinates, for the `spatial_region` panel. By
        default "x", "y".
    spatial_point_size : float, optional
        Marker size in the spatial panel. By default 15.0.
    spatial_grey_alpha, spatial_color_alpha : float, optional
        Opacity of the background (non-highlighted) vs. highlighted-cluster points in the
        spatial panel. By default 0.2 and 0.8.
    spatial_width_inches : float, optional
        Width (inches) of the spatial panel. By default 10.0. Under `transpose=True` this is the
        panel's *height* instead (it now spans the full figure width at the bottom) -- the
        default of 10.0 is sized for a narrow side column, so a transposed call will usually want
        a smaller explicit value here.
    title : str, optional
        Overall figure title.
    style : SPACEcStyle or str or pathlib.Path, optional
        Style to use for this call. Drives: the "auto" cluster color strip and the
        `clustering_markers`/`extra_markers` dot cmap (both via the shared palette/
        `sequential_cmap` system), every font size (axis/cluster/section-title labels,
        heatmap annotations, colorbar/legend text), figure dpi, and the dot/legend edge color.
        `obs_continuous_cmap`/`obs_flags_cmap`, dot size, and all panel-geometry arguments
        (`cell_w_inches`, `row_h_inches`, `dendrogram_width_inches`, ...) stay this function's
        own explicit arguments -- they're specific to what's being shown or to this function's
        own layout math, not a house-style preference.
    transpose : bool, optional
        If True, swaps which axis clusters and features sit on: cell types become columns along
        the top (with the color strip directly below them), and markers/shape-features/QC-flags
        become rows down the left, with each section's own colorbar moved to a vertical strip in
        the left margin instead of a horizontal one underneath it. The cell-count bar chart
        becomes a vertical bar per cluster column instead of a horizontal one per cluster row.
        Reads better than the default layout once there are many cell types (the usual non-
        transposed layout keeps adding rows, which costs little width; transposing keeps adding
        columns instead, which is the direction print figures tend to have more room in). The
        "frac > thr" dot-size legend is placed inside whichever of `clustering_markers`/
        `extra_markers`'s own dot section has enough vertical room for it (preferring
        `clustering_markers`'s), falling back to the space below the cluster-size/spatial panel
        if neither does. Nothing about what's shown changes, only the layout -- see
        `dendrogram_position`, `dendrogram_width_inches`, `spatial_region`, and
        `spatial_width_inches` above for how dendrogram/spatial placement and sizing adapt. By
        default False.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _style = _apply_style(style)
    feature_label_fontsize = _style.font_size_large
    cluster_label_fontsize = _style.font_size_large
    section_title_fontsize = _style.font_size_large
    annotation_fontsize = _style.font_size_annotation
    legend_fontsize = (
        _style.legend_fontsize
        if _style.legend_fontsize is not None
        else _style.font_size_small
    )
    dpi = _style.figure_dpi

    if color_scale not in ("minmax", "zscore"):
        raise ValueError(
            f'color_scale must be "minmax" or "zscore", got {color_scale!r}.'
        )

    # Normalize highlight_cluster to a list, matching marker_dotplot's own lenient handling of
    # either a single label or a list/tuple/set/array of labels.
    if highlight_cluster is None:
        highlight_clusters = []
    elif isinstance(highlight_cluster, (list, tuple, set, np.ndarray, pd.Index)):
        highlight_clusters = list(highlight_cluster)
    else:
        highlight_clusters = [highlight_cluster]
    highlight_cluster_strs = [str(c) for c in highlight_clusters]

    if spatial_region is not None and not highlight_clusters:
        raise ValueError("spatial_region requires highlight_cluster to also be set.")
    spatial_active = bool(highlight_clusters) and spatial_region is not None

    _warn_on_clustering_marker_mismatch(adata, groupby, clustering_markers)

    # --- expression matrix --------------------------------------------------
    if layer is not None:
        expr = _to_dense_array(adata.layers[layer]).astype(np.float32)
    else:
        expr = _to_dense_array(adata.X).astype(np.float32)
    var_names = list(adata.var_names.astype(str))

    # --- cluster ordering ---------------------------------------------------
    groups = adata.obs[groupby].astype(str)

    def _cluster_sort_key(x):
        s = x.lstrip("-")
        return (not s.isdigit(), int(x) if s.isdigit() else x)

    if cluster_order is not None:
        # caller lists clusters top -> bottom; present-but-unlisted ones go to the bottom. The
        # dot/strip axes have their origin at the bottom (index 0 = bottom row), so reverse to
        # render top-down.
        present = set(groups.unique())
        ordered_top_to_bottom = [c for c in cluster_order if c in present]
        listed = set(cluster_order)
        ordered_top_to_bottom += sorted(
            (c for c in present if c not in listed), key=_cluster_sort_key
        )
        unique_clusters = list(reversed(ordered_top_to_bottom))
    else:
        unique_clusters = sorted(groups.unique(), key=_cluster_sort_key)
    n_clusters = len(unique_clusters)
    cl_idx = {cl: i for i, cl in enumerate(unique_clusters)}
    cl_masks = {cl: groups.to_numpy() == cl for cl in unique_clusters}
    cl_sizes = np.array([int(np.sum(cl_masks[cl])) for cl in unique_clusters])

    # --- optional dendrogram on clustering_markers --------------------------
    # Row order must be finalised before any per-section stats are computed below, since those
    # closures read `unique_clusters`/`cl_idx`/`cl_masks` from this enclosing scope.
    if dendrogram and dendrogram_position not in ("after_markers", "end"):
        raise ValueError('dendrogram_position must be "after_markers" or "end"')
    dendrogram_active = bool(dendrogram) and n_clusters >= 2
    dendrogram_Z = None
    if dendrogram_active:
        from scipy.cluster.hierarchy import dendrogram as _scipy_dendrogram
        from scipy.cluster.hierarchy import linkage as _linkage
        from scipy.spatial.distance import squareform

        fi = [var_names.index(f) for f in clustering_markers]
        cluster_means = np.zeros((n_clusters, len(fi)), dtype=np.float64)
        for cl in unique_clusters:
            vals = expr[cl_masks[cl]][:, fi]
            cluster_means[cl_idx[cl]] = np.nanmean(vals, axis=0) if vals.size else 0.0
        # Matches `scanpy.tl.dendrogram`'s own approach (correlation distance between per-cluster
        # mean profiles, not Euclidean distance on the raw means) so this dendrogram agrees with
        # what `sc.pl.dotplot(..., dendrogram=True)` would draw for the same clusters/markers.
        corr = np.clip(np.corrcoef(cluster_means), -1.0, 1.0)
        condensed = squareform(1.0 - corr, checks=False)
        dendrogram_Z = _linkage(condensed, method=dendrogram_linkage_method)
        leaf_order = _scipy_dendrogram(dendrogram_Z, no_plot=True)["leaves"]
        unique_clusters = [unique_clusters[i] for i in leaf_order]
        cl_idx = {cl: i for i, cl in enumerate(unique_clusters)}
        cl_masks = {cl: groups.to_numpy() == cl for cl in unique_clusters}
        cl_sizes = np.array([int(np.sum(cl_masks[cl])) for cl in unique_clusters])

    # By name, not position: unique_clusters may already be in dendrogram-leaf order at this
    # point, not the sorted order adata.uns[groupby + "_colors"] is cached in (the same
    # convention scanpy itself uses) -- zipping a palette list against it positionally, as
    # this used to do, silently misaligned colors vs. any other function's plot of the same
    # `groupby` (e.g. catplot) the moment dendrogram=True reordered anything.
    if palette == "auto":
        palette = _resolve_categorical_palette(_style, adata, groupby, unique_clusters)
    elif isinstance(palette, dict):
        palette = _resolve_categorical_palette(
            _style, adata, groupby, unique_clusters, palette=palette
        )

    # row indices of highlight_cluster(s) in the final (possibly dendrogram-reordered) cluster
    # order -- a label not present is silently skipped (no highlight drawn for it), matching
    # marker_dotplot's own lenient behavior.
    highlight_y_indices = {
        unique_clusters.index(hc)
        for hc in highlight_cluster_strs
        if hc in unique_clusters
    }

    # --- per-cluster statistics helpers ------------------------------------
    def _dot_stats(feature_list):
        """Returns (norm_means, fracs, raw_means) each (n_clusters, n_feat).

        `norm_means` (the dot color value) is computed per `color_scale`: "minmax" rescales
        each marker's cluster means to [0, 1]; "zscore" standardizes them (subtract mean,
        divide by std, across clusters), clipped to style.heatmap_vmin/vmax like
        `marker_dotplot`'s own z-score coloring. See `color_scale`'s docstring entry.
        """
        fi = [var_names.index(f) for f in feature_list]
        mat = expr[:, fi]
        means = np.zeros((n_clusters, len(feature_list)), dtype=np.float32)
        fracs = np.zeros((n_clusters, len(feature_list)), dtype=np.float32)
        for cl in unique_clusters:
            ci = cl_idx[cl]
            vals = mat[cl_masks[cl]]
            if vals.size:
                means[ci] = np.nanmean(vals, axis=0)
                fracs[ci] = np.mean(vals > expression_threshold, axis=0)
        if color_scale == "zscore":
            mu = np.nanmean(means, axis=0, keepdims=True)
            sd = np.nanstd(means, axis=0, keepdims=True)
            sd = np.where(sd > 1e-9, sd, 1.0)
            norm_means = np.clip(
                (means - mu) / sd, _style.heatmap_vmin, _style.heatmap_vmax
            )
        else:
            fmin = np.nanmin(means, axis=0, keepdims=True)
            fmax = np.nanmax(means, axis=0, keepdims=True)
            rng = np.where(fmax - fmin > 1e-9, fmax - fmin, 1.0)
            norm_means = (means - fmin) / rng
        return norm_means, fracs, means

    def _obs_cont_stats(col_list):
        out = np.full((n_clusters, len(col_list)), np.nan)
        for ci_col, col in enumerate(col_list):
            vals = adata.obs[col].to_numpy(dtype=float)
            for cl in unique_clusters:
                v = vals[cl_masks[cl]]
                if v.size:
                    out[cl_idx[cl], ci_col] = float(np.nanmean(v))
        return out

    def _obs_flag_stats(col_list):
        out = np.zeros((n_clusters, len(col_list)))
        for ci_col, col in enumerate(col_list):
            vals = adata.obs[col].to_numpy(dtype=bool)
            for cl in unique_clusters:
                v = vals[cl_masks[cl]]
                if v.size:
                    out[cl_idx[cl], ci_col] = float(np.mean(v))
        return out

    # --- build section definitions -----------------------------------------
    # "zscore" is a signed, centered value -- a diverging cmap and a symmetric vmin/vmax read it
    # correctly, unlike "minmax"'s bounded-[0, 1] sequential cmap.
    dot_cmap = (
        _style.diverging_cmap if color_scale == "zscore" else _style.sequential_cmap
    )
    dot_vmin, dot_vmax = (
        (_style.heatmap_vmin, _style.heatmap_vmax)
        if color_scale == "zscore"
        else (0.0, 1.0)
    )

    sections = []
    sections.append(
        {
            "type": "dots",
            "features": list(clustering_markers),
            "cmap": dot_cmap,
            "title": "Clustering markers",
        }
    )
    if extra_markers:
        sections.append(
            {
                "type": "dots",
                "features": list(extra_markers),
                "cmap": dot_cmap,
                "title": "Other markers",
            }
        )
    if obs_continuous:
        sections.append(
            {
                "type": "cont",
                "features": list(obs_continuous),
                "cmap": obs_continuous_cmap,
                "title": "Shape features",
            }
        )
    if obs_flags:
        sections.append(
            {
                "type": "flags",
                "features": list(obs_flags),
                "cmap": obs_flags_cmap,
                "title": "% flagged in QC",
            }
        )

    # --- compute stats per section -----------------------------------------
    for sec in sections:
        if sec["type"] == "dots":
            sec["norm_means"], sec["fracs"], sec["raw_means"] = _dot_stats(
                sec["features"]
            )
        elif sec["type"] == "cont":
            raw = _obs_cont_stats(sec["features"])
            sec["raw"] = raw
            col_min = np.nanmin(raw, axis=0, keepdims=True)
            col_max = np.nanmax(raw, axis=0, keepdims=True)
            col_rng = np.where(col_max - col_min > 1e-9, col_max - col_min, 1.0)
            sec["norm_raw"] = (raw - col_min) / col_rng
            sec["col_max_raw"] = col_max.ravel()
            sec["vmin"] = 0.0
            sec["vmax"] = 1.0
        elif sec["type"] == "flags":
            raw = _obs_flag_stats(sec["features"])
            sec["raw"] = raw
            sec["vmax"] = max(float(np.nanmax(raw)), 0.02)

    if not transpose:
        # --- figure sizing -----------------------------------------------------
        n_feat_total = sum(len(s["features"]) for s in sections)
        totals_w_inches = totals_width_inches if show_totals else 0.0
        section_gap_inches = section_gap_inches * len(sections)
        strip_w_inches = 0.22 if palette is not None else 0.0
        strip_gap_inches = 0.12 if palette is not None else 0.0
        # y-tick labels are anchored at the plot's left spine and would otherwise sit on top of the
        # colour strip; we pad them left past the strip (see tick_params below) and reserve matching
        # space here so the text + axis label still get a full ~2.0" and don't clip the figure edge.
        ylabel_pad_inches = strip_w_inches + strip_gap_inches
        margin_l = (
            (2.0 if margin_left_inches is None else margin_left_inches)
            + strip_w_inches
            + strip_gap_inches
            + ylabel_pad_inches
        )
        margin_r = 0.3
        # scales with section_title_fontsize (baseline 11) so a larger section-title font still has
        # room to breathe at the figure's top edge, matching the larger pad given to it just above.
        margin_t = 0.55 * (section_title_fontsize / 11.0) + (0.45 if title else 0.0)
        # shared by every set_title(...) call below (section titles, "Cluster size") so they all sit
        # at the same height -- a fixed 4pt pad leaves a descender (e.g. the "gg" in
        # "flagged") touching the panel below it once the title font gets large.
        section_title_pad = max(4, section_title_fontsize * 0.65)

        # x-tick labels are rotated 90 degrees, so a long feature/marker name (e.g. from
        # `label_overrides`) needs a taller bottom margin or it clips/overlaps the colorbar row below
        # it. ~0.6*fontsize per character is a rough estimate for a proportional font. A label split
        # by `_wrap_label` renders as two lines side by side under the rotation rather than one long
        # line, so the margin only needs to fit its longest single line, not the full original length.
        all_display_labels = [
            _wrap_label(
                label_overrides.get(f, f) if label_overrides else f, label_max_chars
            )
            for sec in sections
            for f in sec["features"]
        ]
        max_label_len = max(
            (len(line) for label in all_display_labels for line in label.split("\n")),
            default=0,
        )
        if label_block_inches is None:
            label_block_inches = max(
                label_block_min_inches,
                max_label_len * feature_label_fontsize * 0.6 / 72 + 0.3,
            )

        # Bottom margin sub-layout, bottom to top, all FIXED inches so it stays consistent regardless
        # of n_clusters (unlike figure-fraction placement, which would drift since fig_h scales with
        # n_clusters -- this was the root cause of the dot-size legend/colorbar overlapping or
        # disappearing off-canvas entirely at cluster-count extremes).
        bottom_pad_inches = 0.05
        cb_h_inches = 0.12
        cb_gap_inches = 0.08
        legend_h_inches = 0.35
        legend_gap_inches = 0.05
        cb_y_inches = bottom_pad_inches
        legend_y_inches = cb_y_inches + cb_h_inches + cb_gap_inches
        margin_b = (
            legend_y_inches + legend_h_inches + legend_gap_inches + label_block_inches
        )

        dendrogram_w_inches = dendrogram_width_inches if dendrogram_active else 0.0
        dendrogram_gap_inches = 0.3 if dendrogram_active else 0.0
        spatial_w_inches = spatial_width_inches if spatial_active else 0.0
        spatial_gap_inches = 0.4 if spatial_active else 0.0

        # --- stagger overflowing panel titles -----------------------------------
        # a panel narrower than its own title (most likely a single-feature section, at a large
        # section_title_fontsize) would otherwise have its title visually collide with its
        # neighbor's. Only a title that overflows its *own* panel is ever raised -- a panel whose
        # title already fits stays put even if a wide neighbor's title spills toward it. Whether a
        # spilling title actually needs raising is decided from its real rendered extent (centred on
        # its own panel) against its immediate left/right neighbour, not just "does it overflow" --
        # e.g. an overflowing title next to a very wide, mostly-empty panel doesn't need to move.
        # "Dendrogram" has no title at all (see ax_dendrogram.set_title below) so it never enters
        # this, and doesn't need to: its panel is narrow by design, not a layout problem.
        # Uses *measured* title widths (not a per-character estimate) since that estimate proved
        # unreliable enough to trigger on panels the title actually fits in just fine.
        _title_panels = (
            []
        )  # (title, x0_inches, panel_width_inches), in actual left-to-right render order
        _x = margin_l
        _gap_each = section_gap_inches / max(len(sections), 1)
        for _i, _sec in enumerate(sections):
            _w = len(_sec["features"]) * cell_w_inches
            _title_panels.append((_sec["title"], _x, _w))
            _x += _w + _gap_each
            if dendrogram_active and dendrogram_position == "after_markers" and _i == 0:
                _x += dendrogram_w_inches + dendrogram_gap_inches
        if show_totals:
            _title_panels.append(("Cluster size", _x, totals_w_inches))

        _measured_w = {
            t: _measure_text_width_inches(t, section_title_fontsize, dpi)
            for t, _, _ in _title_panels
        }
        _overflows = {t: _measured_w[t] > w for t, _, w in _title_panels}
        _level = {t: 0 for t, _, _ in _title_panels}

        def _extent(title, x0, w, raised):
            center = x0 + w / 2
            half = _measured_w[title] / 2
            return center - half, center + half

        for (t0, x0, w0), (t1, x1, w1) in zip(_title_panels, _title_panels[1:]):
            if _level[t0] != _level[t1]:
                continue  # already separated by a previous raise -- can't collide
            _, right0 = _extent(t0, x0, w0, _level[t0])
            left1, _ = _extent(t1, x1, w1, _level[t1])
            if right0 <= left1:
                continue  # no actual overlap between the two rendered titles
            if _overflows[t0] and not _overflows[t1]:
                _level[t0] = 1
            else:
                # right one overflows, or both do -- raise the right one, since raising the left one
                # here would just push its own overlap problem onto whatever sits to ITS left instead
                _level[t1] = 1

        raised_titles = {t for t, lvl in _level.items() if lvl == 1}
        # one line height of extra clearance, only reserved if anything actually needs raising
        raised_pad_extra_pt = section_title_fontsize * 1.3
        if raised_titles:
            margin_t += raised_pad_extra_pt / 72

        fig_w = (
            margin_l
            + n_feat_total * cell_w_inches
            + section_gap_inches
            + totals_w_inches
            + dendrogram_w_inches
            + dendrogram_gap_inches
            + spatial_w_inches
            + spatial_gap_inches
            + margin_r
        )
        fig_h = margin_t + n_clusters * row_h_inches + margin_b

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        # Convert margins to figure fractions
        left0 = margin_l / fig_w
        right1 = 1.0 - margin_r / fig_w
        top1 = 1.0 - margin_t / fig_h
        bottom0 = margin_b / fig_h

        avail_w = right1 - left0
        totals_frac = totals_w_inches / fig_w
        gap_frac = section_gap_inches / fig_w / max(len(sections), 1)
        dendrogram_w_frac = dendrogram_w_inches / fig_w
        dendrogram_gap_frac = dendrogram_gap_inches / fig_w
        spatial_frac = spatial_w_inches / fig_w
        spatial_gap_frac = spatial_gap_inches / fig_w
        feat_frac_per_col = (
            avail_w
            - totals_frac
            - gap_frac * len(sections)
            - dendrogram_w_frac
            - dendrogram_gap_frac
            - spatial_frac
            - spatial_gap_frac
        ) / max(n_feat_total, 1)

        # --- colour strip (must be added BEFORE section axes so it renders beneath tick labels) ---
        if palette is not None and strip_w_inches > 0:
            strip_w_frac = strip_w_inches / fig_w
            # normalise via to_rgba so hex strings / named colours / RGBA tuples all work
            color_arr = np.array(
                [
                    mcolors.to_rgba(palette.get(cl, (0.8, 0.8, 0.8, 1.0)))
                    for cl in unique_clusters
                ],
                dtype=float,
            ).reshape(n_clusters, 1, 4)
            ax_strip = fig.add_axes(
                [left0 - strip_w_frac, bottom0, strip_w_frac, top1 - bottom0]
            )
            ax_strip.imshow(
                color_arr, aspect="auto", origin="lower", interpolation="nearest"
            )
            ax_strip.set_ylim(-0.5, n_clusters - 0.5)
            ax_strip.set_axis_off()

        # --- create axes (dot/heatmap sections) --------------------------------
        x_cursor = left0
        axes_list = []
        first_ax = None
        ax_dendrogram = None

        for sec_pos, sec in enumerate(sections):
            n_f = len(sec["features"])
            w_frac = n_f * feat_frac_per_col
            rect = [x_cursor, bottom0, w_frac, top1 - bottom0]
            if first_ax is None:
                ax = fig.add_axes(rect)
                first_ax = ax
            else:
                ax = fig.add_axes(rect, sharey=first_ax)
            axes_list.append(ax)
            x_cursor += w_frac + gap_frac

            if (
                dendrogram_active
                and dendrogram_position == "after_markers"
                and sec_pos == 0
            ):
                rect_dendro = [x_cursor, bottom0, dendrogram_w_frac, top1 - bottom0]
                ax_dendrogram = fig.add_axes(rect_dendro, sharey=first_ax)
                x_cursor += dendrogram_w_frac + dendrogram_gap_frac

        ax_totals = None
        if show_totals and first_ax is not None:
            rect_tot = [x_cursor, bottom0, totals_frac, top1 - bottom0]
            ax_totals = fig.add_axes(rect_tot, sharey=first_ax)
            x_cursor += totals_frac

        if dendrogram_active and dendrogram_position == "end" and first_ax is not None:
            x_cursor += dendrogram_gap_frac
            rect_dendro = [x_cursor, bottom0, dendrogram_w_frac, top1 - bottom0]
            ax_dendrogram = fig.add_axes(rect_dendro, sharey=first_ax)
            x_cursor += dendrogram_w_frac

        ax_spatial = None
        if spatial_active and first_ax is not None:
            x_cursor += spatial_gap_frac
            rect_spatial = [x_cursor, bottom0, spatial_frac, top1 - bottom0]
            ax_spatial = fig.add_axes(rect_spatial)

        # --- render each section -----------------------------------------------
        for sec_i, (sec, ax) in enumerate(zip(sections, axes_list)):
            n_f = len(sec["features"])
            ax.set_xlim(-0.5, n_f - 0.5)
            ax.set_ylim(-0.5, n_clusters - 0.5)
            ax.set_xticks(range(n_f))
            display_labels = [
                _wrap_label(
                    label_overrides.get(f, f) if label_overrides else f, label_max_chars
                )
                for f in sec["features"]
            ]
            # ha="right" (the reference's own choice) places the label's bounding-box *right edge*
            # at the tick -- correct by definition, but that box has real width (about one line
            # height), so the label's visual *center* sits that much left of the tick. Measured
            # directly (get_window_extent vs. the tick's transformed position): at the reference's
            # typical smaller fonts this is a few px and unnoticeable; at the larger fonts here it's
            # ~18px, visibly off. ha="center" with the default (non-anchor) rotation_mode measures at
            # exactly 0.0px offset regardless of font size -- verified numerically, not by eye.
            # `ha` also drives multi-line internal alignment by default, which would recenter each
            # wrapped line relative to the other -- `multialignment` is the separate knob for that,
            # so a 2-line wrapped label can be right-aligned internally while the whole block still
            # centers on its tick (verified: multialignment doesn't change the 0.0px offset above).
            ax.set_xticklabels(
                display_labels,
                rotation=90,
                fontsize=feature_label_fontsize,
                ha="center",
                multialignment="right",
            )
            ax.tick_params(left=False, right=False, length=0)
            for spine in ("top", "right", "bottom", "left"):
                ax.spines[spine].set_visible(False)
            _pad = section_title_pad + (
                raised_pad_extra_pt if sec["title"] in raised_titles else 0
            )
            ax.set_title(sec["title"], fontsize=section_title_fontsize, pad=_pad)
            # importing spacec flips the global axes.grid rcParam on (both axes) as a side effect of
            # one of its dependencies, so explicitly disable it here rather than assume a clean
            # default. The horizontal row-guides are only kept for dot sections -- they help read
            # dots across columns, but sit messily on top of the heatmap sections' colored cells.
            ax.grid(axis="x", visible=False)
            if sec["type"] == "dots":
                ax.grid(axis="y", color="#e8e8e8", linewidth=0.5, zorder=0)
            else:
                ax.grid(axis="y", visible=False)
            ax.set_axisbelow(True)

            if sec_i == 0:
                ax.set_yticks(range(n_clusters))
                ax.set_yticklabels(unique_clusters, fontsize=cluster_label_fontsize)
                ax.set_ylabel(groupby, fontsize=cluster_label_fontsize)
                if palette is not None and strip_w_inches > 0:
                    # shift labels left past the colour strip so the text clears the swatches
                    ax.tick_params(axis="y", pad=ylabel_pad_inches * 72.0 + 2.0)
            else:
                ax.tick_params(labelleft=False)

            if sec["type"] == "dots":
                nm = sec["norm_means"]
                fr = sec["fracs"]
                xs = np.tile(np.arange(n_f), n_clusters)
                ys = np.repeat(np.arange(n_clusters), n_f)
                ax.scatter(
                    xs,
                    ys,
                    c=nm.ravel(),
                    s=fr.ravel() * max_dot_size,
                    cmap=sec["cmap"],
                    vmin=dot_vmin,
                    vmax=dot_vmax,
                    linewidths=0.3,
                    edgecolors=_style.edge_color_default,
                    zorder=3,
                    clip_on=False,
                )
                for hy in highlight_y_indices:
                    ax.axhspan(
                        hy - 0.5,
                        hy + 0.5,
                        color=_style.accent_color_positive,
                        alpha=0.15,
                        zorder=0,
                    )
            else:
                if sec["type"] == "cont":
                    data_img = sec["norm_raw"]
                    vmin_im, vmax_im = 0.0, 1.0
                else:
                    data_img = sec["raw"]
                    vmin_im, vmax_im = 0.0, sec["vmax"]
                ax.imshow(
                    data_img,
                    aspect="auto",
                    cmap=sec["cmap"],
                    vmin=vmin_im,
                    vmax=vmax_im,
                    origin="lower",
                    interpolation="nearest",
                )
                if annotate_obs:
                    for ci in range(n_clusters):
                        for fi in range(n_f):
                            val = sec["raw"][ci, fi]
                            if np.isnan(val):
                                continue
                            if sec["type"] == "flags":
                                # ~2 significant figures (e.g. "13%", "2.5%") rather than a fixed one
                                # decimal place for every value -- keeps this section from getting
                                # visually crowded once cell widths shrink at a larger annotation font.
                                pct = val * 100
                                txt = (
                                    f"{pct:.0f}%"
                                    if pct >= 10 or pct == 0
                                    else f"{pct:.1f}%"
                                )
                            elif sec["type"] == "cont":
                                cm = sec["col_max_raw"][fi]
                                if cm > 10:
                                    txt = f"{val:.0f}"
                                elif cm > 1:
                                    txt = f"{val:.1f}"
                                else:
                                    txt = f"{val:.2f}"
                            else:
                                txt = f"{val:.1f}"
                            ax.text(
                                fi,
                                ci,
                                txt,
                                ha="center",
                                va="center",
                                fontsize=annotation_fontsize,
                                color="black",
                                fontweight=(
                                    "bold" if ci in highlight_y_indices else "normal"
                                ),
                            )

        # --- totals bar chart --------------------------------------------------
        if ax_totals is not None:
            frac_of_total = cl_sizes / max(cl_sizes.sum(), 1)
            for ci in range(n_clusters):
                edge_color, edge_width = "none", 0.0
                if ci in highlight_y_indices:
                    # A highlighted cluster's own row color (same as its color-strip swatch), not
                    # the small-cluster red -- distinct signals, shouldn't share a color. Palette
                    # colors can be pale, so add a black outline when needed to stay visible.
                    bar_color = (palette or {}).get(
                        unique_clusters[ci], _style.accent_color_positive
                    )
                    if _is_light_color(bar_color):
                        edge_color, edge_width = "black", 0.8
                elif flag_small_clusters and frac_of_total[ci] < min_cluster_frac:
                    bar_color = _style.accent_color_negative
                else:
                    bar_color = "#7b8da8"
                ax_totals.barh(
                    ci,
                    cl_sizes[ci],
                    color=bar_color,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    height=0.65,
                )
            ax_totals.set_xlim(0, max(cl_sizes) * 1.12)
            ax_totals.set_ylim(-0.5, n_clusters - 0.5)
            _pad = section_title_pad + (
                raised_pad_extra_pt if "Cluster size" in raised_titles else 0
            )
            ax_totals.set_title(
                "Cluster size", fontsize=section_title_fontsize, pad=_pad
            )
            ax_totals.set_xlabel("n cells", fontsize=feature_label_fontsize)
            ax_totals.tick_params(labelleft=False, left=False, length=0)
            ax_totals.grid(False)  # see the axes.grid rcParam note above
            for spine in ("top", "right", "left"):
                ax_totals.spines[spine].set_visible(False)
            ax_totals.xaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else str(int(x))
                )
            )
            ax_totals.tick_params(
                axis="x", labelsize=feature_label_fontsize, rotation=45
            )

        # --- dendrogram (optional) ----------------------------------------------
        if dendrogram_active and ax_dendrogram is not None:
            dd = _scipy_dendrogram(dendrogram_Z, no_plot=True)
            icoord = np.asarray(dd["icoord"], dtype=float)
            dcoord = np.asarray(dd["dcoord"], dtype=float)
            # scipy's default leaf spacing (5, 15, 25, ...) maps 1:1 to row index (0, 1, 2, ...)
            # because `unique_clusters` was reordered to this exact leaf order above.
            y = (icoord - 5.0) / 10.0
            for xi, yi in zip(dcoord, y):
                ax_dendrogram.plot(
                    xi,
                    yi,
                    color="#555555",
                    linewidth=0.9,
                    solid_capstyle="round",
                    clip_on=False,
                )
            ax_dendrogram.set_ylim(-0.5, n_clusters - 0.5)
            max_h = float(dcoord.max()) if dcoord.size else 1.0
            ax_dendrogram.set_xlim(0, max_h * 1.05 if max_h > 0 else 1.0)
            # no title -- the panel's usually too narrow for "Dendrogram" to fit without triggering
            # the stagger logic for no real reason (a dendrogram is visually self-explanatory anyway)
            ax_dendrogram.set_axis_off()

        # --- spatial panel (optional) --------------------------------------------
        if ax_spatial is not None:
            region_mask = adata.obs[region_col].astype(str) == str(spatial_region)
            region_df = adata.obs.loc[region_mask]
            region_group_str = region_df[groupby].astype(str)
            highlighted_mask = region_group_str.isin(highlight_cluster_strs)

            # Grey background: every cell in the region except the highlighted cluster(s).
            ax_spatial.scatter(
                region_df.loc[~highlighted_mask, x_col],
                region_df.loc[~highlighted_mask, y_col],
                c="#CCCCCC",
                s=spatial_point_size,
                alpha=spatial_grey_alpha,
                linewidths=0,
                rasterized=True,
            )

            # Each highlighted cluster drawn separately, in the same color as its row's color strip
            # (via `palette`) -- falling back to accent_color_positive if the strip is disabled
            # (palette=None) or doesn't cover this label.
            spatial_counts = {}
            for hc in highlight_cluster_strs:
                cluster_mask = region_group_str == hc
                spatial_counts[hc] = int(cluster_mask.sum())
                if cluster_mask.any():
                    color = (palette or {}).get(hc, _style.accent_color_positive)
                    ax_spatial.scatter(
                        region_df.loc[cluster_mask, x_col],
                        region_df.loc[cluster_mask, y_col],
                        color=[color],
                        s=spatial_point_size,
                        alpha=spatial_color_alpha,
                        linewidths=0,
                        rasterized=True,
                        label=hc,
                    )

            if len(highlight_cluster_strs) == 1:
                hc = highlight_cluster_strs[0]
                spatial_title = (
                    f"{hc} -- {spatial_region} (n={spatial_counts.get(hc, 0)})"
                )
            else:
                count_text = ", ".join(
                    f"{hc}: n={spatial_counts[hc]}" for hc in highlight_cluster_strs
                )
                spatial_title = f"{', '.join(highlight_cluster_strs)} -- {spatial_region} ({count_text})"
            ax_spatial.set_title(spatial_title, fontsize=section_title_fontsize)
            ax_spatial.set_xlabel(x_col, fontsize=feature_label_fontsize)
            ax_spatial.set_ylabel(y_col, fontsize=feature_label_fontsize)
            ax_spatial.invert_yaxis()
            ax_spatial.set_aspect("equal", adjustable="datalim")
            ax_spatial.tick_params(axis="both", labelsize=legend_fontsize)
            _apply_spines(ax_spatial, _style)

        # --- colorbars (inset below each section) ------------------------------
        cb_h_rel = cb_h_inches / fig_h  # fixed inches, converted to figure-fraction
        cb_y_fig = cb_y_inches / fig_h
        for sec, ax in zip(sections, axes_list):
            ax_pos = ax.get_position()
            cb_ax = fig.add_axes([ax_pos.x0, cb_y_fig, ax_pos.width * 0.88, cb_h_rel])
            if sec["type"] == "dots":
                smap = plt.cm.ScalarMappable(
                    cmap=sec["cmap"], norm=plt.Normalize(dot_vmin, dot_vmax)
                )
                smap.set_array([])
                cb = fig.colorbar(smap, cax=cb_ax, orientation="horizontal")
                cb.set_label(
                    (
                        "mean expr (z-score across clusters)"
                        if color_scale == "zscore"
                        else "mean expr (per-marker scaled)"
                    ),
                    fontsize=legend_fontsize,
                )
            elif sec["type"] == "cont":
                smap = plt.cm.ScalarMappable(cmap=sec["cmap"], norm=plt.Normalize(0, 1))
                smap.set_array([])
                cb = fig.colorbar(
                    smap, cax=cb_ax, orientation="horizontal", ticks=[0, 1]
                )
                cb.ax.set_xticklabels(["low", "high"])
                cb.set_label("mean (per-feature scaled)", fontsize=legend_fontsize)
            elif sec["type"] == "flags":
                smap = plt.cm.ScalarMappable(
                    cmap=sec["cmap"], norm=plt.Normalize(0, sec["vmax"])
                )
                smap.set_array([])
                cb = fig.colorbar(
                    smap,
                    cax=cb_ax,
                    orientation="horizontal",
                    format=plt.FuncFormatter(lambda x, _: f"{x:.0%}"),
                )
                cb.set_label("fraction flagged", fontsize=legend_fontsize)
            cb.ax.tick_params(labelsize=legend_fontsize)

        # --- dot size legend -----------------------------------------------------
        dot_sections = [s for s in sections if s["type"] == "dots"]
        if dot_sections and axes_list:
            sizes_legend = [0.25, 0.5, 1.0]
            labels_legend = ["25%", "50%", "100%"]
            handles = [
                plt.scatter(
                    [],
                    [],
                    s=f * max_dot_size,
                    c="grey",
                    linewidths=0.3,
                    edgecolors=_style.edge_color_default,
                    label=lbl,
                )
                for f, lbl in zip(sizes_legend, labels_legend)
            ]
            # Anchored by its bottom-left corner at a fixed inch offset from the figure's bottom edge
            # (via the figure's own transform), NOT axes-fraction off axes_list[0], whose height
            # scales with n_clusters and would otherwise push this off the rendered canvas (many
            # clusters) or into the x-tick labels (few clusters).
            legend_y_fig = legend_y_inches / fig_h
            fig.legend(
                handles=handles,
                title="frac > thr",
                loc="lower left",
                bbox_to_anchor=(left0, legend_y_fig),
                bbox_transform=fig.transFigure,
                frameon=False,
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
                ncol=len(handles),
                handletextpad=0.4,
                columnspacing=0.8,
            )

        if title:
            fig.text(
                0.5,
                1.0 - (0.12 / fig_h),
                title,
                ha="center",
                va="top",
                fontsize=section_title_fontsize + 2,
            )

        return fig

    else:
        # ======================== transpose=True ============================
        # Cell types become columns (top), markers/features become rows (left) -- see
        # `transpose`'s docstring entry for why (readability at a large cell-type count, matching
        # a figure layout convention common in papers). The per-cluster statistics computed above
        # are unchanged; only how they're laid out and drawn differs below.
        n_feat_total = sum(len(s["features"]) for s in sections)
        cb_w_in = 0.16
        cb_gap_in = 0.12
        cb_block_in = (
            0.55  # colorbar body + its tick-number labels + rotated axis label
        )
        left_of_labels_gap_in = 0.12
        strip_h_in = 0.2 if palette is not None else 0.0
        strip_gap_in = 0.12 if palette is not None else 0.0
        section_title_pad = max(4, section_title_fontsize * 0.65)
        section_title_block_in = (section_title_fontsize * 1.6) / 72 + 0.08

        # Row-label margin (left): these labels are horizontal (not rotated), so their required
        # width is exactly their own rendered width -- measured directly, not estimated, the same
        # way `_measure_text_width_inches` already measures section-title overflow above.
        all_feature_display_labels = [
            (label_overrides.get(f, f) if label_overrides else f)
            for sec in sections
            for f in sec["features"]
        ]
        feat_label_block_in = (
            max(
                (
                    _measure_text_width_inches(lbl, feature_label_fontsize, dpi)
                    for lbl in all_feature_display_labels
                ),
                default=0.0,
            )
            + 0.3
        )

        # Column-label margin (top): these labels are now the ROTATED ones (cluster names), so --
        # unlike the row labels above -- they get the same `_wrap_label` treatment `extended_dotplot`
        # already applies to whichever axis is rotated, and the same character-count-based margin
        # estimate used for that axis in the non-transpose layout (rotated text's required margin
        # scales with its longest wrapped line, not the measured width of the whole string).
        cluster_display_labels = [
            _wrap_label(str(c), label_max_chars) for c in unique_clusters
        ]
        max_cluster_label_len = max(
            (len(line) for lbl in cluster_display_labels for line in lbl.split("\n")),
            default=0,
        )
        cluster_label_block_in = max(
            1.3, max_cluster_label_len * cluster_label_fontsize * 0.6 / 72 + 0.3
        )

        # Each section's colorbar lives in the left margin with a rotated label that matplotlib
        # centers on the colorbar's own axes regardless of how short that axes is -- a label
        # longer than its section's band (band height = len(features) * row_h_inches, fixed
        # regardless of anything computed below) overflows into the gap on either side no matter
        # how the colorbar itself is sized within that band. A handful of very short sections
        # (e.g. 2-3 `obs_continuous` features) can hit this even though the dot-size-legend host
        # section above is comfortably sized -- so this is checked for every section's own label,
        # not just the one hosting that legend. The fix is to simply widen every inter-section gap
        # enough to swallow the worst single offender, rather than track overflow per section pair.
        # Split before the opening bracket rather than relying on gap space to absorb the whole
        # string's length: under this rotation a line break renders side by side, not stacked (the
        # same reasoning `_wrap_label` already uses for rotated tick labels), so the *rotated*
        # length that matters for sizing is the longest surviving single line, not the full string.
        dot_label_text = (
            "mean expr\n(z-score across clusters)"
            if color_scale == "zscore"
            else "mean expr\n(per-marker scaled)"
        )
        _label_text_by_type = {
            "dots": dot_label_text,
            "cont": "mean\n(per-feature scaled)",
            "flags": "fraction flagged",
        }

        def _rotated_label_height_in(text):
            return (
                max(
                    _measure_text_width_inches(line, legend_fontsize, dpi)
                    for line in text.split("\n")
                )
                + 0.30
            )

        _label_h_in_by_type = {
            t: _rotated_label_height_in(txt) for t, txt in _label_text_by_type.items()
        }
        max_label_shortfall_in = max(
            (
                _label_h_in_by_type[sec["type"]] - len(sec["features"]) * row_h_inches
                for sec in sections
            ),
            default=0.0,
        )

        dendrogram_h_in = dendrogram_width_inches if dendrogram_active else 0.0
        dendrogram_gap_in = 0.3 if dendrogram_active else 0.0
        totals_h_in = 1.0 if show_totals else 0.0
        totals_gap_in = section_title_block_in if show_totals else 0.0
        spatial_h_in = spatial_width_inches if spatial_active else 0.0
        spatial_gap_in = 0.4 if spatial_active else 0.0

        margin_l = feat_label_block_in + left_of_labels_gap_in + cb_block_in + cb_gap_in
        margin_r = 0.3
        margin_t = (
            0.15
            + (0.45 if title else 0.0)
            + cluster_label_block_in
            + 0.15
            + strip_h_in
            + strip_gap_in
            + section_title_block_in
        )
        margin_b = 0.15

        fig_w = margin_l + n_clusters * cell_w_inches + margin_r
        non_grid_h_in = (
            dendrogram_gap_in
            + dendrogram_h_in
            + totals_gap_in
            + totals_h_in
            + spatial_gap_in
            + spatial_h_in
        )
        section_gap_in = section_title_block_in + max(0.0, max_label_shortfall_in)
        fig_h = (
            margin_t
            + n_feat_total * row_h_inches
            + section_gap_in * max(len(sections) - 1, 0)
            + non_grid_h_in
            + margin_b
        )

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        left0 = margin_l / fig_w
        right1 = 1.0 - margin_r / fig_w
        top1 = 1.0 - margin_t / fig_h
        bottom0 = margin_b / fig_h

        avail_h = top1 - bottom0 - non_grid_h_in / fig_h
        gap_frac = section_gap_in / fig_h
        feat_frac_per_row = (avail_h - gap_frac * max(len(sections) - 1, 0)) / max(
            n_feat_total, 1
        )

        # --- colour strip (top, aligned with cluster columns) -------------------
        if palette is not None and strip_h_in > 0:
            strip_h_frac = strip_h_in / fig_h
            color_arr = np.array(
                [
                    mcolors.to_rgba(palette.get(cl, (0.8, 0.8, 0.8, 1.0)))
                    for cl in unique_clusters
                ],
                dtype=float,
            ).reshape(1, n_clusters, 4)
            ax_strip = fig.add_axes([left0, top1, right1 - left0, strip_h_frac])
            ax_strip.imshow(
                color_arr, aspect="auto", origin="lower", interpolation="nearest"
            )
            ax_strip.set_xlim(-0.5, n_clusters - 0.5)
            ax_strip.set_axis_off()

        # --- create axes (dot/heatmap sections, stacked top to bottom) ----------
        y_cursor = top1
        axes_list = []
        first_ax = None
        ax_dendrogram = None
        for sec_pos, sec in enumerate(sections):
            n_f = len(sec["features"])
            h_frac = n_f * feat_frac_per_row
            y_cursor -= h_frac
            rect = [left0, y_cursor, right1 - left0, h_frac]
            if first_ax is None:
                ax = fig.add_axes(rect)
                first_ax = ax
            else:
                ax = fig.add_axes(rect, sharex=first_ax)
            axes_list.append(ax)

            if (
                dendrogram_active
                and dendrogram_position == "after_markers"
                and sec_pos == 0
            ):
                y_cursor -= dendrogram_gap_in / fig_h
                rect_d = [
                    left0,
                    y_cursor - dendrogram_h_in / fig_h,
                    right1 - left0,
                    dendrogram_h_in / fig_h,
                ]
                ax_dendrogram = fig.add_axes(rect_d, sharex=first_ax)
                y_cursor -= dendrogram_h_in / fig_h
            if sec_pos < len(sections) - 1:
                y_cursor -= gap_frac

        ax_totals = None
        if show_totals and first_ax is not None:
            rect_tot = [
                left0,
                y_cursor - totals_gap_in / fig_h - totals_h_in / fig_h,
                right1 - left0,
                totals_h_in / fig_h,
            ]
            ax_totals = fig.add_axes(rect_tot, sharex=first_ax)
            y_cursor = rect_tot[1]

        if dendrogram_active and dendrogram_position == "end" and first_ax is not None:
            y_cursor -= dendrogram_gap_in / fig_h
            rect_d = [
                left0,
                y_cursor - dendrogram_h_in / fig_h,
                right1 - left0,
                dendrogram_h_in / fig_h,
            ]
            ax_dendrogram = fig.add_axes(rect_d, sharex=first_ax)
            y_cursor -= dendrogram_h_in / fig_h

        ax_spatial = None
        if spatial_active and first_ax is not None:
            rect_spatial = [
                left0,
                y_cursor - spatial_gap_in / fig_h - spatial_h_in / fig_h,
                right1 - left0,
                spatial_h_in / fig_h,
            ]
            ax_spatial = fig.add_axes(rect_spatial)

        # --- render each section -------------------------------------------------
        for sec_i, (sec, ax) in enumerate(zip(sections, axes_list)):
            n_f = len(sec["features"])
            ax.set_ylim(-0.5, n_f - 0.5)
            ax.set_xlim(-0.5, n_clusters - 0.5)
            ax.set_yticks(range(n_f))
            display_labels = [
                label_overrides.get(f, f) if label_overrides else f
                for f in sec["features"]
            ]
            ax.set_yticklabels(display_labels, fontsize=feature_label_fontsize)
            ax.invert_yaxis()  # first-listed feature at the top, matching reading order
            ax.tick_params(bottom=False, top=False, left=False, length=0)
            for spine in ("top", "right", "bottom", "left"):
                ax.spines[spine].set_visible(False)
            ax.set_title(
                sec["title"], fontsize=section_title_fontsize, pad=section_title_pad
            )
            ax.grid(axis="y", visible=False)
            if sec["type"] == "dots":
                ax.grid(axis="x", color="#e8e8e8", linewidth=0.5, zorder=0)
            else:
                ax.grid(axis="x", visible=False)
            ax.set_axisbelow(True)

            if sec_i == 0:
                ax.xaxis.set_ticks_position("top")
                ax.xaxis.set_label_position("top")
                ax.set_xticks(range(n_clusters))
                ax.set_xticklabels(
                    cluster_display_labels,
                    rotation=-90,
                    fontsize=cluster_label_fontsize,
                    ha="center",
                )
                if palette is not None and strip_h_in > 0:
                    # the color strip sits directly on this axes' own top edge (same spot tick
                    # labels anchor to by default) -- push labels up past it, mirroring how the
                    # non-transpose layout pads its y-tick labels past the left-hand strip.
                    ax.tick_params(
                        axis="x", pad=(strip_h_in + strip_gap_in) * 72.0 + 2.0
                    )
            else:
                ax.tick_params(labeltop=False, labelbottom=False)

            if sec["type"] == "dots":
                nm, fr = sec["norm_means"], sec["fracs"]
                xs = np.tile(np.arange(n_clusters), n_f)
                ys = np.repeat(np.arange(n_f), n_clusters)
                ax.scatter(
                    xs,
                    ys,
                    c=nm.T.ravel(),
                    s=fr.T.ravel() * max_dot_size,
                    cmap=sec["cmap"],
                    vmin=dot_vmin,
                    vmax=dot_vmax,
                    linewidths=0.3,
                    edgecolors=_style.edge_color_default,
                    zorder=3,
                    clip_on=False,
                )
                for hx in highlight_y_indices:
                    ax.axvspan(
                        hx - 0.5,
                        hx + 0.5,
                        color=_style.accent_color_positive,
                        alpha=0.15,
                        zorder=0,
                    )
            else:
                if sec["type"] == "cont":
                    data_img, vmin_im, vmax_im = sec["norm_raw"].T, 0.0, 1.0
                else:
                    data_img, vmin_im, vmax_im = sec["raw"].T, 0.0, sec["vmax"]
                ax.imshow(
                    data_img,
                    aspect="auto",
                    cmap=sec["cmap"],
                    vmin=vmin_im,
                    vmax=vmax_im,
                    origin="upper",
                    interpolation="nearest",
                )
                if annotate_obs:
                    for ci in range(n_clusters):
                        for fi in range(n_f):
                            val = sec["raw"][ci, fi]
                            if np.isnan(val):
                                continue
                            if sec["type"] == "flags":
                                pct = val * 100
                                txt = (
                                    f"{pct:.0f}%"
                                    if pct >= 10 or pct == 0
                                    else f"{pct:.1f}%"
                                )
                            elif sec["type"] == "cont":
                                cm = sec["col_max_raw"][fi]
                                if cm > 10:
                                    txt = f"{val:.0f}"
                                elif cm > 1:
                                    txt = f"{val:.1f}"
                                else:
                                    txt = f"{val:.2f}"
                            else:
                                txt = f"{val:.1f}"
                            ax.text(
                                ci,
                                fi,
                                txt,
                                ha="center",
                                va="center",
                                fontsize=annotation_fontsize,
                                color="black",
                                fontweight=(
                                    "bold" if ci in highlight_y_indices else "normal"
                                ),
                            )

        # --- totals bar chart (vertical) ------------------------------------------
        if ax_totals is not None:
            frac_of_total = cl_sizes / max(cl_sizes.sum(), 1)
            for ci in range(n_clusters):
                edge_color, edge_width = "none", 0.0
                if ci in highlight_y_indices:
                    bar_color = (palette or {}).get(
                        unique_clusters[ci], _style.accent_color_positive
                    )
                    if _is_light_color(bar_color):
                        edge_color, edge_width = "black", 0.8
                elif flag_small_clusters and frac_of_total[ci] < min_cluster_frac:
                    bar_color = _style.accent_color_negative
                else:
                    bar_color = "#7b8da8"
                ax_totals.bar(
                    ci,
                    cl_sizes[ci],
                    color=bar_color,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    width=0.65,
                )
            ax_totals.set_ylim(0, max(cl_sizes) * 1.12)
            ax_totals.set_xlim(-0.5, n_clusters - 0.5)
            ax_totals.set_title(
                "Cluster size", fontsize=section_title_fontsize, pad=section_title_pad
            )
            ax_totals.set_ylabel("n cells", fontsize=feature_label_fontsize)
            ax_totals.tick_params(labelbottom=False, bottom=False, length=0)
            ax_totals.grid(False)
            for spine in ("top", "right", "bottom"):
                ax_totals.spines[spine].set_visible(False)
            ax_totals.yaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else str(int(x))
                )
            )
            ax_totals.tick_params(axis="y", labelsize=feature_label_fontsize)

        # --- dendrogram (optional, below Clustering markers or at the very end) ---
        if dendrogram_active and ax_dendrogram is not None:
            dd = _scipy_dendrogram(dendrogram_Z, no_plot=True)
            icoord = np.asarray(dd["icoord"], dtype=float)
            dcoord = np.asarray(dd["dcoord"], dtype=float)
            dendro_x = (
                icoord - 5.0
            ) / 10.0  # scipy's leaf spacing -> column index 0..n-1
            for xi, yi in zip(dendro_x, dcoord):
                ax_dendrogram.plot(
                    xi,
                    yi,
                    color="#555555",
                    linewidth=0.9,
                    solid_capstyle="round",
                    clip_on=False,
                )
            ax_dendrogram.set_xlim(-0.5, n_clusters - 0.5)
            max_h = float(dcoord.max()) if dcoord.size else 1.0
            ax_dendrogram.set_ylim(0, max_h * 1.05 if max_h > 0 else 1.0)
            ax_dendrogram.invert_yaxis()  # leaves (distance 0) at the top, root at the bottom
            ax_dendrogram.set_axis_off()

        # --- spatial panel (optional, always last/bottom-most) --------------------
        if ax_spatial is not None:
            region_mask = adata.obs[region_col].astype(str) == str(spatial_region)
            region_df = adata.obs.loc[region_mask]
            region_group_str = region_df[groupby].astype(str)
            highlighted_mask = region_group_str.isin(highlight_cluster_strs)

            ax_spatial.scatter(
                region_df.loc[~highlighted_mask, x_col],
                region_df.loc[~highlighted_mask, y_col],
                c="#CCCCCC",
                s=spatial_point_size,
                alpha=spatial_grey_alpha,
                linewidths=0,
                rasterized=True,
            )

            spatial_counts = {}
            for hc in highlight_cluster_strs:
                cluster_mask = region_group_str == hc
                spatial_counts[hc] = int(cluster_mask.sum())
                if cluster_mask.any():
                    color = (palette or {}).get(hc, _style.accent_color_positive)
                    ax_spatial.scatter(
                        region_df.loc[cluster_mask, x_col],
                        region_df.loc[cluster_mask, y_col],
                        color=[color],
                        s=spatial_point_size,
                        alpha=spatial_color_alpha,
                        linewidths=0,
                        rasterized=True,
                        label=hc,
                    )

            if len(highlight_cluster_strs) == 1:
                hc = highlight_cluster_strs[0]
                spatial_title = (
                    f"{hc} -- {spatial_region} (n={spatial_counts.get(hc, 0)})"
                )
            else:
                count_text = ", ".join(
                    f"{hc}: n={spatial_counts[hc]}" for hc in highlight_cluster_strs
                )
                spatial_title = f"{', '.join(highlight_cluster_strs)} -- {spatial_region} ({count_text})"
            ax_spatial.set_title(spatial_title, fontsize=section_title_fontsize)
            ax_spatial.set_xlabel(x_col, fontsize=feature_label_fontsize)
            ax_spatial.set_ylabel(y_col, fontsize=feature_label_fontsize)
            ax_spatial.invert_yaxis()
            ax_spatial.set_aspect("equal", adjustable="datalim")
            ax_spatial.tick_params(axis="both", labelsize=legend_fontsize)
            _apply_spines(ax_spatial, _style)

        # --- decide where the dot-size legend goes --------------------------------
        # Preferably inside the 1st dot section's own colorbar band, directly below its
        # (measured, not guessed) label height -- that's where it sits in the reference figure
        # this layout is modeled on, and it's almost always the tallest section. Falls through to
        # the 2nd dot section (e.g. `extra_markers`) if the 1st is too short, then to the space
        # just below the totals/spatial band, and only as a last resort (no totals, no spatial,
        # and no dot section tall enough) is it forced into the last dot section anyway.
        dot_label_h_in = _label_h_in_by_type[
            "dots"
        ]  # computed earlier alongside every section's own label
        legend_block_in = 0.95
        host_gap_in = 0.12

        dots_positions = [i for i, s in enumerate(sections) if s["type"] == "dots"]
        legend_host_i = None
        for i in dots_positions:
            band_h_in = axes_list[i].get_position().height * fig_h
            if band_h_in >= dot_label_h_in + host_gap_in + legend_block_in:
                legend_host_i = i
                break
        if legend_host_i is None and ax_totals is None and ax_spatial is None:
            legend_host_i = dots_positions[-1]

        # --- colorbars (left margin, vertical, aligned with each section's y-extent) ---
        cb_x0_frac = (feat_label_block_in + left_of_labels_gap_in) / fig_w
        cb_x_fig = left0 - cb_x0_frac - cb_block_in / fig_w
        cb_w_frac = cb_w_in / fig_w
        legend_bbox_y = None
        for sec_i, (sec, ax) in enumerate(zip(sections, axes_list)):
            ax_pos = ax.get_position()
            if sec_i == legend_host_i:
                # shrink this section's colorbar to just what its label needs, and reserve the
                # remainder of the band (still within THIS section's own y-extent, not the gap to
                # the next section) for the legend directly below it.
                cb_h_frac = dot_label_h_in / fig_h
                cb_y0 = ax_pos.y0 + ax_pos.height - cb_h_frac
                legend_bbox_y = (
                    ax_pos.y0 + ax_pos.height - cb_h_frac - host_gap_in / fig_h
                )
            else:
                cb_h_frac = ax_pos.height * 0.78
                cb_y0 = ax_pos.y0 + ax_pos.height * 0.08
            cb_ax = fig.add_axes([cb_x_fig, cb_y0, cb_w_frac, cb_h_frac])
            if sec["type"] == "dots":
                smap = plt.cm.ScalarMappable(
                    cmap=sec["cmap"], norm=plt.Normalize(dot_vmin, dot_vmax)
                )
                cb = fig.colorbar(smap, cax=cb_ax, orientation="vertical")
                cb.set_label(dot_label_text, fontsize=legend_fontsize)
            elif sec["type"] == "cont":
                smap = plt.cm.ScalarMappable(cmap=sec["cmap"], norm=plt.Normalize(0, 1))
                cb = fig.colorbar(smap, cax=cb_ax, orientation="vertical", ticks=[0, 1])
                cb.ax.set_yticklabels(["low", "high"])
                cb.set_label(_label_text_by_type["cont"], fontsize=legend_fontsize)
            else:
                smap = plt.cm.ScalarMappable(
                    cmap=sec["cmap"], norm=plt.Normalize(0, sec["vmax"])
                )
                cb = fig.colorbar(
                    smap,
                    cax=cb_ax,
                    orientation="vertical",
                    format=plt.FuncFormatter(lambda x, _: f"{x:.0%}"),
                )
                cb.set_label("fraction flagged", fontsize=legend_fontsize)
            cb.ax.tick_params(labelsize=legend_fontsize)

        dot_sections = [s for s in sections if s["type"] == "dots"]
        if dot_sections:
            if legend_bbox_y is None:
                ref_ax = ax_totals if ax_totals is not None else ax_spatial
                ref_pos = ref_ax.get_position()
                legend_bbox_y = ref_pos.y0 + ref_pos.height
            sizes_legend = [0.25, 0.5, 1.0]
            labels_legend = ["25%", "50%", "100%"]
            handles = [
                plt.scatter(
                    [],
                    [],
                    s=f * max_dot_size,
                    c="grey",
                    linewidths=0.3,
                    edgecolors=_style.edge_color_default,
                    label=lbl,
                )
                for f, lbl in zip(sizes_legend, labels_legend)
            ]
            fig.legend(
                handles=handles,
                title="frac > thr",
                loc="upper left",
                bbox_to_anchor=(cb_x_fig, legend_bbox_y),
                bbox_transform=fig.transFigure,
                frameon=False,
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
            )

        if title:
            fig.text(
                0.5,
                1.0 - (0.12 / fig_h),
                title,
                ha="center",
                va="top",
                fontsize=section_title_fontsize + 2,
            )

        return fig


def _outline_other_cells(
    ax, mask_crop, target_label, color="yellow", alpha=1.0, linewidth=0.8
):
    for label in set(np.unique(mask_crop)) - {0, target_label}:
        ax.contour(mask_crop == label, colors=color, linewidths=linewidth, alpha=alpha)


def _percentile_normalize(arr, percentiles=(0.5, 99.5)):
    lo, hi = np.percentile(arr, percentiles)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def _crop_2d(arr2d, cy, cx, patch_h, patch_w, pad_value=0):
    height, width = arr2d.shape
    y0, x0 = cy - patch_h // 2, cx - patch_w // 2
    y1, x1 = y0 + patch_h, x0 + patch_w
    src_y0, src_x0 = max(y0, 0), max(x0, 0)
    src_y1, src_x1 = min(y1, height), min(x1, width)
    crop = arr2d[src_y0:src_y1, src_x0:src_x1]
    pad = ((src_y0 - y0, y1 - src_y1), (src_x0 - x0, x1 - src_x1))
    if pad != ((0, 0), (0, 0)):
        crop = np.pad(crop, pad, mode="constant", constant_values=pad_value)
    return crop


def _crop_out_of_bounds_mask(height, width, cy, cx, patch_h, patch_w):
    """True where a patch pixel falls outside the source array -- used to paint those pixels
    white instead of the black they'd get from _crop_2d's zero-padding (e.g. a patch centered
    near a core's edge)."""
    y0, x0 = cy - patch_h // 2, cx - patch_w // 2
    yy = np.arange(y0, y0 + patch_h)
    xx = np.arange(x0, x0 + patch_w)
    valid_y = (yy >= 0) & (yy < height)
    valid_x = (xx >= 0) & (xx < width)
    return ~(valid_y[:, None] & valid_x[None, :])


def _composite_marker_patch(
    image_dict,
    dapi_name,
    markers,
    marker_colors_rgb,
    cy,
    cx,
    patch_h,
    patch_w,
    percentiles,
    dapi_weight,
    marker_weight,
):
    composite = np.zeros((patch_h, patch_w, 3), dtype=np.float32)
    if dapi_name is not None:
        dapi = _percentile_normalize(
            _crop_2d(image_dict[dapi_name], cy, cx, patch_h, patch_w), percentiles
        )
        composite += (dapi * dapi_weight)[:, :, None]
    for marker in markers:
        signal = _percentile_normalize(
            _crop_2d(image_dict[marker], cy, cx, patch_h, patch_w), percentiles
        )
        composite += (signal * marker_weight)[:, :, None] * marker_colors_rgb[marker][
            None, None, :
        ]
    return np.clip(composite, 0.0, 1.0)


def _label_to_cluster_map_from_obs(obs_sub, groupby, mask, x_col, y_col):
    if len(obs_sub) == 0:
        return {}
    height, width = mask.shape
    ys = obs_sub[y_col].round().astype(int).clip(0, height - 1).to_numpy()
    xs = obs_sub[x_col].round().astype(int).clip(0, width - 1).to_numpy()
    groups = obs_sub[groupby].astype(str).to_numpy()
    out = {}
    for lbl, grp in zip(mask[ys, xs], groups):
        if lbl != 0:
            out[int(lbl)] = grp
    return out


def _label_to_cluster_map(adata, groupby, mask, region_bool, x_col, y_col):
    return _label_to_cluster_map_from_obs(
        adata.obs.loc[region_bool], groupby, mask, x_col, y_col
    )


def _auto_phenotype_colors(values):
    """tab20 for <=20 categories, gist_ncar beyond that -- matches the reference implementation's
    `_resolve_phenotype_colors`."""
    values = list(values)
    cmap = plt.get_cmap("tab20" if len(values) <= 20 else "gist_ncar")
    if len(values) <= 1:
        return {values[0]: cmap(0.0)} if values else {}
    return {v: cmap(i / (len(values) - 1)) for i, v in enumerate(values)}


def _collapse_phenotype_legend(all_values, phenotype_colors_rgb):
    """Group legend entries by identical color, collapsing any group of >1 values that all
    share a common "<prefix>,<suffix>" naming (e.g. every recluster,0 .. recluster,25 assigned
    the same grey) into one "<prefix>,x" entry -- keeps a long list of subcluster IDs sharing a
    single fallback color from bloating the legend. Groups that share a color but don't share
    that naming pattern are left as individual entries."""
    groups = {}
    for v in all_values:
        color = phenotype_colors_rgb[v]
        color_key = "none" if color is None else tuple(np.round(color, 4))
        groups.setdefault(color_key, []).append(v)

    entries = []
    for values in groups.values():
        color = phenotype_colors_rgb[values[0]]
        if len(values) == 1:
            entries.append((values[0], color))
            continue
        prefixes = {v.rsplit(",", 1)[0] for v in values if "," in v}
        if len(prefixes) == 1 and len(values) == sum("," in v for v in values):
            entries.append((f"{prefixes.pop()},x", color))
        else:
            entries.extend((v, color) for v in values)
    return entries


def _phenotype_fill_rgba(mask_patch, label_to_cluster, color_for_cluster, alpha=0.82):
    """`color_for_cluster(cluster_value) -> rgb or None` -- None leaves that cell transparent
    (shows through as background). Covers both the binary highlight-vs-other mode and the full
    per-phenotype multi-color mode via the closure passed in by the caller."""
    rgba = np.zeros((*mask_patch.shape, 4), dtype=np.float32)
    for lbl in np.unique(mask_patch):
        if lbl == 0:
            continue
        cl = label_to_cluster.get(int(lbl))
        if cl is None:
            continue
        rgb = color_for_cluster(cl)
        if rgb is None:
            continue
        rgba[mask_patch == lbl] = (*rgb, alpha)
    return rgba


def patch_QC(
    adata,
    groupby,
    cluster,
    markers,
    masks,
    images,
    *,
    dapi_name="DAPI",
    DAPI_on=True,
    region=None,
    n_patches=10,
    patch_size=(150, 150),
    n_cols=2,
    region_col="unique_region",
    x_col="x",
    y_col="y",
    marker_colors=None,
    show_segmentation=True,
    other_boundary_alpha=0.7,
    selected_boundary_alpha=1.0,
    highlight_color="#3a7fd5",
    other_color="#777777",
    palette=None,
    whole_image_obs=None,
    intensity_percentiles=(0.5, 99.5),
    dapi_weight=0.45,
    marker_weight=0.5,
    whole_image_mode="dots",
    whole_image_region=None,
    overview_max_dim=2500,
    overview_dot_size=1.5,
    overview_highlight_dot_size=6.0,
    image_names=None,
    return_selection=False,
    seed=0,
    title=None,
    style=None,
    cell_in=1.55,
    gap_in=0.55,
    legend_width_in=1.5,
    fontsize=None,
):
    """
    Inspect a cluster's spatial context: sampled marker-image patches, the same patches with
    cells colored by cluster identity, and a whole-image overview -- side by side in one figure,
    to sanity-check whether a cluster's marker
    signal and spatial position match its intended identity before committing to an annotation.

    Parameters
    ----------
    adata : AnnData
    groupby : str
        obs column containing cluster labels.
    cluster : str
        The cluster label (within `groupby`) to inspect.
    markers : Sequence[str]
        Marker names to render as a colored composite over DAPI in the patch grid. Intended for
        2-5 markers; outside that range a warning is raised (not an error).
    masks : dict[str, np.ndarray]
        Maps `region_col` value -> 2D integer label mask for that region (0 = background).
    images : dict[str, dict[str, np.ndarray]]
        Maps `region_col` value -> {channel name: 2D image array}, e.g.
        `sp.hf.hf_load_image_dict(file_name, channel_file, nuclei_channel=dapi_name)` per region.
        Matched by channel *name*, not position -- deliberately not a stacked
        `(n_channels, height, width)` array indexed by `adata.var_names` order, since that
        silently breaks whenever a channel was dropped between the raw image and `adata.var_names`
        (e.g. via `nonFuncAb_list` in `sp.hf.make_anndata`) without erroring or warning.
    dapi_name : str, optional
        Channel name (a key into each region's `images` dict) used as the DAPI background. By
        default "DAPI".
    DAPI_on : bool, optional
        Include the DAPI background in the marker-patch composite. By default True.
    region : str, optional
        Restrict everything -- sampled patches and the whole-image overview -- to this one
        `region_col` value. Use this to call `patch_QC` once per image of interest (e.g. once per
        pre-loaded representative core) rather than mixing cells from every region in `images`
        into one call. Defaults to using all regions present in `images`. See `run_patch_QC` for
        an orchestrator that picks the `region` (or regions) for you, including a rare-cluster fallback.
    n_patches : int, optional
        Number of cells to sample from `cluster` for the patch grids. By default 10.
    patch_size : tuple[int, int], optional
        (height, width) in pixels of each patch. By default (150, 150).
    n_cols : int, optional
        Patches per row in the marker/mask grids. By default 2 (so `n_patches=10` renders as 2x5).
    marker_colors : dict, optional
        Maps marker name to a display color. Defaults to a `tab10` cycle.
    show_segmentation : bool, optional
        Draw segmentation-boundary outlines on the marker and mask patches. By default True.
    other_boundary_alpha, selected_boundary_alpha : float, optional
        Opacity of the boundary outline for non-target cells vs. the sampled (selected) cell
        itself. By default 0.7 and 1.0.
    dapi_weight, marker_weight : float, optional
        Blend weights for the DAPI background and marker tint respectively, applied after each
        channel is independently percentile-normalized to [0, 1] -- higher values saturate to
        white faster where markers/DAPI overlap. By default 0.45 and 0.5.
    highlight_color, other_color : str, optional
        Colors for the inspected cluster vs. every other cell, used in the mask-colored patches
        and the whole-image overview -- ignored when `palette` is given.
    palette : dict or "auto", optional
        Switches the mask-colored patches and whole-image overview from the default binary
        highlight-vs-other scheme to a full per-phenotype color scheme, one color per distinct
        `groupby` value (not just the inspected cluster) -- useful once cells are fully annotated
        and you want to see every cell type at once rather than just this one cluster vs. the
        rest. Pass a dict (`{value: color}`) or `"auto"` to auto-generate one (tab20, or
        gist_ncar beyond 20 categories). Any value missing from an explicit dict is filled in
        automatically. By default None (binary highlight mode).
    whole_image_obs : pandas.DataFrame, optional
        Use this obs table -- instead of `adata.obs` -- to color neighboring (non-sampled) cells
        in both the mask-colored patches and the whole-image overview. Pass the *unfiltered* obs
        here if `adata` has already had some cells dropped (e.g. QC/noise filtering), so
        neighboring cells still render with their true identity instead of showing as unresolved
        background. Must have the same `region_col`/`x_col`/`y_col`/`groupby` columns as `adata`.
    whole_image_mode : {"dots", "mask"}, optional
        How to render the whole-image overview panel. "dots" (default) plots cell centroids as a
        fast scatter -- use this for large images where rendering the full segmentation mask
        would be slow. "mask" renders the actual downsampled segmentation mask fill, more
        faithful but much slower on large cores.
    whole_image_region : str, optional
        Which region to show in the overview panel. Defaults to `region` if given, else whichever
        region contains the most sampled patch cells.
    overview_max_dim : int, optional
        In "mask" mode, the mask is downsampled so its largest dimension doesn't exceed this, to
        keep rendering fast. By default 2500.
    overview_dot_size : float, optional
        Marker size for non-target cells in "dots" mode. By default 1.5.
    overview_highlight_dot_size : float, optional
        Marker size for the inspected cluster's own cells in "dots" mode -- bigger than
        `overview_dot_size` by default so the target cluster stays visible in dense tissue where a
        matching small dot size would nearly disappear into the background. By default 6.0.
    image_names : dict or Sequence, optional
        Display name for the region shown, used as the figure title and the overview panel's own
        title instead of the bare `region_col` value. Pass a dict (`{region: name}`) or a list
        aligned with `images`' key order. Defaults to the region key itself -- callers iterating
        over several regions typically derive a nicer default themselves (e.g. the final part of
        each region's source file path) and pass it in here.
    return_selection : bool, optional
        Also return the sampled cells' obs rows (with a `patch_number` column matching the number
        shown in the figure), e.g. for record-keeping or a follow-up `filter_from_mixed_clusters`
        call. By default False.
    seed : int, optional
        Random seed for patch sampling. By default 0.
    title : str, optional
        Overall figure title. Defaults to the resolved display name (see `image_names`).
    style : SPACEcStyle or str or pathlib.Path, optional
        Style to use for this call -- drives the marker-color palette when `marker_colors` is
        left at its default (resolved via the shared palette system), and the returned figure's
        `dpi` (`_style.figure_dpi`). There's no `dpi=` parameter here anymore: this function never
        calls `savefig` itself (per this project's plotting contract -- return `fig`, save it
        outside), so a fixed dpi baked into the signature only ever controlled the in-notebook
        preview resolution, not any saved output; `fig.savefig(..., dpi=...)` at the call site
        (or `run_patch_QC`'s own `save_dpi`) always overrides it anyway. Every other visual
        parameter here (patch geometry, legend/annotation fontsizes, ...) is still this
        function's own explicit argument or a hardcoded literal, not yet routed through the
        shared style system.

    cell_in : float, optional
        Side length (inches) of one patch. By default 1.55. With `gap_in` and `legend_width_in`
        this sets the figure size, so smaller values give a print-sized figure.
    gap_in : float, optional
        Gap (inches) between the marker grid, the mask grid and the overview. By default 0.55.
    legend_width_in : float, optional
        Space (inches) right of the overview for the two legends. By default 1.5.
    fontsize : float or None, optional
        Base font size (pt) for patch titles, legends and overview labels; block headers and
        the overview title use fontsize + 1, the figure title fontsize + 2. None (default) keeps
        the previous sizes (9/8/7, headers 10, title 12).

    Returns
    -------
    matplotlib.figure.Figure, or (Figure, pandas.DataFrame) if `return_selection=True`
    """
    _style = _apply_style(style)
    if whole_image_mode not in ("dots", "mask"):
        raise ValueError('whole_image_mode must be "dots" or "mask"')
    if not (2 <= len(markers) <= 5):
        warnings.warn(
            f"patch_QC is intended for 2-5 markers; got {len(markers)}. Plotting them anyway.",
            stacklevel=2,
        )

    markers = list(markers)
    required_channels = set(markers) | ({dapi_name} if DAPI_on else set())
    for region_key, region_channels in images.items():
        missing = sorted(required_channels - set(region_channels))
        if missing:
            raise ValueError(
                f"images[{region_key!r}] is missing channel(s) {missing} -- available: "
                f"{sorted(region_channels)}"
            )

    if marker_colors is None:
        _marker_palette = _get_palette(_style, len(markers))
        marker_colors = {
            m: _marker_palette[i % len(_marker_palette)] for i, m in enumerate(markers)
        }
    marker_colors_rgb = {
        m: np.asarray(mcolors.to_rgb(marker_colors[m]), dtype=np.float32)
        for m in markers
    }
    highlight_rgb = mcolors.to_rgb(highlight_color)
    other_rgb = mcolors.to_rgb(other_color)
    patch_h, patch_w = patch_size

    groups = adata.obs[groupby].astype(str)
    cluster = str(cluster)
    regions_all = adata.obs[region_col].astype(str)
    is_cluster = (groups == cluster).to_numpy()
    if region is not None:
        is_cluster &= (regions_all == str(region)).to_numpy()
    candidate_idx = np.flatnonzero(is_cluster)
    if candidate_idx.size == 0:
        scope = f" in region {region!r}" if region is not None else ""
        raise ValueError(f"No cells found with {groupby}={cluster!r}{scope}")

    rng = np.random.default_rng(seed)
    n_sel = min(n_patches, candidate_idx.size)
    if n_sel < n_patches:
        warnings.warn(
            f"Only {n_sel} cells available for {groupby}={cluster!r} (requested n_patches={n_patches})",
            stacklevel=2,
        )
    sel_idx = rng.choice(candidate_idx, size=n_sel, replace=False)
    selected = adata.obs.iloc[sel_idx].copy()
    # one shared 1-indexed number per sampled patch, used as both the grid title and the
    # whole-image overview annotation, so "patch 4" means the same cell in every panel -- rather
    # than the overview re-numbering just its own (possibly region-filtered) subset from 1.
    selected["_patch_number"] = np.arange(1, n_sel + 1)
    regions = regions_all

    # --- phenotype coloring: binary highlight-vs-other (default), or a full per-value palette ---
    multi_color = palette is not None
    if multi_color:
        all_values = set(groups.unique())
        if whole_image_obs is not None:
            all_values |= set(whole_image_obs[groupby].astype(str).unique())
        all_values = sorted(all_values)
        if palette == "auto":
            palette = _auto_phenotype_colors(all_values)
        else:
            missing_vals = [v for v in all_values if v not in palette]
            if missing_vals:
                palette = {**palette, **_auto_phenotype_colors(missing_vals)}
        # a value explicitly mapped to None (rather than a color string) is left uncolored --
        # transparent in the mask fill and skipped in the whole-image overview -- e.g. for a
        # category like "Noise" that shouldn't visually compete with real phenotypes.
        palette_rgb = {
            v: (
                np.asarray(mcolors.to_rgb(palette[v]), dtype=np.float32)
                if palette[v] is not None
                else None
            )
            for v in all_values
        }

        def _color_for(cl):
            return palette_rgb.get(cl)

    else:

        def _color_for(cl):
            return highlight_rgb if cl == cluster else other_rgb

    n_rows = max(1, math.ceil(n_sel / n_cols))

    label_maps = {}

    def _label_map_for(region_key):
        if region_key not in label_maps:
            region_bool = (regions == region_key).to_numpy()
            label_maps[region_key] = _label_to_cluster_map(
                adata, groupby, masks[region_key], region_bool, x_col, y_col
            )
        return label_maps[region_key]

    def _neighbor_label_map_for(region_key):
        """Neighbor-cell identities for coloring -- from `whole_image_obs` (unfiltered) if given,
        else the same `adata`-derived map used for the sampled cells themselves."""
        if whole_image_obs is None:
            return _label_map_for(region_key)
        obs_sub = whole_image_obs[whole_image_obs[region_col].astype(str) == region_key]
        return _label_to_cluster_map_from_obs(
            obs_sub, groupby, masks[region_key], x_col, y_col
        )

    # --- figure layout: marker grid | mask grid | whole-image overview, all fixed inches ---
    fs_title = 9 if fontsize is None else fontsize
    fs_legend = 8 if fontsize is None else fontsize
    fs_number = 7 if fontsize is None else fontsize
    fs_header = 10 if fontsize is None else fontsize + 1
    fs_suptitle = 12 if fontsize is None else fontsize + 2
    # each row gets its own title slot above the patch image, so the per-patch number can be a
    # proper title instead of overlapping text -- without this gap, row r+1's title would render
    # on top of row r's image, since axes tiled with zero spacing leave no room for it.
    title_gap_in = 0.26 if fontsize is None else fontsize * 2.1 / 72
    row_stride_in = cell_in + title_gap_in
    # small white gap between columns within a grid -- without it, adjacent patches (e.g. patch 1
    # and patch 2) touch directly with nothing but a thin axes-spine line between them.
    col_gap_in = 0.08
    col_stride_in = cell_in + col_gap_in
    margin_l, margin_r, margin_t, margin_b = 0.15, legend_width_in, 0.75, 0.15
    grid_w = n_cols * cell_in + (n_cols - 1) * col_gap_in
    grid_h = n_rows * row_stride_in
    overview_in = max(grid_h, 2.2)

    fig_w = margin_l + grid_w + gap_in + grid_w + gap_in + overview_in + margin_r
    fig_h = margin_t + max(grid_h, overview_in) + margin_b
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=_style.figure_dpi)

    def _grid_axes(x0_in, n_r, n_c):
        axes = np.empty((n_r, n_c), dtype=object)
        for r in range(n_r):
            for c in range(n_c):
                x = (x0_in + c * col_stride_in) / fig_w
                y = (fig_h - margin_t - r * row_stride_in - cell_in) / fig_h
                axes[r, c] = fig.add_axes([x, y, cell_in / fig_w, cell_in / fig_h])
        return axes

    marker_axes = _grid_axes(margin_l, n_rows, n_cols)
    mask_axes = _grid_axes(margin_l + grid_w + gap_in, n_rows, n_cols)
    overview_x0 = margin_l + 2 * grid_w + 2 * gap_in
    ax_overview = fig.add_axes(
        [
            overview_x0 / fig_w,
            (fig_h - margin_t - overview_in) / fig_h,
            overview_in / fig_w,
            overview_in / fig_h,
        ]
    )

    # --- marker patches + mask-colored patches ---
    for i, (_, row) in enumerate(selected.iterrows()):
        r, c = divmod(i, n_cols)
        patch_number = int(row["_patch_number"])
        # named patch_region (not `region`) so this doesn't shadow the outer `region` parameter --
        # a `for` loop variable in Python isn't block-scoped, so a plain `region =` here used to
        # silently clobber the function parameter for every line below the loop, breaking the
        # `whole_image_region = region if region is not None else ...` fallback whenever `region`
        # was left as None (the loop always left it not-None, pointing at whichever patch happened
        # to be sampled last).
        patch_region = str(row[region_col])
        cy, cx = int(round(row[y_col])), int(round(row[x_col]))
        image_dict = images[patch_region]
        mask = masks[patch_region]

        ax_m = marker_axes[r, c]
        composite = _composite_marker_patch(
            image_dict,
            dapi_name if DAPI_on else None,
            markers,
            marker_colors_rgb,
            cy,
            cx,
            patch_h,
            patch_w,
            intensity_percentiles,
            dapi_weight,
            marker_weight,
        )
        # patches sampled near a core's edge extend past the raw image bounds; paint those
        # pixels white (matching the rest of the figure) instead of the black _crop_2d's
        # zero-padding would otherwise bake into the composite.
        image_h, image_w = next(iter(image_dict.values())).shape
        out_of_bounds = _crop_out_of_bounds_mask(
            image_h, image_w, cy, cx, patch_h, patch_w
        )
        composite[out_of_bounds] = 1.0
        ax_m.imshow(composite, interpolation="nearest")
        mask_patch = _crop_2d(mask, cy, cx, patch_h, patch_w)
        target_label = mask[min(cy, mask.shape[0] - 1), min(cx, mask.shape[1] - 1)]
        if show_segmentation:
            _outline_other_cells(
                ax_m,
                mask_patch,
                target_label,
                color="white",
                alpha=other_boundary_alpha,
            )
            if target_label != 0:
                ax_m.contour(
                    mask_patch == target_label,
                    colors="red",
                    linewidths=1.2,
                    alpha=selected_boundary_alpha,
                )
        ax_m.set_xticks([])
        ax_m.set_yticks([])
        ax_m.set_title(str(patch_number), fontsize=fs_title, pad=2)

        ax_p = mask_axes[r, c]
        ax_p.set_facecolor("white")
        label_to_cluster = _neighbor_label_map_for(patch_region)
        ax_p.imshow(
            _phenotype_fill_rgba(mask_patch, label_to_cluster, _color_for),
            interpolation="nearest",
        )
        # cell boundaries on top of the flat phenotype fill -- without these, same-colored
        # neighboring cells merge into one undifferentiated blob and individual cells (their
        # actual segmentation shapes) aren't visible. Black (not white, unlike the marker panel)
        # since this panel's background is now white -- white outlines would vanish into it.
        if show_segmentation:
            _outline_other_cells(
                ax_p,
                mask_patch,
                target_label,
                color="black",
                alpha=other_boundary_alpha,
            )
            if target_label != 0:
                ax_p.contour(
                    mask_patch == target_label,
                    colors="red",
                    linewidths=1.2,
                    alpha=selected_boundary_alpha,
                )
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        ax_p.set_title(str(patch_number), fontsize=fs_title, pad=2)

    for i in range(n_sel, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        marker_axes[r, c].set_axis_off()
        mask_axes[r, c].set_axis_off()

    # Both legends live in the right margin (like the reference implementation's own
    # `bbox_to_anchor=(0.84, 0.5)` placement) rather than above the grids -- anchoring them above
    # the grid instead made their fixed content width collide with the block-title text once the
    # grid became narrow (n_cols=2), since the title is centered over a much smaller span while
    # the legend's own width stayed the same regardless of grid width.
    legend_x = (fig_w - margin_r + 0.15) / fig_w
    marker_legend = (
        [mpatches.Patch(facecolor="#bdbdbd", edgecolor="none", label="DAPI")]
        if DAPI_on
        else []
    ) + [
        mpatches.Patch(facecolor=marker_colors_rgb[m], edgecolor="none", label=m)
        for m in markers
    ]
    fig.legend(
        handles=marker_legend,
        loc="upper left",
        bbox_to_anchor=(legend_x, (fig_h - margin_t) / fig_h),
        frameon=False,
        fontsize=fs_legend,
        title="markers",
        title_fontsize=fs_legend,
        borderaxespad=0,
    )
    # fixed per-line height estimate at fontsize 8, so the second legend's vertical offset scales
    # with the first legend's actual entry count instead of assuming a specific marker count.
    marker_legend_h_in = (0.30 + 0.19 * len(marker_legend)) * fs_legend / 8
    if multi_color:
        # a None color (deliberately left uncolored/transparent, e.g. "Noise") gets a white
        # swatch with a thin outline in the legend -- otherwise it'd be an invisible white-on-
        # white box, indistinguishable from empty space.
        phenotype_legend = [
            mpatches.Patch(
                facecolor="white" if color is None else color,
                edgecolor="black" if color is None else "none",
                linewidth=0.6 if color is None else 0,
                label=label,
            )
            for label, color in _collapse_phenotype_legend(all_values, palette_rgb)
        ]
    else:
        phenotype_legend = [
            mpatches.Patch(
                facecolor=highlight_color, edgecolor="none", label=str(cluster)
            ),
            mpatches.Patch(facecolor=other_color, edgecolor="none", label="other"),
        ]
    fig.legend(
        handles=phenotype_legend,
        loc="upper left",
        bbox_to_anchor=(legend_x, (fig_h - margin_t - marker_legend_h_in) / fig_h),
        frameon=False,
        fontsize=fs_legend,
        title=groupby,
        title_fontsize=fs_legend,
        borderaxespad=0,
    )

    # --- whole-image overview ---
    if whole_image_region is None:
        whole_image_region = (
            region
            if region is not None
            else selected[region_col].astype(str).value_counts().idxmax()
        )

    display_name = None
    if image_names:
        if isinstance(image_names, dict):
            display_name = image_names.get(whole_image_region)
        else:
            keys = list(images.keys())
            if whole_image_region in keys:
                idx = keys.index(whole_image_region)
                if idx < len(image_names):
                    display_name = image_names[idx]
    if display_name is None:
        display_name = whole_image_region
    if title is None:
        title = f"{cluster} ({display_name})"

    ax_overview.set_facecolor("white")
    ax_overview.set_xticks([])
    ax_overview.set_yticks([])
    overview_mask = masks[whole_image_region]
    height, width = overview_mask.shape

    scale_y, scale_x = 1.0, 1.0
    if whole_image_mode == "mask":
        downsample = max(1, int(np.ceil(max(height, width) / overview_max_dim)))
        display_mask = overview_mask[::downsample, ::downsample]
        label_to_cluster = _neighbor_label_map_for(whole_image_region)
        ax_overview.imshow(
            _phenotype_fill_rgba(display_mask, label_to_cluster, _color_for, alpha=1.0),
            interpolation="nearest",
            origin="upper",
        )
        title_suffix = f" (1:{downsample})" if downsample > 1 else ""
        scale_y = display_mask.shape[0] / height
        scale_x = display_mask.shape[1] / width
    else:
        if whole_image_obs is not None:
            obs_sub = whole_image_obs[
                whole_image_obs[region_col].astype(str) == whole_image_region
            ]
        else:
            region_bool = (regions == whole_image_region).to_numpy()
            obs_sub = adata.obs.loc[region_bool]
        ax_overview.set_xlim(0, width)
        ax_overview.set_ylim(height, 0)
        if multi_color:
            obs_groups = obs_sub[groupby].astype(str)
            for v in all_values:
                if palette_rgb[v] is None:
                    continue
                sub = obs_sub[obs_groups == v]
                if sub.empty:
                    continue
                size = (
                    overview_highlight_dot_size if v == cluster else overview_dot_size
                )
                zorder = 3 if v == cluster else 2
                ax_overview.scatter(
                    sub[x_col],
                    sub[y_col],
                    s=size,
                    c=[palette_rgb[v]],
                    linewidths=0,
                    zorder=zorder,
                )
        else:
            is_target = (obs_sub[groupby].astype(str) == cluster).to_numpy()
            ax_overview.scatter(
                obs_sub.loc[~is_target, x_col],
                obs_sub.loc[~is_target, y_col],
                s=overview_dot_size,
                c=[other_color],
                linewidths=0,
                zorder=2,
            )
            ax_overview.scatter(
                obs_sub.loc[is_target, x_col],
                obs_sub.loc[is_target, y_col],
                s=overview_highlight_dot_size,
                c=[highlight_color],
                linewidths=0,
                zorder=3,
            )
        title_suffix = ""

    # mark + number the sampled patch cells on the overview -- reusing each cell's `_patch_number`
    # (its position in the grids above) rather than re-counting from 1 within this region-filtered
    # subset, so "patch 4" here is the same cell as grid position 4 even when the sampled cluster
    # spans multiple regions.
    sel_overview = selected[selected[region_col].astype(str) == whole_image_region]
    for _, row in sel_overview.iterrows():
        j = int(row["_patch_number"])
        px, py = row[x_col] * scale_x, row[y_col] * scale_y
        ax_overview.plot(
            px,
            py,
            marker="o",
            markersize=5,
            markeredgecolor="red",
            markerfacecolor="none",
            markeredgewidth=1.2,
            zorder=4,
        )
        ax_overview.annotate(
            str(j),
            (px, py),
            color="red",
            fontsize=fs_number,
            zorder=5,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax_overview.set_title(f"{display_name}{title_suffix}", fontsize=fs_header)
    for spine in ax_overview.spines.values():
        spine.set_visible(False)

    fig.text(
        (margin_l + grid_w / 2) / fig_w,
        (fig_h - margin_t + 0.35) / fig_h,
        "marker patches",
        ha="center",
        fontsize=fs_header,
    )
    fig.text(
        (margin_l + grid_w + gap_in + grid_w / 2) / fig_w,
        (fig_h - margin_t + 0.35) / fig_h,
        f"{groupby} masks",
        ha="center",
        fontsize=fs_header,
    )
    fig.text(
        (overview_x0 + overview_in / 2) / fig_w,
        (fig_h - margin_t + 0.35) / fig_h,
        "whole-image overview",
        ha="center",
        fontsize=fs_header,
    )
    if title:
        fig.suptitle(title, fontsize=fs_suptitle, y=1.0 - (0.05 / fig_h))

    if return_selection:
        return fig, selected.rename(columns={"_patch_number": "patch_number"})
    return fig


def _select_regions_for_cluster(
    adata,
    groupby,
    cluster,
    region_col,
    *,
    main_regions=None,
    example_regions=None,
    min_cells=25,
    show_all_regions=False,
):
    """Tiered region/core selection for a cluster, ported from the reference implementation's
    round-7/8 patch-vis block (generalised there into `run_cluster_patch_vis`): prefer the
    smallest/most-convenient pre-loaded set that actually has enough cells, falling back to a
    dataset-wide stratified sample for rare clusters so they still get a representative spread
    instead of an empty or single-region view."""
    cluster_obs = adata.obs[adata.obs[groupby].astype(str) == str(cluster)]
    if cluster_obs.empty:
        raise ValueError(f"No cells found with {groupby}={cluster!r}")
    counts = (
        cluster_obs.groupby(region_col, observed=True)
        .size()
        .sort_values(ascending=False)
    )

    if show_all_regions:
        return counts.index.tolist()

    if main_regions:
        if counts.reindex(main_regions).fillna(0).sum() >= min_cells:
            return list(main_regions)
    if example_regions:
        if counts.reindex(example_regions).fillna(0).sum() >= min_cells:
            return list(example_regions)

    # Rare-cluster fallback: stratified sample across the whole dataset -- 2 most abundant, 2
    # middle, 2 least abundant regions (max 6), so even a rare cluster gets a representative
    # spread rather than just its single most-abundant region.
    all_regions = counts.index.tolist()
    if len(all_regions) <= 6:
        return all_regions
    top2, bot2, mid = all_regions[:2], all_regions[-2:], all_regions[2:-2]
    mid2 = [mid[len(mid) // 2 - 1], mid[len(mid) // 2]] if len(mid) >= 2 else mid
    seen, picked = set(), []
    for r in top2 + mid2 + bot2:
        if r not in seen:
            seen.add(r)
            picked.append(r)
    return picked


def run_patch_QC(
    adata,
    groupby,
    clusters,
    marker_panel,
    masks,
    images,
    *,
    image_paths=None,
    channel_file=None,
    mask_paths=None,
    region_col="unique_region",
    main_regions=None,
    example_regions=None,
    min_cells=25,
    show_all_regions=False,
    out_dir=None,
    save_dpi=300,
    title_prefix="",
    **patch_qc_kwargs,
):
    """
    Batch-run `patch_QC` across multiple clusters, choosing which region(s) to show per cluster
    via tiered abundance-based selection: use `main_regions` if the cluster has >= `min_cells` cells there,
    else `example_regions` if it has enough there, else fall back to a stratified <= 6-region
    sample across the whole dataset (so rare clusters still get a representative view instead of
    silently coming up near-empty or confined to one over-represented region).
    `show_all_regions=True` overrides all of this and renders every region the cluster appears
    in, uncapped -- for manually curating a cluster you suspect is inconsistent across
    batches/regions.

    Parameters
    ----------
    adata : AnnData
    groupby : str
        obs column containing cluster labels.
    clusters : Sequence[str]
        Cluster labels to inspect -- one full `patch_QC` figure per cluster per selected region.
    marker_panel : Sequence[str] or dict
        A single marker list used for every cluster, or a `{cluster: markers}` dict -- clusters
        missing from the dict are skipped with a warning.
    masks : dict[str, np.ndarray]
        Pre-loaded regions, as in `patch_QC`.
    images : dict[str, dict[str, np.ndarray]]
        Pre-loaded regions, as in `patch_QC` (`{region: {channel name: 2D array}}`).
        Regions you pass in `masks`/`images` are never evicted and are reused as-is across every
        cluster in this call. Any *additional* region loaded on demand during this call (see
        `image_paths`/`mask_paths`) is reused across clusters within the same call too, but is
        evicted from `images`/`masks` again before this function returns -- so a rare cluster's
        stratified fallback pulling in extra regions from a large WSI dataset can't leave them
        sitting in memory after the call, only the regions you explicitly chose to keep
        pre-loaded survive it.
    image_paths : dict[str, str], optional
        Maps `region_col` value -> multi-channel image file path, used to lazily load a region
        (via `sp.hf.hf_load_image_dict`, so it comes in already keyed by channel name, matching
        `images`' own format) if the tiered selection picks one not already pre-loaded. Requires
        `channel_file`. Only actually read from disk once per call, even if multiple clusters in
        that call need it -- see `masks`/`images` above for why it doesn't persist past the call.
        Regions the tiering picks that are neither pre-loaded nor in this path map are skipped
        with a warning.
    channel_file : str or pathlib.Path, optional
        Path to the channel-names file (one name per line) shared by every region in
        `image_paths`, passed to `sp.hf.hf_load_image_dict`. Required if `image_paths` is given.
    mask_paths : dict[str, str], optional
        Maps `region_col` value -> mask file path, used to lazily load a region into `masks` the
        same way, via a plain `tifffile.imread` (masks have no channel-name concept).
    main_regions, example_regions : Sequence[str], optional
        Tier 1 / tier 2 preferred region sets, checked in that order before falling back to the
        stratified whole-dataset sample.
    min_cells : int, optional
        Minimum cluster-cell count for a tier to be used as-is. By default 25.
    show_all_regions : bool, optional
        Override the tiering and use every region the cluster appears in (no cap). By default
        False.
    out_dir : str or pathlib.Path, optional
        If given, save each figure to `out_dir/<cluster_stub>/` instead of returning them (and
        close the figure immediately after, to keep memory bounded across a big batch).
    save_dpi : int, optional
        Save dpi (only used when `out_dir` is given). By default 300.
    title_prefix : str, optional
        Prepended to each saved figure's filename.
    **patch_qc_kwargs
        Passed through to every `patch_QC` call (e.g. `n_patches`, `whole_image_mode`,
        `palette`, `whole_image_obs`, `dapi_name`).

    Returns
    -------
    dict[str, dict[str, matplotlib.figure.Figure]]
        `{cluster: {region: fig}}` for clusters/regions actually rendered -- empty (figures are
        saved and closed instead) if `out_dir` was given.
    """
    out_path = pathlib.Path(out_dir) if out_dir is not None else None
    results = {}
    # remember what the caller actually pre-loaded -- anything this call loads on demand beyond
    # that (e.g. a rare cluster's stratified fallback pulling in a region from a large WSI
    # dataset) gets evicted again before returning, so a batch call can't silently balloon
    # `images`/`masks` past what the user chose to keep warm. Still loaded only once even if
    # multiple clusters in this same call need the same on-demand region.
    preloaded_image_keys = set(images)
    preloaded_mask_keys = set(masks)

    for cluster in clusters:
        if isinstance(marker_panel, dict):
            markers = marker_panel.get(cluster)
            if markers is None:
                warnings.warn(
                    f"no marker panel for {cluster!r}; skipping", stacklevel=2
                )
                continue
        else:
            markers = list(marker_panel)

        try:
            regions_to_use = _select_regions_for_cluster(
                adata,
                groupby,
                cluster,
                region_col,
                main_regions=main_regions,
                example_regions=example_regions,
                min_cells=min_cells,
                show_all_regions=show_all_regions,
            )
        except ValueError as exc:
            print(f"  {cluster}: {exc}")
            continue

        cluster_results = {}
        for r in regions_to_use:
            if r not in images:
                if image_paths and r in image_paths:
                    if channel_file is None:
                        raise ValueError(
                            "image_paths was given but channel_file was not -- required to "
                            "load images by channel name via sp.hf.hf_load_image_dict."
                        )
                    print(f"  {cluster} | {r}: loading image (not pre-loaded)...")
                    images[r] = hf_load_image_dict(
                        image_paths[r],
                        channel_file,
                        nuclei_channel=patch_qc_kwargs.get("dapi_name", "DAPI"),
                    )
                else:
                    warnings.warn(
                        f"cluster {cluster!r}: region {r!r} not pre-loaded in `images` and no "
                        "image_paths entry for it; skipping",
                        stacklevel=2,
                    )
                    continue
            if r not in masks:
                if mask_paths and r in mask_paths:
                    import tifffile

                    masks[r] = tifffile.imread(mask_paths[r])
                else:
                    warnings.warn(
                        f"cluster {cluster!r}: region {r!r} not pre-loaded in `masks` and no "
                        "mask_paths entry for it; skipping",
                        stacklevel=2,
                    )
                    continue

            n_cells = int(
                (
                    (adata.obs[groupby].astype(str) == str(cluster)).to_numpy()
                    & (adata.obs[region_col].astype(str) == str(r)).to_numpy()
                ).sum()
            )
            print(f"  {cluster} | {r} ({n_cells} cells)")

            fig = patch_QC(
                adata,
                groupby,
                cluster,
                markers,
                masks,
                images,
                region=r,
                region_col=region_col,
                **patch_qc_kwargs,
            )
            if out_path is not None:
                cluster_stub = (
                    str(cluster).replace(" ", "_").replace("/", "__").replace(",", "-")
                )
                cluster_dir = out_path / cluster_stub
                cluster_dir.mkdir(parents=True, exist_ok=True)
                region_stub = str(r).replace("/", "__")
                fig.savefig(
                    cluster_dir / f"{title_prefix}{cluster_stub}_{region_stub}.png",
                    dpi=save_dpi,
                    bbox_inches="tight",
                )
                plt.close(fig)
            else:
                cluster_results[r] = fig
        if cluster_results:
            results[cluster] = cluster_results

    # evict anything loaded on demand during this call that the caller didn't pre-load --
    # already-rendered figures hold their own rasterized copy, not a live reference into these
    # arrays, so this is safe even when figures were returned rather than saved.
    for r in set(images) - preloaded_image_keys:
        del images[r]
    for r in set(masks) - preloaded_mask_keys:
        del masks[r]

    return results


# ---------------------------------------------------------------------------
# cluster_qc
# ---------------------------------------------------------------------------

# obs columns that identify a cell rather than describe it -- plotting them
# per cluster shows nothing.
_QC_EXCLUDED_OBS = (
    "x",
    "y",
    "z",
    "label",
    "unique_cell_id",
    "cell_id",
    "region_num",
    "first_index",
    "tile_num",
    "x_tile",
    "y_tile",
)


def _qc_default_features(adata):
    """Pick the numeric ``adata.obs`` columns worth comparing across clusters.

    Keeps measured per-cell properties (cell area, nuclear-stain intensity,
    and any morphology columns the user chose to keep) and drops coordinates
    and identifier columns.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.

    Returns
    -------
    list of str
        Column names, in ``adata.obs`` order.
    """
    features = []
    for name in adata.obs.columns:
        if name in _QC_EXCLUDED_OBS:
            continue
        col = adata.obs[name]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        n_unique = col.nunique()
        # <3 distinct values carries no distribution. An integer column with
        # one distinct value per cell is an identifier; a float one is just a
        # continuous measurement, which is exactly what we want to plot.
        if n_unique < 3:
            continue
        if pd.api.types.is_integer_dtype(col) and n_unique == adata.n_obs:
            continue
        features.append(name)
    return features


def _qc_default_flags(adata):
    """Pick the boolean ``adata.obs`` columns worth comparing across clusters (QC flags).

    Complements `_qc_default_features`, not a subset of it: a boolean column has at most 2
    distinct values, so `_qc_default_features`'s own `n_unique < 3` rule already excludes every
    boolean column -- it was never going to double as a flag detector.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.

    Returns
    -------
    list of str
        Column names, in ``adata.obs`` order.
    """
    return [
        name
        for name in adata.obs.columns
        if name not in _QC_EXCLUDED_OBS and pd.api.types.is_bool_dtype(adata.obs[name])
    ]


def _qc_flag_label_overrides(flag_cols):
    """Default `extended_dotplot` `label_overrides` for QC-flag obs columns.

    Mirrors the shortening convention from the clustering tutorial's own
    ``QC_LABEL_OVERRIDES`` (drop the redundant ``qc_flag_`` prefix -- it's already inside the
    "% flagged in QC" panel title -- and ``noise`` -> ``noisy`` since it's only a
    flag/suspicion), generalized to whatever flag columns are actually present instead of a
    fixed hardcoded list.

    Parameters
    ----------
    flag_cols : Sequence[str]
        Column names to build overrides for (e.g. from `_qc_default_flags`). A name without
        the ``qc_flag_`` prefix is left out -- nothing to shorten.

    Returns
    -------
    dict
        Maps each ``qc_flag_*`` name in `flag_cols` to its shortened display label.
    """
    overrides = {}
    for name in flag_cols:
        if not name.startswith("qc_flag_"):
            continue
        short = name[len("qc_flag_") :]
        overrides[name] = "noisy" if short == "noise" else short
    return overrides


def cluster_qc(
    adata,
    cluster_key,
    features=None,
    highlight_cluster=None,
    show_outliers=False,
    figsize=None,
    savefig=False,
    output_fname="",
    output_dir="./",
    return_data=False,
    style=None,
):
    """
    Compare clusters on the per-cell properties measured during segmentation.

    Draws one box per cluster per feature, with a dashed reference line at the
    median across all cells. Cell area and nuclear-stain intensity are
    recorded for every cell during segmentation, so a cluster made of
    segmentation fragments or of cells truncated at the tissue edge shows up
    as visibly smaller or dimmer than every real cluster.

    This function reports measurements only. It applies no thresholds, emits
    no flags and draws no conclusions -- a cluster looking different is a
    reason to look closer, not by itself a reason to delete it.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cluster_key : str
        Column in ``adata.obs`` holding the cluster or cell-type labels.
    features : list of str, optional
        Numeric ``adata.obs`` columns to plot, one panel each. Defaults to
        every numeric obs column except coordinates and identifiers -- on the
        standard SPACEc pipeline that is ``area`` and the nuclear channel
        (``DAPI``, ``Hoechst1``, ...), plus any morphology columns kept via
        ``pp.format(list_keep=...)``.
    highlight_cluster : str or int or list, optional
        Cluster label(s) to draw in the highlight color. Matched as strings,
        so ``7`` and ``"7"`` are equivalent. Purely cosmetic.
    show_outliers : bool, optional
        Whether to draw the boxplot flier points, by default False. Cell area
        and intensity have long tails, and drawing every outlier compresses
        the boxes until they are unreadable.
    figsize : tuple, optional
        ``(width, height)`` in inches. Derived from the panel and cluster
        count if omitted.
    savefig : bool, optional
        Whether to save the figure, by default False.
    output_fname : str, optional
        Base filename without extension, by default ``""``.
    output_dir : str, optional
        Directory to save into, by default ``"./"``.
    return_data : bool, optional
        If True, also return the per-cluster summary table, by default False.
    style : SPACEcStyle or str or pathlib.Path, optional
        Style override for this call. Uses the active style if omitted.

    Returns
    -------
    matplotlib.figure.Figure or tuple
        The figure, with one panel (``fig.axes[i]``) per feature. If
        *return_data* is True, returns ``(fig, summary)`` where ``summary``
        is a DataFrame indexed by cluster label with ``n_cells`` and the
        median of each feature, plus a final ``"all cells"`` row.

    Raises
    ------
    ValueError
        If *cluster_key* is missing from ``adata.obs``, or if no plottable
        numeric obs column was found.

    See Also
    --------
    spacec.plotting.marker_dotplot : marker expression per cluster, and
        ``highlight_cluster`` to see where a cluster sits in the tissue.
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(
            f"'{cluster_key}' is not a column of adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    if features is None:
        features = _qc_default_features(adata)
        if not features:
            raise ValueError(
                "No numeric adata.obs column found to compare. Pass features="
                "[...] explicitly. Available obs columns: "
                f"{list(adata.obs.columns)}"
            )
    else:
        features = list(features)
        missing = [f for f in features if f not in adata.obs.columns]
        if missing:
            raise ValueError(f"Not in adata.obs: {missing}")

    _style = _apply_style(style)

    groups = adata.obs[cluster_key]
    if isinstance(groups.dtype, pd.CategoricalDtype):
        clusters = list(groups.cat.categories)
    else:
        # dropna() before sorting -- a plain (non-categorical) column can carry a real NaN,
        # and sorting a list that mixes strings with a float NaN raises "'<' not supported
        # between instances of 'float'/'numpy.str_' and the other" (the same bug fixed in
        # catplot/plot_masks/the 8-function palette sweep earlier this session).
        clusters = sorted(groups.dropna().unique().tolist())

    # Labels are commonly passed as strings while obs stores them as
    # categorical or integer, so compare as strings throughout.
    groups_str = groups.astype(str)

    if highlight_cluster is None:
        highlight = set()
    elif isinstance(highlight_cluster, (list, tuple, set, np.ndarray, pd.Index)):
        highlight = {str(c) for c in highlight_cluster}
    else:
        highlight = {str(highlight_cluster)}

    n_features = len(features)

    if figsize is None:
        width = max(_style.figure_width_double, 0.32 * len(clusters) + 1.5)
        height = 1.6 * n_features + 0.8
    else:
        width, height = figsize

    with _style.context():
        fig, axes_arr = plt.subplots(
            n_features,
            1,
            figsize=(width, height),
            sharex=True,
            squeeze=False,
        )
        axes = list(axes_arr[:, 0])

        for ax, feature in zip(axes, features):
            values = adata.obs[feature]
            per_cluster = [
                values[(groups_str == str(c)).to_numpy()].dropna().to_numpy()
                for c in clusters
            ]

            bp = ax.boxplot(
                per_cluster,
                positions=np.arange(len(clusters)),
                widths=0.6,
                patch_artist=True,
                showfliers=show_outliers,
                medianprops=dict(color=_style.edge_color_default, linewidth=1.0),
            )

            for patch, cluster in zip(bp["boxes"], clusters):
                patch.set_facecolor(
                    _style.accent_color_negative
                    if str(cluster) in highlight
                    else _style.accent_color_neutral
                )
                patch.set_edgecolor(_style.edge_color_default)
                patch.set_linewidth(0.6)

            ax.axhline(
                float(values.median()),
                color=_style.edge_color_default,
                linestyle="--",
                linewidth=0.8,
                label="median of all cells",
            )

            ax.set_ylabel(feature, fontsize=_style.font_size_small)
            ax.tick_params(axis="both", labelsize=_style.font_size_tiny)
            _apply_spines(ax, _style)

        axes[0].legend(loc="best", fontsize=_style.font_size_tiny)
        _style_legend(axes[0], _style)

        axes[-1].set_xticks(np.arange(len(clusters)))
        axes[-1].set_xticklabels(
            [str(c) for c in clusters],
            rotation=90,
            fontsize=_style.font_size_tiny,
        )
        axes[-1].set_xlabel(cluster_key, fontsize=_style.font_size_small)

        fig.align_ylabels(axes)

        if savefig:
            _savefig(fig, output_dir, output_fname, "_cluster_qc", _style)

    if return_data:
        summary = pd.DataFrame(
            {
                "n_cells": [int((groups_str == str(c)).sum()) for c in clusters],
                **{
                    feature: [
                        float(
                            adata.obs.loc[
                                (groups_str == str(c)).to_numpy(), feature
                            ].median()
                        )
                        for c in clusters
                    ]
                    for feature in features
                },
            },
            index=[str(c) for c in clusters],
        )
        summary.index.name = cluster_key
        summary.loc["all cells"] = [adata.n_obs] + [
            float(adata.obs[f].median()) for f in features
        ]
        return fig, summary

    return fig


def pl_generate_CN_comb_map(
    graph,
    tops,
    e0,
    e1,
    simp_freqs,
    palette,
    figsize=(40, 20),
    savefig=False,
    output_dir="./",
    style=None,
):
    """
    Generate a cellular neighbourhood combination map as a network graph.

    Draws the directed graph of neighbourhood combinations where node sizes
    reflect combination frequencies and node colours indicate cellular
    neighbourhood membership.

    Parameters
    ----------
    graph : networkx.DiGraph
        Directed graph of neighbourhood combinations.
    tops : list
        List of top combinations to mark with an asterisk.
    e0 : object
        Source node of the first edge (used for edge iteration; ignored in
        the current implementation where all edges are iterated directly).
    e1 : object
        Target node of the first edge (see ``e0``).
    simp_freqs : pandas.Series
        Series mapping neighbourhood-combination tuples to their relative
        frequencies.
    palette : dict
        Dictionary mapping cellular neighbourhood indices to colours.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (40, 20).
    savefig : bool, optional
        If True, save the figure to ``output_dir``. Default is False.
    output_dir : str, optional
        Directory for saving the figure. Default is ``"./"``.
    style : SPACEcStyle or None, optional
        SPACEc style configuration object. Default is None.

    Returns
    -------
    None
    """
    _style = _apply_style(style)
    draw = graph
    pos = _graphviz_layout(draw, prog="dot", seed=1)
    height = 8

    plt.figure(figsize=figsize)
    for n in draw.nodes():
        col = "black"
        if len(draw.in_edges(n)) < len(n):
            col = "black"
        plt.scatter(
            pos[n][0],
            pos[n][1] - 5,
            s=simp_freqs[list(simp_freqs.index).index(n)] * 10000,
            c=col,
            zorder=-1,
        )
        if n in tops:
            plt.text(
                pos[n][0],
                pos[n][1] - 7,
                "*",
                fontsize=25,
                color="white",
                ha="center",
                va="center",
                zorder=20,
            )
        delta = 8

        # l is just the color keys
        l = list(palette.keys())
        plt.scatter(
            [pos[n][0]] * len(n),
            [pos[n][1] + delta * (i + 1) for i in range(len(n))],
            c=[palette[l[i]] for i in n],
            marker="s",
            zorder=5,
            s=400,
        )

    j = 0
    for e0, e1 in draw.edges():
        weight = 0.2
        alpha = 0.3
        color = "black"
        if len(draw.in_edges(e1)) < len(e1):
            color = "black"
            lw = 1
            weight = 0.4

        plt.plot(
            [pos[e0][0], pos[e1][0]],
            [pos[e0][1], pos[e1][1]],
            color=color,
            linewidth=weight,
            alpha=alpha,
            zorder=-10,
        )

    plt.axis("off")

    if savefig:
        _savefig(None, output_dir, "_CNMap", "", _style)

    plt.show()


def stacked_bar_plot(
    adata,
    color,
    grouping,
    cell_list,
    output_dir,
    norm=True,
    savefig=False,  # new
    output_fname="",  # new
    col_order=None,
    sub_col=None,
    name_cat="celltype",
    fig_sizing=(8, 4),
    plot_order=None,
    palette=None,
    remove_leg=False,
    style=None,
):
    """
    Plot a stacked bar plot based on the given data.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing the necessary information for plotting.
    color : str
        The column name representing the categories.
    grouping : str
        The column name representing the grouping.
    cell_list : list
        The list of cell types to include in the plot.
    output_dir : str
        The output directory for saving the plot.
    norm : bool, optional
        Flag indicating whether to normalize the values. Defaults to True.
    save_name : str, optional
        The name to use when saving the plot. Defaults to None.
    col_order : list, optional
        The order of columns/categories for plotting. Defaults to None.
    sub_col : str, optional
        The column name representing sub-categories. Defaults to None.
    name_cat : str, optional
        The name for the category column in the plot. Defaults to 'celltype'.
    fig_sizing : tuple, optional
        The size of the figure (width, height) in inches. Defaults to (8, 4).
    plot_order : list, optional
        The order of categories for plotting. Defaults to None.
    palette : dict, optional
        A dictionary mapping categories to colors for custom colorization -- any category
        not covered is colored grey, with a warning. Defaults to None, which reuses/creates
        a persistent mapping cached in `adata.uns[color + "_colors"]`, consistent with any
        other function (e.g. `catplot`, `extended_dotplot`) already plotting `color`.
    remove_leg : bool, optional
        Flag indicating whether to remove the legend. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        The pivoted data used for plotting.
    list
        The order of categories used for plotting.
    """

    _style = _apply_style(style)
    data = adata.obs

    # Find Percentage of cell type
    if norm == True:
        if sub_col is None:
            test1 = data.loc[data[color].isin(cell_list)]
            sub_cell_list = list(test1[color].unique())
        else:
            test1 = data.loc[data[sub_col].isin(cell_list)]
            sub_cell_list = list(test1[color].unique())
    else:
        if sub_col is None:
            test1 = data.copy()
            sub_cell_list = list(data.loc[data[color].isin(cell_list)][color].unique())
        else:
            test1 = data.copy()
            sub_cell_list = list(
                data.loc[data[sub_col].isin(cell_list)][color].unique()
            )
    palette = _resolve_categorical_palette(
        _style, adata, color, sub_cell_list, palette=palette
    )

    test1[color] = test1[color].astype("category")
    test_freq = test1.groupby(grouping).apply(
        lambda x: x[color].value_counts(normalize=True, sort=False) * 100
    )
    test_freq.columns = test_freq.columns.astype(str)

    test_freq.reset_index(inplace=True)
    sub_cell_list.append(grouping)
    test_freq = test_freq[sub_cell_list]
    melt_test = pd.melt(
        test_freq, id_vars=[grouping]
    )  # , value_vars=test_freq.columns)
    # melt_test.rename(columns={per_cat: name_cat, "value": "percent"}, inplace=True)
    melt_test.rename(columns={"value": "percent"}, inplace=True)

    if norm == True:
        if col_order is None:
            bb = melt_test.groupby([grouping, color]).sum().reset_index()
            col_order = (
                bb.loc[bb[color] == bb[color][0]]
                .sort_values(by="percent")[grouping]
                .to_list()
            )
    else:
        if col_order is None:
            col_order = (
                melt_test.groupby(grouping)
                .sum()
                .reset_index()
                .sort_values(by="percent")[grouping]
                .to_list()
            )

    if plot_order is None:
        plot_order = list(melt_test[color].unique())

    # Set up for plotting
    melt_test_piv = pd.pivot_table(
        melt_test, columns=[color], index=[grouping], values=["percent"]
    )
    melt_test_piv.columns = melt_test_piv.columns.droplevel(0)
    melt_test_piv.reset_index(inplace=True)
    melt_test_piv.set_index(grouping, inplace=True)
    melt_test_piv = melt_test_piv.reindex(col_order)
    melt_test_piv = melt_test_piv[plot_order]

    # Get color dictionary
    ax1 = melt_test_piv.plot.bar(
        alpha=0.8,
        linewidth=1,
        color=[palette.get(x) for x in melt_test_piv.columns],
        figsize=fig_sizing,
        rot=90,
        stacked=True,
        edgecolor="black",
    )

    for line in ax1.lines:
        line.set_color("black")
    _apply_spines(ax1, _style)
    if remove_leg == True:
        ax1.set_ylabel("")
        ax1.set_xlabel("")
    else:
        ax1.set_ylabel("percent", fontsize=_style.font_size_small)
    plt.xticks(
        list(range(len(list(melt_test_piv.index)))),
        list(melt_test_piv.index),
        rotation=90,
        fontsize=_style.font_size_tiny,
    )
    lgd2 = ax1.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        frameon=_style.legend_frameon,
        fontsize=_style.font_size_small,
    )
    if savefig:
        _savefig(None, output_dir, output_fname, "", _style)

    plt.show()
    return melt_test_piv, plot_order


_SWARM_BOX_USAGE = """
How to use swarm_box_plot (same idea as sp.pl.stacked_bar_plot, plus `replicate`):

    df, plot_order = sp.pl.swarm_box_plot(
        adata,
        color="celltype_fine",      # adata.obs column with the categories (cell types, neighborhoods, communities)
        grouping="condition",       # adata.obs column for the x-axis groups (condition, tissue, ...)
        replicate="patient",        # adata.obs column with the replicates -> one dot each, nested in `grouping`
        cell_list=["GCB", "Treg"],  # LIST of categories to plot, one panel each
        output_dir="./",
        savefig=False,
    )
"""


def swarm_box_plot(
    adata,
    color,
    grouping,
    replicate,
    cell_list,
    output_dir,
    norm=True,
    savefig=False,
    output_fname="",
    col_order=None,
    sub_col=None,
    fig_sizing=None,
    plot_order=None,
    palette=None,
    n_columns=3,
    point_size=5,
    style=None,
):
    """
    Plot the abundance of categories per group as box plots with one dot per replicate.

    The companion of :func:`stacked_bar_plot`: it takes the same arguments, but
    instead of one stacked bar per group it draws one panel per category (e.g. a
    cell type, neighborhood or community). In each panel the x-axis shows the
    groups (e.g. conditions), each dot is one replicate (e.g. a patient or
    region) and the box summarizes the replicates. The returned long-format
    table has one row per group x replicate x category and can be used directly
    for statistics.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object; ``adata.obs`` holds one row per cell.
    color : str
        Column in ``adata.obs`` with the categories whose abundance is plotted
        (cell types, neighborhoods, communities, ...).
    grouping : str
        Column in ``adata.obs`` defining the x-axis groups (e.g. condition).
    replicate : str
        Column in ``adata.obs`` defining the replicates (e.g. patient, sample,
        region). Each replicate should belong to exactly one group and every
        group needs at least two replicates for a meaningful box.
    cell_list : list
        Categories to plot, one panel each. Values of ``color``, or of
        ``sub_col`` if that is given.
    output_dir : str
        Output directory for the plot (only used if ``savefig=True``).
    norm : bool, optional
        If True (default), percentages are computed within the subset selected
        by ``cell_list`` (they sum to 100 per group and replicate). If False,
        percentages are relative to all cells of that group and replicate.
    savefig : bool, optional
        Save the figure to ``output_dir``. Defaults to False.
    output_fname : str, optional
        File name stem used when saving; ``_swarm_boxplot`` is appended.
        Defaults to "".
    col_order : list, optional
        Order of the groups on the x-axis. Defaults to the categorical order of
        ``grouping`` if it is categorical, otherwise sorted order. Groups not
        listed are left out of the plot and the returned table.
    sub_col : str, optional
        Column in ``adata.obs`` to subset by instead of ``color``: ``cell_list``
        then holds values of ``sub_col`` (e.g. coarse cell types) and the
        panels are the ``color`` categories (e.g. fine cell types) found in
        them. Defaults to None.
    fig_sizing : tuple, optional
        Figure size (width, height) in inches. Defaults to None, which scales
        the figure with the number of panels.
    plot_order : list, optional
        Order of the panels (categories). Defaults to the order of ``cell_list``
        (or of first appearance when ``sub_col`` is used).
    palette : dict, optional
        Dictionary mapping categories to colors -- any category not covered is
        colored grey, with a warning. Defaults to None, which reuses/creates a
        persistent mapping cached in ``adata.uns[color + "_colors"]``,
        consistent with any other function (e.g. ``catplot``,
        ``stacked_bar_plot``) already plotting ``color``.
    n_columns : int, optional
        Maximum number of panels per row. Defaults to 3.
    point_size : float, optional
        Marker size (points) of the replicate dots. Defaults to 5.
    style : SPACEcStyle or None, optional
        SPACEc style configuration object. Defaults to the active style.

    Returns
    -------
    pandas.DataFrame
        Long-format table with the columns ``grouping``, ``replicate``,
        ``color`` (the category) and ``percent``. A replicate without any cell
        of a category gets ``0``, not a missing row.
    list
        The order of the categories (panels) used for plotting.

    Raises
    ------
    ValueError
        If an argument is used incorrectly (unknown column, ``cell_list`` not a
        list or without matches, ``replicate`` equal to ``grouping``, ...). The
        message explains how to call the function.

    Examples
    --------
    >>> df, order = sp.pl.swarm_box_plot(
    ...     adata,
    ...     color="celltype_fine",
    ...     grouping="condition",
    ...     replicate="patient",
    ...     cell_list=["GCB", "Treg"],
    ...     output_dir="./",
    ... )
    >>> # statistics on the returned table, e.g. a Mann-Whitney U test for GCB
    >>> from scipy.stats import mannwhitneyu
    >>> gcb = df[df["celltype_fine"] == "GCB"]
    >>> a, b = [g["percent"].values for _, g in gcb.groupby("condition")]
    >>> mannwhitneyu(a, b)
    """

    def fail(msg):
        raise ValueError(f"{msg}\n{_SWARM_BOX_USAGE}")

    if not hasattr(adata, "obs"):
        fail(
            "`adata` must be an AnnData object (the cell table lives in "
            f"`adata.obs`), got {type(adata).__name__}."
        )
    _style = _apply_style(style)
    data = adata.obs

    for name, col in (
        ("color", color),
        ("grouping", grouping),
        ("replicate", replicate),
        ("sub_col", sub_col),
    ):
        if col is None and name == "sub_col":
            continue
        if not isinstance(col, str) or col not in data.columns:
            fail(
                f"`{name}={col!r}` is not a column of adata.obs. "
                f"Available columns: {list(data.columns)}."
            )
    if replicate == grouping:
        fail(
            "`replicate` and `grouping` are the same column. `grouping` defines the "
            "x-axis groups, `replicate` the dots inside each group (e.g. patient)."
        )
    if not isinstance(cell_list, (list, tuple, set, np.ndarray, pd.Index)):
        fail(
            f"`cell_list` must be a list of categories, got {cell_list!r}. "
            f'For one category use cell_list=["{cell_list}"].'
        )
    cell_list = list(cell_list)

    search_col = color if sub_col is None else sub_col
    present = list(pd.unique(data[search_col].dropna()))
    found = [c for c in cell_list if c in present]
    if not found:
        fail(
            f"None of the values in `cell_list` {cell_list} occur in "
            f"adata.obs[{search_col!r}]. Example values: {present[:10]}."
            + (
                ""
                if sub_col is not None
                else " To select by a different column (e.g. coarse cell types) "
                "and plot the `color` categories inside them, pass `sub_col`."
            )
        )
    if len(found) < len(cell_list):
        warnings.warn(
            f"{[c for c in cell_list if c not in present]} not found in "
            f"adata.obs[{search_col!r}] and ignored.",
            stacklevel=2,
        )

    # Work on a copy -- adata.obs must not change.
    d = data[[grouping, replicate, color] + ([sub_col] if sub_col else [])].copy()
    d = d.dropna(subset=[grouping, replicate, color])
    if sub_col is None:
        cats = found
    else:
        cats = list(pd.unique(d.loc[d[sub_col].isin(found), color]))
    if norm:
        d = d[d[search_col].isin(found)]

    # Percent per group x replicate; unobserved categories are filled with 0
    counts = (
        d.groupby([grouping, replicate, color], observed=True)
        .size()
        .unstack(color, fill_value=0)
    )
    counts.columns = counts.columns.astype(object)
    perc = counts.div(counts.sum(axis=1), axis=0) * 100
    perc = perc.reindex(columns=cats, fill_value=0).reset_index()

    reps_per_group = perc.groupby(grouping, observed=True)[replicate].nunique()
    if (reps_per_group < 2).any():
        warnings.warn(
            f"Group(s) {list(reps_per_group[reps_per_group < 2].index)} have only one "
            f"replicate in {replicate!r} -- a box needs several. `replicate` should be "
            "e.g. the patient or sample column, with several per group.",
            stacklevel=2,
        )
    groups_per_rep = perc.groupby(replicate, observed=True)[grouping].nunique()
    if (groups_per_rep > 1).any():
        warnings.warn(
            f"Replicate(s) {list(groups_per_rep[groups_per_rep > 1].index)} occur in "
            f"more than one {grouping!r} group. Replicates should be nested in "
            "`grouping` (e.g. each patient belongs to one condition).",
            stacklevel=2,
        )

    df = perc.melt(id_vars=[grouping, replicate], var_name=color, value_name="percent")

    if col_order is None:
        in_df = set(df[grouping])
        if isinstance(data[grouping].dtype, pd.CategoricalDtype):
            col_order = [g for g in data[grouping].cat.categories if g in in_df]
        else:
            try:
                col_order = sorted(in_df)
            except TypeError:
                col_order = sorted(in_df, key=str)
    else:
        col_order = list(col_order)
        unknown = [g for g in col_order if g not in set(df[grouping])]
        if unknown:
            fail(
                f"`col_order` contains {unknown} which are not values of "
                f"adata.obs[{grouping!r}]. Example values: {sorted(set(df[grouping]), key=str)[:10]}."
            )
        df = df[df[grouping].isin(col_order)]

    if plot_order is not None:
        plot_order = list(plot_order)
        unknown = [c for c in plot_order if c not in cats]
        if unknown:
            fail(
                f"`plot_order` contains {unknown}, which are not among the plotted categories {cats}."
            )
        cats = plot_order
    df = df[df[color].isin(cats)].reset_index(drop=True)

    palette = _resolve_categorical_palette(_style, adata, color, cats, palette=palette)

    # one panel per category
    n_cols = max(1, min(int(n_columns), len(cats)))
    n_rows = math.ceil(len(cats) / n_cols)
    if fig_sizing is None:
        fig_sizing = (1.2 * len(col_order) * n_cols + 1.5, 3.6 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_sizing, squeeze=False)
    for ax, cat in zip(axes.flat, cats):
        sub = df[df[color] == cat]
        c = palette.get(cat, "grey")
        sns.boxplot(
            data=sub,
            x=grouping,
            y="percent",
            order=col_order,
            color=c,
            linecolor="black",
            showfliers=False,
            width=0.6,
            ax=ax,
        )
        for patch in ax.patches:
            patch.set_facecolor(mcolors.to_rgba(c, 0.4))
        sns.swarmplot(
            data=sub,
            x=grouping,
            y="percent",
            order=col_order,
            color=c,
            edgecolor="black",
            linewidth=1,
            size=point_size,
            ax=ax,
        )
        ax.set_title(str(cat), fontsize=_style.font_size_small)
        ax.set_xlabel("")
        ax.set_ylabel("percent", fontsize=_style.font_size_small)
        ax.tick_params(axis="x", labelrotation=90, labelsize=_style.font_size_tiny)
        # small padding below 0 and above the top so dots on the axis (value 0) are not cut in half
        top = max(float(sub["percent"].max()), 1e-9)
        ax.set_ylim(-0.05 * top, 1.06 * top)
        ax.spines["left"].set_bounds(
            0, ax.get_yticks()[ax.get_yticks() <= 1.06 * top].max()
        )
        _apply_spines(ax, _style)
    for ax in axes.flat[len(cats) :]:
        ax.set_visible(False)
    fig.tight_layout()

    if savefig:
        _savefig(fig, output_dir, output_fname, "_swarm_boxplot", _style)

    plt.show()
    return df, cats


# legacy name
pl_swarm_box_ad = swarm_box_plot


def _spread_y(ys, min_gap, limit=1.3):
    """Push label y-positions apart so neighbours are >= min_gap, keeping order, within +-limit if possible."""
    ys = np.asarray(ys, dtype=float)
    if len(ys) < 2:
        return ys
    order = np.argsort(ys)
    out = ys[order].copy()
    for k in range(1, len(out)):
        out[k] = max(out[k], out[k - 1] + min_gap)
    # keep the stack centred on its anchors instead of only growing upwards
    out += ys.mean() - out.mean()
    if out[-1] > limit:
        out[-1] = limit
        for k in range(len(out) - 2, -1, -1):
            out[k] = min(out[k], out[k + 1] - min_gap)
    res = np.empty_like(ys)
    res[order] = out
    return res


def create_pie_charts(
    adata,
    color,
    grouping,
    plot_order=None,
    show_percentages=True,
    palette=None,
    figsize=None,
    savefig=False,
    output_fname="",
    output_dir="./",
    style=None,
):
    """
    Create pie charts for each group based on a grouping column, showing the percentage of total rows based on a
    count column.

    Parameters
    ----------
    adata : pd.DataFrame
        The input DataFrame.
    color : str
        The column name used for counting occurrences.
    grouping : str
        The column name for grouping the data.
    plot_order : list, optional
        The order of groups for plotting. Defaults to None.
    show_percentages : bool, optional
        Whether to show the percentage numbers on the pie charts. Defaults to True.
    palette : dict, optional
        A dictionary to manually set colors for neighborhoods -- any category not covered is
        colored grey, with a warning. Defaults to None, which reuses/creates a persistent
        mapping cached in `adata.uns[color + "_colors"]`, consistent with any other function
        (e.g. `catplot`, `extended_dotplot`) already plotting `color`.
    figsize : tuple, optional
        Figure width, height in inches. Defaults to (7 * columns, 5 * rows).
    savefig : bool, optional
        Whether to save the figure or not. Defaults to False.
    output_fname : str, optional
        The output file name. Defaults to "".
    output_dir : str, optional
        The output directory. Defaults to "./".

    Returns
    -------
    None
    """
    _style = _apply_style(style)
    data = adata.obs

    # Group the data by the grouping column
    grouped_data = data.groupby(grouping)

    # Sort the groups based on the plot_order if provided
    if plot_order:
        grouped_data = sorted(grouped_data, key=lambda x: plot_order.index(x[0]))

    # Calculate the number of rows and columns for subplots
    num_groups = len(grouped_data)
    num_cols = min(3, num_groups)  # Number of columns for subplots
    num_rows = (num_groups - 1) // num_cols + 1

    # Create subplots
    # each pie needs ~7in so its outside labels do not run into the neighbouring pie
    if figsize is None:
        figsize = (7 * num_cols, 5 * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    # Create a color dictionary if not provided
    # Keep categories in `color`'s own dtype (e.g. int32 CN ids) rather than stringifying --
    # `percentages.index` below comes from value_counts() on the same raw dtype, so a
    # stringified categories list here would resolve `palette` to str keys that no longer
    # match it, silently falling back to "gray" for every slice (and defeating a user's
    # native int-keyed `palette` dict, which would never match the string keys either).
    categories = sorted(adata.obs[color].dropna().unique())
    palette = _resolve_categorical_palette(
        _style, adata, color, categories, palette=palette
    )

    # Iterate over each group and create a pie chart
    pies = []
    xlim = 3.8  # room for the labels on both sides of the pie
    for i, (group, group_df) in enumerate(grouped_data):
        # Count the occurrences of each neighborhood within the group
        neighborhood_counts = group_df[color].value_counts()

        # Calculate the percentage of total rows for each neighborhood
        percentages = neighborhood_counts / group_df.shape[0] * 100

        # Create a color list for neighborhoods in the count column
        colors = [
            palette.get(neighborhood, "gray") for neighborhood in percentages.index
        ]

        # Rotate so the run of small wedges (sorted last) is centred at 9 o'clock: their
        # labels then stack vertically beside them and the leader lines stay short
        small = percentages < 10
        startangle = 90
        if small.any():
            startangle = 180 + 3.6 * (
                percentages[~small].sum() + percentages[small].sum() / 2
            )

        ax = axes[i]
        wedges, _ = ax.pie(
            percentages,
            colors=colors,
            startangle=startangle,
            counterclock=False,
            wedgeprops=dict(edgecolor="white", linewidth=0.8),
        )
        ax.set_title(f"Group: {group}", fontsize=_style.font_size_medium)
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-1.4, 1.4)
        pies.append((ax, wedges, percentages))

    # Remove unused subplots
    for j in range(num_groups, num_rows * num_cols):
        fig.delaxes(axes[j])

    # Adjust spacing between subplots; labels are placed afterwards so their spacing can
    # be derived from the final axes size
    fig.tight_layout()
    fig_w, fig_h = fig.get_size_inches()
    for ax, wedges, percentages in pies:
        pos = ax.get_position()
        # inches per data unit (equal aspect -> the tighter of the two axes limits)
        inch_per_unit = min(pos.width * fig_w / (2 * xlim), pos.height * fig_h / 2.8)
        min_gap = 1.5 * _style.font_size_small / 72 / inch_per_unit

        # Labels in clean columns left/right of the pie, joined to the wedges by elbow
        # leader lines (radial tick out of the rim, then to the label); label order follows
        # wedge order on each side, so lines never cross
        rads = [np.deg2rad((w.theta1 + w.theta2) / 2) for w in wedges]
        xs, ys = np.cos(rads), np.sin(rads)
        side = np.where(xs >= 0, 1, -1)
        label_y = np.empty(len(ys))
        for sd in (1, -1):
            idx = np.where(side == sd)[0]
            label_y[idx] = _spread_y(1.15 * ys[idx], min_gap=min_gap, limit=1.3)

        for j, (name, pct) in enumerate(percentages.items()):
            text = f"{name} ({pct:.1f}%)" if show_percentages else name
            ax.plot(
                [xs[j], 1.08 * xs[j], 1.18 * side[j]],
                [ys[j], 1.08 * ys[j], label_y[j]],
                color="0.35",
                lw=0.6,
                solid_capstyle="butt",
                clip_on=False,
            )
            ax.text(
                1.22 * side[j],
                label_y[j],
                text,
                ha="left" if side[j] > 0 else "right",
                va="center",
                fontsize=_style.font_size_small,
            )
        ax.set_ylim(min(-1.4, label_y.min() - 0.15), max(1.4, label_y.max() + 0.15))

    if savefig:
        _savefig(fig, output_dir, output_fname, "_piechart", _style)

    plt.show()


def cn_exp_heatmap(
    adata,
    cluster_col,
    cn_col,
    palette=None,
    savefig=False,
    output_fname="",
    output_dir="./",
    row_clus=True,
    col_clus=True,
    figsize=(10, 5),
    grouping=None,
    show_cn_size=True,
    row_dendro_ratio=0.2,
    col_dendro_ratio=0.2,
    bar_width_in=0.8,
    dendrogram=True,
    style=None,
):
    """
    Create a heatmap of expression data, clustered by rows and columns.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cluster_col : str
        The column name for clustering the data.
    cn_col : str
        The column name for the color selection.
    palette : dict, optional
        A dictionary to manually set colors for neighborhoods -- any category not covered is
        colored grey, with a warning. Defaults to None, which reuses/creates a persistent
        mapping cached in `adata.uns[cn_col + "_colors"]`, consistent with any other function
        (e.g. `catplot`) already plotting `cn_col`.
    figsize : tuple, optional
        The size of the figure. Defaults to (18, 12).
    savefig : bool, optional
        Whether to save the figure or not. Defaults to False.
    output_fname : str, optional
        The output file name. Defaults to "".
    output_dir : str, optional
        The output directory. Defaults to "./".
    row_clus : bool, optional
        Whether to cluster the rows or not. Defaults to True.
    col_clus : bool, optional
        Whether to cluster the columns or not. Defaults to True.
    grouping : str, optional
        An `adata.obs` column assigning each `cluster_col` value to a major group (e.g. cell
        types into "Immune", "Stroma", ...). Columns are then arranged as one contiguous block
        per group -- clustered within each block if `col_clus`, otherwise sorted -- with a
        labeled color strip on top and white separators between blocks. The column
        dendrogram is not shown. Group colors are cached in `adata.uns[grouping + "_colors"]`.
        Each `cluster_col` value must belong to exactly one group. Defaults to None.
    show_cn_size : bool, optional
        Show a slim bar chart right of the heatmap with the number of cells in each
        neighborhood, aligned with its row. Defaults to True.
    row_dendro_ratio, col_dendro_ratio : float, optional
        Share of the figure width (row dendrogram) or height (column dendrogram) used by the
        dendrograms. Small values (e.g. 0.06) leave more room for the heatmap columns in
        compact figures. Defaults to 0.2.
    bar_width_in : float, optional
        Width in inches of the "n cells" bar panel. Defaults to 0.8.

    dendrogram : bool, optional
        Draw the dendrograms. False keeps the clustered row/column order but frees the space the
        dendrograms take. Defaults to True.

    Returns
    -------
    None
    """
    _style = _apply_style(style)
    data = adata.obs
    output_dir = pathlib.Path(output_dir)

    # Keep categories in cn_col's own dtype (e.g. int32 CN ids) rather than stringifying --
    # niche_sub/fc_2 below are indexed by the same raw dtype via groupby(cn_col), so a
    # stringified categories list here would resolve `palette` to str keys that no longer
    # match fc_2.index, silently reindexing to all-NaN colors (matplotlib then raises
    # "Invalid RGBA argument: nan"). It also previously defeated a user's native int-keyed
    # `palette` dict by comparing it against string keys.
    categories = sorted(adata.obs[cn_col].dropna().unique())
    palette = _resolve_categorical_palette(
        _style, adata, cn_col, categories, palette=palette
    )

    neigh_data = pd.DataFrame(
        {cn_col: list(palette.keys()), "color": list(palette.values())}
    )
    neigh_data.set_index(keys=cn_col, inplace=True)

    df3 = pd.concat([data, pd.get_dummies(data[cluster_col])], axis=1)
    sum_cols2 = df3[cluster_col].unique()
    values2 = df3[sum_cols2].values
    cell_list = sum_cols2.copy()
    cell_list = cell_list.tolist()
    cell_list.append(cn_col)

    subset = df3[cell_list]
    niche_sub = subset.groupby(cn_col).sum()
    niche_df = niche_sub.apply(lambda x: x / x.sum() * 10, axis=1)
    neigh_clusters = niche_df.to_numpy()

    tissue_avgs = values2.mean(axis=0)

    fc_2 = np.log2(
        (
            (neigh_clusters + tissue_avgs)
            / (neigh_clusters + tissue_avgs).sum(axis=1, keepdims=True)
        )
        / tissue_avgs
    )
    fc_2 = pd.DataFrame(fc_2, columns=sum_cols2)
    fc_2.set_index(niche_df.index, inplace=True)

    col_colors = None
    if grouping is not None:
        from scipy.cluster.hierarchy import leaves_list, linkage

        pairs = data[[cluster_col, grouping]].dropna().drop_duplicates().astype(str)
        group_of = pairs.set_index(cluster_col)[grouping]
        ambiguous = sorted(group_of.index[group_of.index.duplicated()].unique())
        if ambiguous:
            raise ValueError(
                f"`grouping` must assign each `{cluster_col}` value to exactly one group, "
                f"but these belong to several groups in `{grouping}`: {ambiguous}"
            )
        group_of = group_of.reindex([str(c) for c in fc_2.columns])
        unassigned = group_of.index[group_of.isna()].tolist()
        if unassigned:
            raise ValueError(
                f"These `{cluster_col}` values have no group in `{grouping}`: {unassigned}"
            )

        # one contiguous block per group, clustered within the block
        order = []
        for group in sorted(group_of.unique()):
            cols = fc_2.columns[(group_of == group).to_numpy()]
            cols = cols[np.argsort([str(c) for c in cols], kind="stable")]
            if col_clus and len(cols) > 2:
                cols = cols[leaves_list(linkage(fc_2[cols].T, "average"))]
            order.extend(cols)
        fc_2 = fc_2[order]
        col_groups = np.array([group_of[str(c)] for c in fc_2.columns])

        group_palette = {
            str(g): c
            for g, c in _resolve_categorical_palette(
                _style, adata, grouping, sorted(group_of.unique())
            ).items()
        }
        col_colors = pd.Series(
            [group_palette[g] for g in col_groups], index=fc_2.columns, name=grouping
        )

    def _draw(row_colors_ratio):
        return sns.clustermap(
            fc_2,
            vmin=-3,
            vmax=3,
            cmap=_style.diverging_cmap,
            figsize=figsize,
            xticklabels=True,  # label every column (seaborn "auto" skips labels when columns are narrow)
            row_colors=[neigh_data.reindex(fc_2.index)["color"]],
            col_colors=col_colors,
            colors_ratio=(row_colors_ratio, 0.03),
            col_cluster=grouping is None,
            # no column dendrogram with grouping, so don't reserve space for it
            dendrogram_ratio=(
                row_dendro_ratio if dendrogram else 0.004,
                (
                    (col_dendro_ratio if grouping is None else 0.02)
                    if dendrogram
                    else 0.004
                ),
            ),
            cbar_pos=(
                (1 - (bar_width_in + 0.35) / figsize[0], 0.02, 0.03, 0.08)
                if (show_cn_size and not dendrogram)
                else (0.03, 0.02, 0.03, 0.08)
            ),
        )

    # widen the CN color strip so the longest CN name fits inside its box (+ padding).
    # tight_layout shrinks the gridspec by an amount only known after drawing, so measure
    # the strip actually drawn and redraw once with a corrected ratio if it came out short.
    needed_in = 0.15 + max(
        _measure_text_width_inches(str(cn), _style.font_size_annotation, dpi=100)
        for cn in fc_2.index
    )
    row_colors_ratio = max(0.03, needed_in / figsize[0])
    s = _draw(row_colors_ratio)
    strip_in = s.ax_row_colors.get_position().width * figsize[0]
    if strip_in < needed_in:
        plt.close(s.figure)
        s = _draw(row_colors_ratio * needed_in / strip_in)

    s.ax_row_dendrogram.set_visible(row_clus and dendrogram)
    s.ax_col_dendrogram.set_visible(col_clus and grouping is None and dendrogram)

    if grouping is not None:
        s.ax_col_colors.set_yticks([])
        starts = np.flatnonzero(np.r_[True, col_groups[1:] != col_groups[:-1]])
        ends = np.r_[starts[1:], len(col_groups)]
        for start, end in zip(starts, ends):
            group = col_groups[start]
            # ponytail: name can overflow a 1-column block; rotate/abbreviate if that bites
            s.ax_col_colors.text(
                (start + end) / 2,
                0.5,
                group,
                ha="center",
                va="center",
                fontsize=_style.font_size_annotation,
                color=(
                    "black"
                    if _is_light_color(group_palette[group], threshold=0.5)
                    else "white"
                ),
            )
            if start:
                for ax in (s.ax_heatmap, s.ax_col_colors):
                    ax.axvline(start, color="white", linewidth=2)
    s.ax_heatmap.set_ylabel("")
    s.ax_heatmap.set_yticks([])

    # write each CN id inside its color box instead of as a tick label on the far side
    row_order = (
        fc_2.index[s.dendrogram_row.reordered_ind]
        if s.dendrogram_row is not None
        else fc_2.index
    )
    s.ax_row_colors.set_xticks([])
    for i, cn in enumerate(row_order):
        color = neigh_data.loc[cn, "color"]
        s.ax_row_colors.text(
            0.5,
            i + 0.5,
            str(cn),
            ha="center",
            va="center",
            fontsize=_style.font_size_annotation,
            color="black" if _is_light_color(color, threshold=0.5) else "white",
        )

    if show_cn_size:
        # slim "n cells" bar panel right of the heatmap, sized in inches so it stays slim
        fig_w = figsize[0]
        gap, bar_w, pad = 0.1 / fig_w, bar_width_in / fig_w, 0.1 / fig_w
        for ax in (s.ax_heatmap, s.ax_col_dendrogram, s.ax_col_colors):
            if ax is not None:
                p = ax.get_position()
                ax.set_position([p.x0, p.y0, p.width - (gap + bar_w + pad), p.height])
        hm = s.ax_heatmap.get_position()
        ax_size = s.figure.add_axes(
            [hm.x1 + gap, hm.y0, bar_w, hm.height], sharey=s.ax_heatmap
        )
        counts = adata.obs[cn_col].value_counts().reindex(row_order).fillna(0)
        ax_size.barh(
            np.arange(len(row_order)) + 0.5,
            counts.values,
            height=0.6,
            color=_style.accent_color_neutral,
        )
        ax_size.set_xlabel("n cells", fontsize=_style.font_size_small)
        ax_size.tick_params(axis="both", labelsize=_style.font_size_tiny)
        ax_size.tick_params(axis="y", left=False, labelleft=False)
        _apply_spines(ax_size, _style)
        if not dendrogram:
            # axis on top of the bar panel: leaves the bottom-right corner free for the colorbar
            ax_size.xaxis.tick_top()
            ax_size.xaxis.set_label_position("top")
            ax_size.spines["bottom"].set_visible(False)
            ax_size.spines["top"].set_visible(True)

    # anchor label to the colorbar so it follows it at any figsize
    s.ax_cbar.set_title(
        "log2 fold change\nover tissue average",
        fontsize=_style.font_size_annotation,
        loc="left",
        pad=6,
    )

    if savefig:
        _savefig(
            s.figure,
            output_dir,
            output_fname,
            "",
            _style,
            dpi=_style.figure_dpi_lineart,
        )

    plt.show()


def pl_area_nuc_cutoff(
    df,
    cutoff_area,
    cutoff_nuc,
    cellsize_column="area",
    nuc_marker_column="Hoechst1",
    color_by="unique_label",
    palette="Paired",
    alpha=0.8,
    size=0.4,
    log_scale=True,
):
    """
    Plot a joint scatter plot to determine area and nuclear-intensity cutoffs for cell segmentation QC.

    Displays a seaborn ``jointplot`` with cell area on the y-axis and a
    nuclear marker intensity on the x-axis, coloured by a categorical label.
    Dashed lines indicate the specified area and nuclear-intensity cutoffs.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame containing cell-level measurements.
    cutoff_area : float
        Horizontal dashed line value for the cell-area cutoff.
    cutoff_nuc : float
        Vertical dashed line value for the nuclear-intensity cutoff.
    cellsize_column : str, optional
        Column name for cell area. Default is ``"area"``.
    nuc_marker_column : str, optional
        Column name for nuclear marker intensity. Default is
        ``"Hoechst1"``.
    color_by : str, optional
        Column name used to colour the data points. Default is
        ``"unique_label"``.
    palette : str or dict, optional
        Colour palette passed to seaborn. Default is ``"Paired"``.
    alpha : float, optional
        Transparency of the scatter points. Default is 0.8.
    size : float, optional
        Size of scatter points. Default is 0.4.
    log_scale : bool, optional
        If True, both axes are shown on a log scale. Default is True.

    Returns
    -------
    None
    """
    # Custom the inside plot: options are: "scatter" | "reg" | "resid" | "kde" | "hex"
    g = sns.jointplot(
        x=df[nuc_marker_column],
        y=df[cellsize_column],
        hue=df[color_by],
        palette=palette,
        alpha=alpha,
    )

    if log_scale == True:
        # log scale joint plot
        g.ax_joint.set_xscale("log")
        g.ax_joint.set_yscale("log")

    # add horizontal and vertical lines
    g.ax_joint.axhline(cutoff_area, color="k", linestyle="dashed", linewidth=1)
    g.ax_joint.axvline(cutoff_nuc, color="k", linestyle="dashed", linewidth=1)

    # place legend outside
    g.ax_joint.legend(bbox_to_anchor=(1.2, 1), loc="upper left", borderaxespad=0)

    # show plot
    plt.show()


def pl_plot_scatter_correlation(
    data, x, y, xlabel=None, ylabel=None, save_path=None, style=None
):
    """
    Plot a scatter plot with a linear regression line and Pearson correlation annotation.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data frame containing the two variables to correlate.
    x : str
        Column name for the x-axis variable.
    y : str
        Column name for the y-axis variable.
    xlabel : str, optional
        Custom label for the x-axis. When None the column name is used.
        Default is None.
    ylabel : str, optional
        Custom label for the y-axis. When None the column name is used.
        Default is None.
    save_path : str, optional
        Full file path for saving the plot. When None the plot is only
        displayed. Default is None.
    style : SPACEcStyle or None, optional
        SPACEc style configuration object. Default is None.

    Returns
    -------
    None
    """
    _style = _apply_style(style)
    g = sns.lmplot(x=x, y=y, data=data, height=5, aspect=1)
    g.map_dataframe(hf_annotate_cor_plot, x=x, y=y, data=data)

    if xlabel:
        plt.xlabel(xlabel, fontsize=_style.font_size_small)
    if ylabel:
        plt.ylabel(ylabel, fontsize=_style.font_size_small)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if save_path:
        _savefig(
            None,
            pathlib.Path(save_path).parent,
            pathlib.Path(save_path).name,
            "_corrplot",
            _style,
        )
    plt.show()


def pl_plot_scatter_correlation_ad(
    adata, x, y, xlabel=None, ylabel=None, save_path=None, style=None
):
    """
    Plot a scatter plot with a linear regression line and Pearson correlation annotation from an AnnData object.

    Extracts cell metadata from ``adata.obs`` and passes it to a seaborn
    ``lmplot`` together with a Pearson-correlation annotation.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object whose ``.obs`` attribute provides the data.
    x : str
        Column name in ``adata.obs`` for the x-axis variable.
    y : str
        Column name in ``adata.obs`` for the y-axis variable.
    xlabel : str, optional
        Custom label for the x-axis. When None the column name is used.
        Default is None.
    ylabel : str, optional
        Custom label for the y-axis. When None the column name is used.
        Default is None.
    save_path : str, optional
        Full file path for saving the plot. When None the plot is only
        displayed. Default is None.
    style : SPACEcStyle or None, optional
        SPACEc style configuration object. Default is None.

    Returns
    -------
    None
    """
    _style = _apply_style(style)
    data = adata.obs

    g = sns.lmplot(x=x, y=y, data=data, height=5, aspect=1)
    g.map_dataframe(hf_annotate_cor_plot, x=x, y=y, data=data)

    if xlabel:
        plt.xlabel(xlabel, fontsize=_style.font_size_small)
    if ylabel:
        plt.ylabel(ylabel, fontsize=_style.font_size_small)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if save_path:
        _savefig(
            None,
            pathlib.Path(save_path).parent,
            pathlib.Path(save_path).name,
            "_corrplot",
            _style,
        )
    plt.show()


########


def correlation_heatmap(
    cmat,
    figsize=(10, 10),
    annot=True,
    cbar_orientation="vertical",
    savefig=False,
    output_fname="",
    output_dir="./",
    return_data=False,
    style=None,
):
    """
    Plot the lower-triangle of a correlation matrix as an annotated heatmap.

    A general-purpose correlation figure — e.g. marker-marker correlation
    (``adata.to_df().corr()``) or cell-type-mean correlation
    (``adata.to_df().groupby(adata.obs["cell_type"]).mean().T.corr()``).

    Parameters
    ----------
    cmat : pandas.DataFrame
        Square correlation matrix with variable names as both index and
        columns.
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 10)``.
    annot : bool, optional
        Whether to annotate each cell with its correlation value. Defaults
        to True.
    cbar_orientation : {"vertical", "inside"}, optional
        Colorbar position. "inside" puts a horizontal colorbar into the empty upper triangle
        and saves space in narrow panels. Defaults to "vertical".
    savefig : bool, optional
        Whether to save the figure. Defaults to False.
    output_fname : str, optional
        Base filename (without extension) used when ``savefig=True``.
    output_dir : str, optional
        Directory to save into when ``savefig=True``. Defaults to ``"./"``.
    return_data : bool, optional
        If True, also return ``cmat`` (source data for publication).
        Defaults to False.
    style : SPACEcStyle, str, or pathlib.Path, optional
        Style to apply for this call. Defaults to the active style.

    Returns
    -------
    matplotlib.axes.Axes or (matplotlib.axes.Axes, pandas.DataFrame)
        The heatmap axes, and — if ``return_data=True`` — ``cmat`` itself.
    """
    _style = _apply_style(style)
    mask = np.triu(np.ones_like(cmat, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cmat,
        annot=annot,
        fmt=".2f",
        cmap=_style.correlation_cmap,
        square=True,
        mask=mask,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar=cbar_orientation != "inside",
        cbar_kws={"label": "correlation", "orientation": "vertical"},
    )
    if cbar_orientation == "inside":
        # colorbar in the empty upper triangle: no extra space needed
        cax = ax.inset_axes([0.52, 0.80, 0.42, 0.035])
        cb = fig.colorbar(ax.collections[0], cax=cax, orientation="horizontal")
        cb.set_ticks([-1, 0, 1])
        cb.ax.tick_params(labelsize=_style.font_size_tiny, length=2)
        cb.set_label("correlation", fontsize=_style.font_size_tiny)
    _apply_spines(ax, _style)

    if savefig:
        _savefig(fig, output_dir, output_fname, "_correlation_heatmap", _style)

    if return_data:
        return ax, cmat
    return ax


def dumbbell(
    data,
    figsize=(10, 10),
    colors=None,
    savefig=False,
    output_fname="",
    output_dir="./",
    style=None,
):
    """
    Create a dumbbell plot.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame. The DataFrame should have two columns representing two different conditions.
    figsize : tuple, optional
        The size of the figure. Defaults to (10, 10).
    colors : list, optional
        The colors to use for the two conditions. Defaults to ["#DB444B", "#006BA2"].

    Returns
    -------
    None
    """
    _style = _apply_style(style)
    if colors is None:
        colors = [_style.accent_color_negative, _style.accent_color_positive]

    fig, ax = plt.subplots(figsize=figsize, facecolor=_style.background_color)
    ax.grid(which="major", axis="both", color="#758D99", alpha=0.6, zorder=1)

    ax.spines[["top", "right", "bottom"]].set_visible(False)

    # Plot data
    comp_cat = data.columns

    # Plot horizontal lines first
    ax.hlines(
        y=data.index,
        xmin=data[comp_cat[0]],
        xmax=data[comp_cat[1]],
        color="#758D99",
        zorder=2,
        linewidth=2,
        label="_nolegend_",
        alpha=0.8,
    )
    # Plot bubbles next
    sc0 = ax.scatter(
        data[comp_cat[0]], data.index, label="2014", s=60, color=colors[0], zorder=3
    )
    sc1 = ax.scatter(
        data[comp_cat[1]], data.index, label="2018", s=60, color=colors[1], zorder=3
    )
    # room around the data so the outer dots and the dashed zero line are not cut off
    ax.margins(x=0.06, y=0.012)

    # Set xlim
    # ax.set_xlim(-3, 3)

    # Reformat x-axis tick labels

    ax.xaxis.set_tick_params(
        labeltop=True,
        labelbottom=False,
        bottom=False,
        labelsize=_style.font_size_tiny,
        pad=-1,
    )

    ax.axvline(x=0, color="k", linestyle="--")

    # Set Legend
    ax.legend(
        data.columns,
        loc=(0, 1.11),
        ncol=2,
        frameon=False,
        handletextpad=0.4,
        handleheight=1,
    )

    # dot diameter follows the row pitch (dots never touch their neighbours), at most the default size
    fig.canvas.draw()
    pitch_pt = ax.get_window_extent().height * 72 / fig.dpi / max(len(data), 1)
    size = min(60.0, (0.72 * pitch_pt) ** 2)
    sc0.set_sizes([size])
    sc1.set_sizes([size])

    if savefig:
        _savefig(fig, output_dir, output_fname, "", _style)

    plt.show()


def plot_top_n_distances(
    dist_table_filt,
    dist_data_filt,
    n=5,
    palette=None,
    dodge=0.3,
    savefig=False,
    output_fname="",
    output_dir="./",
    figsize=(5, 5),
    unit="px",
    errorbars=True,
    condition_col="condition",
    style=None,
    colors=None,
    show_expected=True,
):
    """
    Plot observed distances for the N pairs whose spatial enrichment changes most between conditions.

    Pairs are ranked by ``|log2FC_B - log2FC_A|``, where ``log2FC`` is
    ``log2(observed_mean / expected_mean)`` from ``sp.tl.identify_interactions`` and A/B
    are the two columns of ``dist_table_filt``. The ranking is therefore a change in
    distance *relative to the permutation expectation*, not a change in raw distance.
    The y-axis shows raw observed distances; with ``show_expected=True`` the expected
    (permutation) mean for each pair is drawn as a dashed line in the same colour, so
    the gap that drives the ranking is visible.

    Parameters
    ----------
    dist_table_filt : pandas.DataFrame
        First output of ``sp.tl.filter_interactions``: one row per pair (index
        ``"pairs"``), one ``log2(observed/expected)`` column per condition.
    dist_data_filt : pandas.DataFrame
        Second output of ``sp.tl.filter_interactions``. Must include columns
        ``"pairs"``, ``"interaction"``, ``"observed"`` (list of per-cell distances) and
        ``condition_col``; ``"expected_mean"`` is needed for ``show_expected``.
    n : int, optional
        Number of top pairs to display. Default is 5.
    palette : list, dict or str, optional
        Colours for the pairs: a list (in rank order), a ``{interaction: colour}`` dict,
        or a matplotlib colormap name. Default None uses the SPACEc style palette.
    dodge : bool or float, optional
        Horizontal offset between pairs at each condition, so points and error bars
        don't overlap. Default is 0.3.
    savefig : bool, optional
        If True, save the figure to ``output_dir``. Default is False.
    output_fname : str, optional
        Base filename for the saved figure. Default is ``""``.
    output_dir : str, optional
        Directory for saving the figure. Default is ``"./"``.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (5, 5).
    unit : str, optional
        Distance unit label shown on the y-axis. Default is ``"px"``.
    errorbars : bool, optional
        If True, draw standard-error bars. The SE is computed over per-cell mean
        distances pooled across all regions, so it treats cells as independent and
        understates between-replicate variability. Default is True.
    condition_col : str, optional
        Column in ``dist_data_filt`` holding the condition labels plotted on the
        x-axis -- the ``comparison`` column passed to ``identify_interactions`` /
        ``filter_interactions``. Default is ``"condition"``.
    style : SPACEcStyle or None, optional
        SPACEc style configuration object. Default is None.
    colors : list or dict, optional
        Deprecated alias for `palette`, kept only for backwards compatibility -- raises a
        `DeprecationWarning` and overrides `palette` if given. Will be removed in a future
        version.
    show_expected : bool, optional
        If True, overlay the expected (permutation) mean distance for each pair as a
        dashed line. Default is True.

    Returns
    -------
    pandas.DataFrame
        The top ``n`` rows of ``dist_table_filt`` in rank order, with added columns
        ``"interaction"`` and ``"delta_log2fc"`` (second column minus first).
    """
    if colors is not None:
        warnings.warn(
            "'colors' is deprecated and will be removed in a future version. "
            "Use 'palette' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        palette = colors

    if dist_table_filt.empty:
        raise ValueError(
            "dist_table_filt is empty -- filter_interactions found no pair significant "
            "in both conditions. Try a looser `pvalue` or `logfold_group_abs`."
        )

    _style = _apply_style(style)
    cond_a, cond_b = dist_table_filt.columns[:2]

    # work on a copy so the caller's table is left untouched
    top = dist_table_filt.assign(
        delta_log2fc=dist_table_filt[cond_b] - dist_table_filt[cond_a]
    )
    top = top.loc[top["delta_log2fc"].abs().nlargest(n).index]

    data = dist_data_filt[dist_data_filt["pairs"].isin(top.index)]
    if data.empty:
        raise ValueError(
            "None of the top pairs in dist_table_filt were found in "
            "dist_data_filt['pairs'] -- were both made by the same filter_interactions call?"
        )
    pair_to_interaction = data.drop_duplicates("pairs").set_index("pairs")[
        "interaction"
    ]
    top["interaction"] = pair_to_interaction.reindex(top.index)
    hue_order = top["interaction"].tolist()
    order = [c for c in dist_table_filt.columns if c in set(data[condition_col])]

    if isinstance(palette, (list, tuple)):
        palette = dict(zip(hue_order, palette))
    palette = _resolve_categorical_palette(
        _style, None, None, hue_order, palette=palette
    )

    exploded = data.explode("observed")
    exploded["observed"] = exploded["observed"].astype(float)

    g = sns.catplot(
        data=exploded,
        x=condition_col,
        y="observed",
        hue="interaction",
        order=order,
        hue_order=hue_order,
        palette=palette,
        errorbar="se" if errorbars else None,
        capsize=0.1 if errorbars else 0,
        kind="point",
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        dodge=dodge,
        legend=False,
    )
    ax = g.ax

    handles = [
        plt.Line2D([0], [0], color=palette[i], marker="o", label=f"{i} ({d:+.2f})")
        for i, d in zip(hue_order, top["delta_log2fc"])
    ]

    if show_expected:
        if "expected_mean" in data.columns:
            sns.pointplot(
                data=data,
                x=condition_col,
                y="expected_mean",
                hue="interaction",
                order=order,
                hue_order=hue_order,
                palette=palette,
                errorbar=None,
                dodge=dodge,
                markers="",
                linestyles="--",
                legend=False,
                ax=ax,
            )
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="grey",
                    linestyle="--",
                    label="expected (permutation)",
                )
            )
        else:
            warnings.warn(
                "dist_data_filt has no 'expected_mean' column; skipping the expected overlay.",
                UserWarning,
                stacklevel=2,
            )

    ax.legend(
        handles=handles,
        title=f"Δ log2(obs/exp), {cond_b} − {cond_a}",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    _style_legend(ax, _style)
    ax.set_xlabel(condition_col)
    ax.set_ylabel(f"Observed distance ({unit})")
    ax.set_title(f"Top {len(top)} pairs by |Δ log2(obs/exp)|")
    g.despine(left=True)

    if savefig:
        _savefig(g, output_dir, output_fname, "", _style)

    plt.show()
    return top


def _cnmap_palette(_style, adata, cnmap_dict, cn_col, palette, rand_seed):
    """Resolve the CN palette for cn_map/cn_upset; CNs it misses get random colors."""
    categories = sorted(adata.obs[cn_col].dropna().unique().tolist())
    palette = _resolve_categorical_palette(
        _style, adata, cn_col, categories, palette=palette
    )
    missing = [n for n in cnmap_dict["l"] if n not in palette]
    if missing:
        extra = hf_generate_random_colors(len(missing), rand_seed=rand_seed)
        palette = {**palette, **dict(zip(missing, extra))}
    return palette


def cn_map(
    adata,
    cnmap_dict,
    cn_col,
    palette=None,
    figsize=(40, 20),
    savefig=False,
    output_fname="",
    output_dir="./",
    rand_seed=1,
    style=None,
    min_freq=None,
    show_cn_legend=True,
):
    """
    Generates a CNMap plot using the provided data and parameters.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    cnmap_dict : dict
        Dictionary containing graph, tops, e0, e1, and simp_freqs.
    cn_col : str
        Column name in adata to be used for color coding.
    palette : dict, optional
        Color palette to use for the plot -- any category not covered is colored grey, with
        a warning. If None, reuses/creates a persistent mapping cached in
        `adata.uns[cn_col + "_colors"]`, consistent with any other function (e.g. `catplot`)
        already plotting `cn_col`.
    figsize : tuple, optional
        Size of the figure. Defaults to (40, 20).
    savefig : bool, optional
        Whether to save the figure or not. Defaults to False.
    output_fname : str, optional
        The filename for the saved figure. Required if savefig is True. Defaults to "".
    output_dir : str, optional
        The directory where the figure will be saved. Defaults to "./".
    rand_seed : int, optional
        Seed for the graph layout (`_graphviz_layout`). Defaults to 1.
    min_freq : float, optional
        Only draw neighborhood combinations that make up at least this fraction of
        windows (e.g. 0.005). Defaults to None, which draws every combination
        kept by build_cn_map.

    show_cn_legend : bool, optional
        Draw the legend of the neighborhood colors below the map. Set False when the colors
        are explained elsewhere in the figure. Defaults to True.

    Returns
    -------
    None

    See Also
    --------
    cn_upset : The same combinations as ranked bars with a CN membership matrix.
    """
    _style = _apply_style(style)
    draw = cnmap_dict["g"]
    tops = cnmap_dict["tops"]
    simp_freqs = cnmap_dict["simp_freqs"]
    # node tuples are indices into the CN list used by build_cn_map -- colour through
    # that list, never through the key order of the palette passed here
    names = list(cnmap_dict["l"])
    freq_lookup = dict(zip(simp_freqs.index, simp_freqs.values))
    if min_freq is not None:
        draw = draw.subgraph([n for n in draw.nodes() if freq_lookup[n] >= min_freq])
    pos = _graphviz_layout(draw, prog="dot", seed=rand_seed)

    palette = _cnmap_palette(_style, adata, cnmap_dict, cn_col, palette, rand_seed)

    nodes = sorted(draw.nodes())
    freq = {n: float(freq_lookup[n]) for n in nodes}
    xy = np.array([pos[n] for n in nodes], dtype=float)

    # square size from the tightest spacing between nodes of the same rank
    gaps = np.concatenate(
        [np.diff(np.sort(xy[xy[:, 1] == y, 0])) for y in np.unique(xy[:, 1])]
    )
    gaps = gaps[gaps > 0]
    side = 0.6 * gaps.min() if len(gaps) else 20.0
    step = 1.15 * side
    r_max = 0.5 * side

    # stretch ranks vertically so each stack of squares plus its circle fits
    rank_ys = np.unique(xy[:, 1])
    needed = max(len(n) for n in nodes) * step + 2 * r_max + side
    y_scale = max(1.0, needed / np.diff(rank_ys).min()) if len(rank_ys) > 1 else 1.0
    pos = {n: (pos[n][0], pos[n][1] * y_scale) for n in nodes}

    fig, ax = plt.subplots(figsize=figsize)

    for e0, e1 in draw.edges():
        ax.plot(
            [pos[e0][0], pos[e1][0]],
            [pos[e0][1], pos[e1][1]],
            color="black",
            linewidth=3 if len(draw.in_edges(e1)) < len(e1) else 2,
            alpha=0.3,
            zorder=1,
        )

    max_freq = max(freq.values())
    top_xy = []
    for n in nodes:
        x, y = pos[n]
        # circle area is proportional to the frequency of the combination
        r = r_max * np.sqrt(freq[n] / max_freq)
        ax.add_patch(mpatches.Circle((x, y - r_max), r, color="black", zorder=2))
        if n in tops:
            # asterisk just right of the circle, so it never covers circle or squares
            top_xy.append((x + r + 0.2 * side, y - r_max))
        for k, i in enumerate(n):
            ax.add_patch(
                mpatches.Rectangle(
                    (x - side / 2, y + k * step),
                    side,
                    side,
                    facecolor=palette[names[i]],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
            )

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.02)
    ax.axis("off")

    # frequency legend: marker diameters in points matching the drawn circles
    ax.apply_aspect()
    pts_per_unit = (
        (ax.transData.transform((1, 0)) - ax.transData.transform((0, 0)))[0]
        * 72
        / fig.dpi
    )
    freq_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=2 * r_max * np.sqrt(f) * pts_per_unit,
            label=f"{100 * f * max_freq:.1f}%",
        )
        for f in (0.25, 0.5, 1.0)
    ]
    if top_xy:
        # six-armed asterisk marker (centered, unlike a text "*"), 0.3 squares wide
        star = dict(
            marker=(6, 2, 0),
            markersize=0.3 * side * pts_per_unit,
            color="black",
            markeredgewidth=1,
            linestyle="none",
        )
        tx, ty = zip(*top_xy)
        # no rescaling: the axes limits are fixed above and pts_per_unit depends on them
        ax.plot(tx, ty, zorder=4, scalex=False, scaley=False, **star)
        freq_handles.append(
            plt.Line2D([0], [0], label=f"Top {len(tops)} by frequency", **star)
        )
    # figure-level legend: an axes artist would be clipped to the axes area
    fig.legend(
        handles=freq_handles,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=ax.transAxes,
        title="Total frequency",
        fontsize=_style.font_size_small,
        title_fontsize=_style.font_size_small,
        frameon=_style.legend_frameon,
        # row pitch and handle box must exceed the largest circle, or big circles overlap
        # their neighbours and their own labels
        labelspacing=max(
            1.5,
            max(h.get_markersize() for h in freq_handles[:3]) / _style.font_size_small,
        ),
        handlelength=max(
            2.0,
            max(h.get_markersize() for h in freq_handles[:3]) / _style.font_size_small
            + 0.5,
        ),
        handletextpad=0.6,
    )

    # CN legend: only neighborhoods that occur in the drawn combinations
    used = sorted({i for n in nodes for i in n})
    if show_cn_legend:
        ax.legend(
            handles=[
                mpatches.Patch(color=palette[names[i]], label=names[i]) for i in used
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=3,
            fontsize=_style.font_size_small,
            frameon=_style.legend_frameon,
        )

    if savefig:
        _savefig(None, output_dir, output_fname, "_CNMap", _style)

    plt.show()


def cn_upset(
    adata,
    cnmap_dict,
    cn_col,
    palette=None,
    figsize=None,
    savefig=False,
    output_fname="",
    output_dir="./",
    rand_seed=1,
    style=None,
    min_freq=None,
    show_values="auto",
    value_decimals=1,
    min_plot_in=(2.0, 1.2),
    row_pitch=1.8,
):
    """
    Plot neighborhood combinations from build_cn_map as an UpSet-style chart.

    Alternative to `cn_map` with the same arguments: one column per combination,
    ranked by frequency, with bars for the fraction of windows (top) and the member
    CNs drawn as cn_map's colored squares (bottom), labeled by CN boxes as in
    cn_exp_heatmap. Bars on the left give, per CN, the fraction of windows whose
    combination contains it. Unlike `cn_map`, combinations
    without a parent or child combination in the graph are shown too.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    cnmap_dict : dict
        Output of build_cn_map; uses "simp_freqs" and "l".
    cn_col : str
        Column name in adata to be used for color coding.
    palette : dict, optional
        Color palette for the CNs, resolved as in `cn_map`. Defaults to None.
    figsize : tuple, optional
        Size of the figure. Defaults to None, which uses the style's
        (figure_width_double, figure_height_default). The figure grows if it is too
        small for the CN rows and name labels at the style's font sizes.
    savefig : bool, optional
        Whether to save the figure or not. Defaults to False.
    output_fname : str, optional
        The filename for the saved figure. Required if savefig is True. Defaults to "".
    output_dir : str, optional
        The directory where the figure will be saved. Defaults to "./".
    rand_seed : int, optional
        Seed for colors of CNs missing from the palette. Defaults to 1.
    min_freq : float, optional
        Only show combinations that make up at least this fraction of windows
        (e.g. 0.005). Defaults to None, which uses 0.001 -- the cutoff build_cn_map
        uses for the cn_map graph.
    show_values : bool or "auto", optional
        Print each combination's percentage of windows on top of its bar. "auto"
        (default) prints them only if they fit the column width (turned vertical
        if needed); True always prints them, shrinking the font if needed; False
        never does.
    value_decimals : int, optional
        Decimals of the printed percentages. Defaults to 1.

    min_plot_in : tuple, optional
        Minimum (width, height) in inches of the bar chart above the matrix; lowering it allows
        compact figures. Defaults to (2.0, 1.2).
    row_pitch : float, optional
        Row height of the matrix in multiples of the annotation font size. Defaults to 1.8.

    Returns
    -------
    None

    See Also
    --------
    cn_map : The same combinations as a hierarchy graph.
    """
    _style = _apply_style(style)
    names = list(cnmap_dict["l"])
    simp_freqs = cnmap_dict["simp_freqs"]
    cutoff = 0.001 if min_freq is None else min_freq
    kept = simp_freqs[simp_freqs >= cutoff].sort_values(ascending=False)
    if kept.empty:
        raise ValueError(
            f"No neighborhood combination makes up at least {cutoff} of windows; "
            "lower min_freq."
        )
    combos = list(kept.index)
    pct = 100 * kept.values
    # node tuples are indices into cnmap_dict["l"], as in cn_map
    rows = sorted({i for c in combos for i in c})
    row_of = {i: r for r, i in enumerate(rows)}
    palette = _cnmap_palette(_style, adata, cnmap_dict, cn_col, palette, rand_seed)

    if figsize is None:
        figsize = (_style.figure_width_double, _style.figure_height_default)
    fs_label, fs_box, fs_pct = (
        _style.font_size_medium,
        _style.font_size_annotation,
        _style.font_size_tiny,
    )
    n_rows, n_cols = len(rows), len(combos)
    row_names = [str(names[i]) for i in rows]
    cn_xlabel = "windows with CN (%)"

    # size the label strip, the left column and the CN rows to their text (inches), as
    # cn_exp_heatmap does for its CN strip; the figure grows if figsize can't fit them
    lab_in = 0.15 + max(
        _measure_text_width_inches(n, fs_box, dpi=100) for n in row_names
    )
    cn_in = max(1.0, 0.1 + _measure_text_width_inches(cn_xlabel, fs_label, dpi=100))
    matrix_in = n_rows * row_pitch * fs_box / 72
    fig_w = max(figsize[0], cn_in + lab_in + min_plot_in[0])
    fig_h = max(figsize[1], matrix_in + 0.8 + min_plot_in[1])

    fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, wspace=0.005, hspace=0.005)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[cn_in, lab_in, fig_w - cn_in - lab_in],
        height_ratios=[max(min_plot_in[1], fig_h - matrix_in - 0.8), matrix_in],
    )
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_dot = fig.add_subplot(gs[1, 2], sharex=ax_bar)
    ax_cn = fig.add_subplot(gs[1, 0], sharey=ax_dot)
    ax_lab = fig.add_subplot(gs[1, 1], sharey=ax_dot)
    x = np.arange(n_cols)

    ax_bar.bar(
        x, pct, width=0.6, color=_style.edge_color_default, alpha=_style.data_alpha_bar
    )
    ax_bar.set_ylabel("windows (%)", fontsize=fs_label)
    # y axis on the right: on the left its ticks and label widen the whole matrix column,
    # leaving a gap between the CN label boxes and the squares
    ax_bar.yaxis.tick_right()
    ax_bar.yaxis.set_label_position("right")
    ax_bar.set_title(
        f"{n_cols} combinations \u2265 {cutoff:.1%} of windows",
        fontsize=_style.font_size_large,
        pad=10,
    )
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.spines[["top", "left"]].set_visible(False)

    # CN names inside their color boxes, as in cn_exp_heatmap
    for r, (i, name) in enumerate(zip(rows, row_names)):
        color = palette[names[i]]
        ax_lab.add_patch(
            mpatches.Rectangle((0, r - 0.4), 1, 0.8, facecolor=color, linewidth=0)
        )
        ax_lab.text(
            0.5,
            r,
            name,
            ha="center",
            va="center",
            fontsize=fs_box,
            color="black" if _is_light_color(color, threshold=0.5) else "white",
        )
    ax_lab.set_xlim(0, 1)
    ax_dot.set_ylim(n_rows - 0.5, -0.5)
    ax_dot.set_xlim(-0.6, n_cols - 0.4)
    ax_dot.axis("off")
    ax_lab.axis("off")

    totals = [100 * kept.values[[i in c for c in combos]].sum() for i in rows]
    ax_cn.barh(
        range(n_rows), totals, height=0.6, color=[palette[names[i]] for i in rows]
    )
    ax_cn.invert_xaxis()
    ax_cn.set_xlabel(cn_xlabel, fontsize=fs_label)
    ax_cn.tick_params(axis="y", left=False, labelleft=False)
    ax_cn.spines[["top", "left"]].set_visible(False)

    # constrained layout takes its margins out of every column and row, so the strip and
    # the matrix can come out short of what their text needs: measure and correct once
    fig.draw_without_rendering()
    lab_w = ax_lab.get_window_extent().width / fig.dpi
    dot_h = ax_dot.get_window_extent().height / fig.dpi
    if lab_w < lab_in or dot_h < matrix_in:
        wr, hr = list(gs.get_width_ratios()), list(gs.get_height_ratios())
        wr[1] *= max(1.0, lab_in / lab_w)
        hr[1] *= max(1.0, matrix_in / dot_h)
        gs.set_width_ratios(wr)
        gs.set_height_ratios(hr)
        fig.draw_without_rendering()

    # percentages on the bars: horizontal if they fit the column pitch, else vertical;
    # if a vertical line doesn't fit either, "auto" leaves them out and True shrinks the font
    values = [f"{p:.{value_decimals}f}" for p in pct]
    label_h_in = 0.0
    if show_values:
        pitch_in = ax_dot.get_window_extent().width / fig.dpi / n_cols
        fs_val = fs_pct
        widest = max(
            _measure_text_width_inches(v, fs_val, dpi=100) for v in set(values)
        )
        rotation = 0 if widest <= 0.9 * pitch_in else 90
        if rotation == 90 and 1.2 * fs_val / 72 > 0.9 * pitch_in:
            fs_val = None if show_values == "auto" else 0.9 * pitch_in * 72 / 1.2
            widest *= (fs_val or 0) / fs_pct
        if fs_val:
            gap_pt = max(2.0, 0.3 * fs_val)  # clear space between bar top and number
            for xi, p, v in zip(x, pct, values):
                ax_bar.annotate(
                    v,
                    (xi, p),
                    xytext=(0, gap_pt),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=rotation,
                    fontsize=fs_val,
                )
            label_h_in = (widest if rotation == 90 else 1.2 * fs_val / 72) + gap_pt / 72
    # headroom so the tallest bar's label stays inside the panel
    bar_h_in = ax_bar.get_window_extent().height / fig.dpi
    ax_bar.set_ylim(0, pct.max() / max(0.3, 1 - label_h_in / bar_h_in - 0.03))

    # cn_map's node glyph laid on its side: CN-colored squares with a white edge, joined
    # by a faint line like cn_map's edges. Square side (points) fits the column and row pitch.
    fig.draw_without_rendering()
    box = ax_dot.get_window_extent()
    col_pts = box.width * 72 / fig.dpi / n_cols
    row_pts = box.height * 72 / fig.dpi / n_rows
    side = min(14, 0.7 * col_pts, 0.7 * row_pts)
    gx, gy = np.meshgrid(x, np.arange(n_rows))
    ax_dot.scatter(
        gx.ravel(),
        gy.ravel(),
        s=(0.45 * side) ** 2,
        marker="s",
        color=_style.network_node_default_color,
        linewidths=0,
        zorder=1,
    )
    for xi, c in zip(x, combos):
        ys = [row_of[i] for i in c]
        ax_dot.plot(
            [xi, xi],
            [min(ys), max(ys)],
            color=_style.edge_color_default,
            alpha=0.3,
            linewidth=0.25 * side,
            zorder=2,
        )
        ax_dot.scatter(
            [xi] * len(ys),
            ys,
            s=side**2,
            marker="s",
            c=[palette[names[i]] for i in c],
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    if savefig:
        _savefig(None, output_dir, output_fname, "_CNUpSet", _style)

    plt.show()


def coordinates_on_image(
    df,
    overlay_data,
    color=None,
    x="x",
    y="y",
    fig_width=20,
    fig_height=20,
    dot_size=10,
    convert_to_grey=True,
    scale=False,
    cmap=None,
    savefig=False,
    output_dir="./",
    output_fname="",
    style=None,
    scalebar: bool = True,
    scalebar_length_px=None,
    um_per_px=None,
    scalebar_show_text: bool = True,
):
    """
    Plot coordinates on an image.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame. The DataFrame should have columns 'x' and 'y' representing the coordinates.
    overlay_data : ndarray
        The image data to overlay the coordinates on.
    color : str, optional
        The column name in df for the color variable. Defaults to None.
    x : str, optional
        The column name in df for the x-coordinate. Defaults to "x".
    y : str, optional
        The column name in df for the y-coordinate. Defaults to "y".
    fig_width : int, optional
        The width of the figure. Defaults to 20.
    fig_height : int, optional
        The height of the figure. Defaults to 20.
    dot_size : int, optional
        The size of the dots representing the coordinates. Defaults to 10.
    convert_to_grey : bool, optional
        Whether to convert the image to grayscale. Defaults to True.
    scale : bool, optional
        Whether to scale the color variable. Defaults to False.
    cmap : str, optional
        The colormap to use. Defaults to "inferno".
    savefig : bool, optional
        Whether to save the figure or not. Defaults to False.
    output_dir : str, optional
        The output directory. Defaults to "./".
    output_fname : str, optional
        The output file name. Defaults to "".

    Returns
    -------
    None
    """
    _style = _apply_style(style)
    if cmap is None:
        cmap = _style.sequential_cmap

    plt.figure(figsize=(fig_width, fig_height))

    if convert_to_grey:
        # Convert the image to grayscale
        overlay_data = skimage.color.rgb2gray(overlay_data)
        plt.imshow(overlay_data, cmap="gray")

    else:
        plt.imshow(overlay_data)

    image_width, image_height = overlay_data.shape[1], overlay_data.shape[0]

    if scalebar:
        _add_scalebar(
            plt.gca(),
            image_width,
            _style,
            scalebar_length_px=scalebar_length_px,
            um_per_px=um_per_px,
            show_text=scalebar_show_text,
        )

    # Plot the coordinates on top of the image
    # colorscale by area

    if color != None:
        if scale:
            # minmax scale the variable
            df[color] = (df[color] - df[color].min()) / (
                df[color].max() - df[color].min()
            )
            # change dot size based on variable
            plt.scatter(df[x], df[y], s=df[color] * 30, c=df[color], cmap=cmap)
        else:
            plt.scatter(df[x], df[y], c=df[color], s=dot_size, cmap=cmap)

    else:
        plt.scatter(df[x], df["y"], s=dot_size)

    # add colorbar
    plt.colorbar()

    # set axis limits
    plt.xlim(0, image_width)
    plt.ylim(image_height, 0)

    if savefig:
        _savefig(None, output_dir, output_fname, "_seg_masks_overlay", _style)

    plt.show()


def count_patch_proximity_res(
    adata,
    x,
    hue,
    palette="Set3",
    order=True,
    key_name="ppa_result",
    savefig=False,
    output_dir="./",
    output_fname="",
    style=None,
):
    """
    Create a count plot for patch proximity results.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    x : str
        The column name in the DataFrame for the x-axis variable.
    hue : str
        The column name in the DataFrame for the hue variable.
    palette : str, optional
        The palette to use for the plot. Defaults to "Set3".
    order : bool, optional
        Whether to order the count plot. Defaults to True.
    key_name : str, optional
        The key name for the patch proximity results in adata.uns. Defaults to "ppa_result".
    savefig : bool, optional
        Whether to save the figure or not. Defaults to False.
    output_dir : str, optional
        The output directory. Defaults to "./".
    output_fname : str, optional
        The output file name. Defaults to "".

    Returns
    -------
    None
    """
    _style = _apply_style(style)
    region_results = adata.uns[key_name]
    ax = sns.countplot(
        x=x,
        hue=hue,
        data=region_results,
        palette=palette,
        order=region_results[x].value_counts().index,
    )
    _apply_spines(ax, _style)

    tick_positions = range(len(region_results[x].value_counts()))
    tick_labels = region_results[x].value_counts().index
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=_style.font_size_tiny)
    if savefig:
        _savefig(None, output_dir, output_fname, "_count_ppa_result", _style)

    plt.show()


def BC_projection(
    adata,
    cnmap_dict,
    cn_col,
    plot_list,  # list of 3 elements from cn_col
    cn_col_annt=None,
    palette=None,
    figsize=(7, 7),
    rand_seed=1,
    SMALL_SIZE=None,
    MEDIUM_SIZE=None,
    BIGGER_SIZE=None,
    n_num=None,
    threshold=None,
    savefig=False,
    output_fname="",
    output_dir="",  # output directory
    dpi=None,
    style=None,
):
    """
    Plot barycentric projection.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    cnmap_dict : dict
        Dictionary containing keys 'w', 'l', 'k', and 'threshold'.
    cn_col : str
        Column name in adata.obs containing copy number information.
    plot_list : list
        List of 3 elements from cn_col.
    cn_col_annt : str, optional
        Annotated column name, by default None.
    palette : dict, optional
        Color palette -- any category not covered is colored grey, with a warning. By
        default None, which reuses/creates a persistent mapping cached in
        `adata.uns[cn_col + "_colors"]`, consistent with any other function (e.g. `catplot`)
        already plotting `cn_col`.
    figsize : tuple, optional
        Figure size, by default (7, 7).
    rand_seed : int, optional
        Seed for the point jitter, by default 1.
    SMALL_SIZE : int, optional
        Font size for small text, by default 14.
    MEDIUM_SIZE : int, optional
        Font size for medium text, by default 16.
    BIGGER_SIZE : int, optional
        Font size for large text, by default 18.
    n_num : int, optional
        The window size (``k``) used when the CN combinations in ``cnmap_dict``
        were computed. Defaults to ``cnmap_dict["k"]`` if not given.
    threshold : float, optional
        Minimum combined coverage (as a 0-1 fraction) the three ``plot_list``
        CNs must reach within a window for that window to be plotted -- e.g.
        ``0.9`` means "these three CNs together cover at least 90% of the
        window". Defaults to ``cnmap_dict["threshold"]`` if not given.
    savefig : bool, optional
        Whether to save the figure, by default False.
    output_fname : str, optional
        Output file name, by default "".
    output_dir : str, optional
        Output directory, by default "".
    dpi : int, optional
        Dots per inch, by default 300.
    """
    import warnings as _warnings

    _style = _apply_style(style)
    if SMALL_SIZE is not None or MEDIUM_SIZE is not None or BIGGER_SIZE is not None:
        _warnings.warn(
            "SMALL_SIZE, MEDIUM_SIZE, and BIGGER_SIZE are deprecated. "
            "Use the 'style' parameter (SPACEcStyle) to control font sizes.",
            DeprecationWarning,
            stacklevel=2,
        )
    _small = SMALL_SIZE if SMALL_SIZE is not None else _style.font_size_small
    _medium = MEDIUM_SIZE if MEDIUM_SIZE is not None else _style.font_size_medium
    _bigger = BIGGER_SIZE if BIGGER_SIZE is not None else _style.font_size_large
    _dpi = dpi if dpi is not None else _style.figure_dpi

    if cn_col_annt is None:
        plot_list_annot = plot_list
        all_list = list(adata.obs[cn_col].unique())
        if all(x in all_list for x in plot_list):
            pass
        else:
            print(
                "Please provide a valid list of plot_list from cn_col or specify cn_col_annt!"
            )
            return
    else:
        if cn_col == cn_col_annt:
            plot_list_annot = plot_list
        else:
            annt_map = dict(
                zip(adata.obs[cn_col].unique(), adata.obs[cn_col_annt].unique())
            )
            plot_list_annot = [annt_map[i] for i in plot_list]

    # generate color -- always keyed to cn_col (not cn_col_annt, which is only ever used
    # above to relabel the legend text, not to look up colors)
    categories = sorted(adata.obs[cn_col].dropna().unique().tolist())
    palette = _resolve_categorical_palette(
        _style, adata, cn_col, categories, palette=palette
    )

    w = cnmap_dict["w"]
    l = cnmap_dict["l"]
    if n_num is None:
        n_num = cnmap_dict["k"]
    else:
        n_num = n_num

    if threshold is None:
        threshold = cnmap_dict["threshold"]
        threshold = threshold * 100
    else:
        threshold = threshold * 100

    # lmap = {j: i for i, j in enumerate(l)}

    # w[plot_list].sum(axis=1) is a raw window count capped at n_num (e.g. 0-75), while
    # `threshold` above was scaled to a 0-100 percent (threshold * 100) -- comparing a
    # raw count directly against a percent silently makes any realistic coverage
    # threshold (e.g. 0.9 -> 90) unreachable for n_num < 90, producing an empty
    # (blank) plot with no error. Normalize the raw count to the same 0-100 percent
    # scale before comparing.
    wgc = w.loc[(w.loc[:, plot_list].sum(axis=1) / n_num * 100) > threshold, :]
    # idx = wgc.index.values
    xl = wgc.loc[:, plot_list]
    proj = np.array([[0, 0], [np.cos(np.pi / 3), np.sin(np.pi / 3)], [1, 0]])
    coords = np.dot(xl / n_num, proj)  #####window size fraction

    jit = 0.002
    rng = np.random.default_rng(rand_seed)
    cols = [palette[a] for a in wgc[cn_col]]
    top = (np.cos(np.pi / 3), np.sin(np.pi / 3))

    # style applied only for this figure (no global rcParams change)
    with _style.context():
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            coords[:, 0] + jit * rng.standard_normal(len(coords)),
            coords[:, 1] + jit * rng.standard_normal(len(coords)),
            s=15,
            alpha=0.5,
            c=cols,
        )

        # corner labels anchored at the triangle vertices
        label_kws = dict(fontsize=_style.font_size_annotation, zorder=5)
        ax.text(0, -0.02, plot_list_annot[0], ha="right", va="top", **label_kws)
        ax.text(1, -0.02, plot_list_annot[2], ha="left", va="top", **label_kws)
        ax.text(
            top[0],
            top[1] + 0.02,
            plot_list_annot[1],
            ha="center",
            va="bottom",
            **label_kws,
        )
        # full triangle outline and limits, even when the windows cover only part of it
        ax.plot(
            [0, top[0], 1, 0], [0, top[1], 0, 0], color="grey", linewidth=0.8, zorder=0
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, top[1] + 0.02)
        ax.set_aspect("equal")
        ax.axis("off")

        # legend below the triangle, only neighborhoods present in the plotted windows
        present = set(wgc[cn_col])
        ax.legend(
            handles=[
                mpatches.Patch(color=color, label=label)
                for label, color in palette.items()
                if label in present
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=3,
            fontsize=_style.font_size_small,
            frameon=_style.legend_frameon,
        )

        if savefig:
            _savefig(fig, output_dir, output_fname, "_bc_proj", _style, dpi=_dpi)

        plt.show()


def distance_graph(
    dist_table,
    distance_pvals,
    palette=None,
    condition_pair=None,
    interaction_col="interaction",
    condition_col="condition",
    logfold_group_col="logfold_group",
    celltype1_col="celltype1",
    celltype2_col="celltype2",
    pair_col="pairs",
    with_labels=True,
    node_size=910,
    font_size=None,
    multiplication_factor=10,
    savefig=False,
    output_fname="",
    output_dir="",  # output directory
    dpi=None,
    style=None,
):
    """
    Generates a distance graph from a dataframe.

    Parameters
    ----------
    dist_table : pandas.DataFrame
        The distance table (e.g. from `sp.tl.filter_interactions`).
    distance_pvals : pandas.DataFrame
        The distance p-value table.
    palette : dict, optional
        A dictionary mapping nodes to colors -- any node not covered is
        colored grey, with a warning. If None, colors every node in `distance_pvals`'s
        `celltype1_col`/`celltype2_col` via the active style's categorical palette (there's no
        AnnData here to cache into, so this isn't reused across separate calls the way
        `catplot`/etc. reuse `adata.uns[col + "_colors"]`).
    condition_pair : list, optional
        A list of two conditions to compare. If None, uses unique conditions from the dataframe.
    interaction_col : str, optional
        The name of the interaction column in the dataframe. Defaults to 'interaction'.
    condition_col : str, optional
        The name of the condition column in the dataframe. Defaults to 'condition'.
    logfold_group_col : str, optional
        The name of the logfold group column in the dataframe. Defaults to 'logfold_group'.
    celltype1_col : str, optional
        The name of the first cell type column in the dataframe. Defaults to 'celltype1'.
    celltype2_col : str, optional
        The name of the second cell type column in the dataframe. Defaults to 'celltype2'.
    pair_col : str, optional
        The name holding pair identifiers in `dist_table` (e.g. "celltype1_celltype2").
        Accepted as either a column or the DataFrame's index -- `sp.tl.filter_interactions`'s own `dist_table`
        output carries this as its index, not a column. Defaults to 'pairs'.
    with_labels : bool, optional
        Whether to draw labels on the nodes. Defaults to True.
    node_size : int, optional
        The size of the nodes. Defaults to 910.
    font_size : int, optional
        The font size for the labels. Defaults to 7.

    Returns
    -------
    None
        The function generates a plot and does not return anything.
    """

    _style = _apply_style(style)
    if font_size is None:
        font_size = _style.font_size_small
    if dpi is None:
        dpi = _style.figure_dpi

    # Generate df for graph
    # this is to restrict to the pairs that were filtered
    # sp.tl.filter_interactions returns dist_table with `pair_col` as the index (its
    # own internal .set_index(pair_col) + .pivot()), not a column -- accept either
    # shape here rather than requiring the caller to reset_index() first.
    if pair_col in dist_table.columns:
        pairs = dist_table[pair_col].unique()
    else:
        pairs = dist_table.index.unique()
    distance_pvals[pair_col] = (
        distance_pvals[celltype1_col] + "_" + distance_pvals[celltype2_col]
    )
    distance_pvals = distance_pvals[distance_pvals[pair_col].isin(pairs)]
    print(distance_pvals.shape)
    pairs = distance_pvals[interaction_col].unique()
    distance_pvals_grouped = distance_pvals.groupby([celltype1_col, celltype2_col])

    result_list = {}

    if condition_pair == None:
        conditions = distance_pvals[condition_col].unique()
    else:
        conditions = condition_pair

    for p in pairs:
        distance_pvals_c1 = distance_pvals[
            (distance_pvals[interaction_col] == p)
            & (distance_pvals[condition_col] == conditions[0])
        ]
        distance_pvals_c2 = distance_pvals[
            (distance_pvals[interaction_col] == p)
            & (distance_pvals[condition_col] == conditions[1])
        ]

        difference = abs(
            distance_pvals_c2[logfold_group_col].values[0]
            - distance_pvals_c1[logfold_group_col].values[0]
        )

        if (
            0
            > (
                distance_pvals_c2[logfold_group_col].values[0]
                - distance_pvals_c1[logfold_group_col]
            ).values[0]
        ):
            direction = _style.accent_color_positive
        else:
            direction = _style.accent_color_negative

        df_res = pd.DataFrame(
            {
                celltype1_col: distance_pvals_c2[celltype1_col],
                celltype2_col: distance_pvals_c2[celltype2_col],
                "difference": difference,
                "direction": direction,
            }
        )

        result_list[p] = df_res

    graph_df = pd.concat(result_list.values())
    graph_df = graph_df.drop_duplicates(
        subset=[celltype1_col, celltype2_col], keep="first"
    )

    ## remove nan values
    graph_df = graph_df[graph_df["difference"].notna()]

    # Generate graph
    G = nx.from_pandas_edgelist(
        graph_df,
        source=celltype1_col,
        target=celltype2_col,
        edge_attr=["difference", "direction"],
    )

    for u, v, d in G.edges(data=True):
        d["weight"] = (
            1e-100
            if d["difference"] == 0
            else abs(d["difference"] * multiplication_factor)
        )

    # Generate plot
    weights = [G[u][v]["weight"] for u, v in G.edges()]

    pos = nx.circular_layout(G)

    edge_colors = [d["direction"] for u, v, d in G.edges(data=True)]

    combined_list = (
        distance_pvals[celltype1_col].tolist() + distance_pvals[celltype2_col].tolist()
    )
    # sorted(), not the plain set() this used to iterate -- a bare set's iteration order
    # for strings isn't guaranteed stable across processes/hash seeds, so the old
    # color-per-node assignment could silently vary run to run even with a fixed palette.
    unique_values = sorted(set(combined_list))
    palette = _resolve_categorical_palette(
        _style, None, None, unique_values, palette=palette
    )
    node_colors = [palette[node] for node in G.nodes()]

    nx.draw(
        G,
        pos,
        node_color=node_colors,
        with_labels=False,  # Don't draw labels here
        edge_color=edge_colors,
        node_size=node_size,
        width=weights,
    )

    for node, (x, y) in pos.items():
        wrapped_node = textwrap.fill(
            node, 10, break_long_words=False
        )  # split at spaces only, never inside a word
        plt.text(
            x,
            y,
            wrapped_node,
            fontsize=font_size,
            color="white" if is_dark(palette.get(node, "lightgrey")) else "black",
            ha="center",
            va="center",
        )

    # add legend
    plt.plot(
        [],
        [],
        color=_style.accent_color_positive,
        label="More distant in " + conditions[0] + " than " + conditions[1],
    )
    plt.plot(
        [],
        [],
        color=_style.accent_color_negative,
        label="Closer in " + conditions[0] + " than " + conditions[1],
    )

    # add width legend
    plt.plot(
        [],
        [],
        color="k",
        label="edge width = absolute difference * " + str(multiplication_factor),
    )

    # position legend at bottom middle
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc="lower center", ncol=2)

    if savefig:
        _savefig(None, output_dir, output_fname, "_dist_graph", _style, dpi=dpi)

    plt.show()


def plot_masks(
    adata,
    color,
    unique_region,
    subset=None,
    X="x",
    Y="y",
    invert_y=False,
    palette=None,  # default is None which means the color comes from the anndata object
    savefig=False,
    output_dir="./",
    output_fname="",
    figsize=5,
    background="white",
    n_columns=4,
    outline=True,  # New parameter to add outline
    style=None,
    scalebar: bool = True,
    scalebar_length_px=None,
    um_per_px=None,
    scalebar_show_text: bool = True,
):
    """
    Plot segmentation masks colored by categorical annotation.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    color : str
        Column name in `adata.obs` that contains the color information.
    unique_region : str
        Column name in `adata.obs` that contains the unique region information.
    subset : str, optional
        Specific region to plot. If None, all regions are plotted. Default is None.
    X : str, optional
        Column name in `adata.obs` for x-coordinates. Default is "x".
    Y : str, optional
        Column name in `adata.obs` for y-coordinates. Default is "y".
    invert_y : bool, optional
        If True, invert the y-coordinates. Default is False.
    palette : dict or str, optional
        A ``{category: color}`` dict -- any category not covered is colored grey, with a
        warning -- or a matplotlib colormap name to sample evenly across `color`'s
        categories. Default is None, which reuses/creates a persistent mapping cached in
        `adata.uns[color + "_colors"]`, consistent with any other function (e.g. `catplot`)
        already plotting `color`.
    savefig : bool, optional
        If True, save the figure to a file. Default is False.
    output_dir : str, optional
        Directory to save the figure if `savefig` is True. Default is "./".
    output_fname : str, optional
        Filename to save the figure if `savefig` is True. Default is "".
    figsize : int, optional
        Size of each subplot. Default is 5.
    background : str, optional
        Background color, either 'black' or 'white'. Default is "white".
    n_columns : int, optional
        Number of columns in the subplot grid. Default is 4.
    outline : bool, optional
        If True, add an outline to the masks. Default is False.

    Returns
    -------
    None
        This function does not return any value. It plots the segmentation masks.

    Notes
    -----
    The function uses `regionprops` from `skimage.measure` to calculate centroids of the masks and `cKDTree` from `scipy.spatial` for fast spatial lookup.
    """

    _style = _apply_style(style)
    df = pd.DataFrame(adata.obs[[X, Y, color, unique_region]])
    df[color] = df[color].astype("category")
    if invert_y:
        df[Y] *= -1

    categories = list(df[color].cat.categories)
    palette = _resolve_categorical_palette(
        _style, adata, color, categories, palette=palette
    )
    # the mask-coloring lookup below matches `color` values cast to str -- str-key the
    # (possibly native-dtype, e.g. int cluster id) resolved palette to match, same fix as
    # `catplot`'s own mask-rendering path. Also normalize every value to an RGB tuple: the
    # `[:3]` slice below silently mis-slices a hex string or plain color name (e.g. "grey",
    # _resolve_categorical_palette's fallback for an uncovered category) instead of dropping
    # an alpha channel from an actual RGB(A) tuple.
    palette = {str(k): mcolors.to_rgb(v) for k, v in palette.items()}

    # subset regions
    if subset is None:
        region_list = list(
            pd.Series(df[unique_region].unique()).sort_values()
        )  # display all experiments
    else:
        if subset not in list(pd.Series(df[unique_region].unique()).sort_values()):
            print(subset + " is not in unique_region!")
            return
        else:
            region_list = [subset]

    n_rows = int(np.ceil(len(region_list) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(figsize * n_columns, figsize * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": _style.subplot_wspace, "hspace": _style.subplot_hspace},
    )

    if background == "black":
        fig.patch.set_facecolor("black")
        for ax in axes.flatten():
            ax.set_facecolor("black")
    else:
        fig.patch.set_facecolor("white")
        for ax in axes.flatten():
            ax.set_facecolor("white")

    for i_ax, (name, ax) in enumerate(zip(region_list, axes.flatten())):
        data = df[df[unique_region] == name]

        # test if name is in the masks
        if name not in adata.uns["masks"].keys():
            print(
                name
                + " is not a valid keys in adata.uns['masks']. Please check if the masks were named correctly. Masks should be named after a region, batch or condition."
            )
            print("These are the valid keys: ", adata.uns["masks"].keys())
            return
        masks = adata.uns["masks"][name]

        if background == "black":
            background_color = [0, 0, 0]
            outline_color = [1, 1, 1]  # White outline
        else:
            if background == "white":
                background_color = [1, 1, 1]
                outline_color = [0, 0, 0]  # Black outline
            else:
                print(
                    "Choose either black or white for background, defaulting to black"
                )
                background_color = [0, 0, 0]
                outline_color = [1, 1, 1]  # White outline

        # Calculate centroids
        props = regionprops(masks)
        centroids = [
            prop.centroid for prop in props if prop.label != 0
        ]  # Exclude background

        # Create a KDTree for fast spatial lookup
        # Built in (row, col) = (Y, X) order to match regionprops' centroid convention
        tree = cKDTree(data[[Y, X]].values)

        # Initialize color map with zeros (black for background)
        color_map = np.zeros((masks.max() + 1, 4))  # RGBA
        color_map[0, :3] = background_color  # Set the background color
        color_map[0, 3] = 1  # Set alpha for background

        # Query the KDTree for all centroids at once
        distances, indices = tree.query(centroids)

        valid_labels = np.array([prop.label for prop in props if prop.label != 0])
        cell_types = data.iloc[indices][color].astype(str).values

        # Ensure each dot is mapped to at most one mask
        unique_indices, unique_pos = np.unique(indices, return_index=True)
        valid_labels = valid_labels[unique_pos]
        cell_types = cell_types[unique_pos]

        # Assign colors using numpy indexing, excluding background
        color_map[valid_labels, :3] = np.array(
            [
                mcolors.to_rgb(palette.get(cell_type, [0, 0, 0]))
                for cell_type in cell_types
            ]
        )
        color_map[valid_labels, 3] = 1  # Set alpha for valid labels

        # Create a colored image using vectorized operations
        colored_masks = color_map[masks]

        if outline:
            boundaries = find_boundaries(masks, mode="outer")
            valid_boundaries = boundaries & np.isin(masks, valid_labels)
            colored_masks[valid_boundaries, :3] = outline_color
            colored_masks[valid_boundaries, 3] = 1  # Ensure outline is not transparent

        # Set alpha to 0 for all invalid masks
        colored_masks[masks == 0, 3] = 0

        # Rotate the colored masks 90 degrees to the left
        rotated_colored_masks = np.rot90(colored_masks)

        # Check if background is black
        if background == "black":
            legend_text_color = "white"
            title_color = "white"
        else:
            legend_text_color = "black"  # default color
            title_color = "black"  # default color

        # Plot the rotated colored masks
        ax.imshow(rotated_colored_masks)
        ax.axis("off")  # Hide the axis

        ax.set_title(name, color=title_color)
        ax.set_aspect("equal")

        if scalebar:
            _add_scalebar(
                ax,
                rotated_colored_masks.shape[1],
                _style,
                scalebar_length_px=scalebar_length_px,
                um_per_px=um_per_px,
                show_text=scalebar_show_text,
            )

        # Add padding to the legend
        # Create custom legend
        unique_cell_types = data[color].unique()
        legend_patches = [
            mpatches.Patch(color=palette[cell_type], label=cell_type)
            for cell_type in unique_cell_types
            if cell_type in palette
        ]
        legend = ax.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize="large",
            title=color,
            title_fontsize="large",
        )
        plt.setp(legend.get_texts(), color=legend_text_color)  # Set legend text color
        plt.setp(legend.get_title(), color=legend_text_color)
        # frame = legend.get_frame()
        # frame.set_facecolor('white')  # Adjust the legend background color

    for i in range(i_ax + 1, n_rows * n_columns):
        axes.flatten()[i].axis("off")

    if savefig:
        _savefig(fig, output_dir, output_fname, "_spatial_plot", _style)

    plt.show()


def _resolve_continuous_values(adata, name, obs_mask, layer=None):
    """Return one continuous value per cell selected by ``obs_mask`` for ``name``.

    ``obs_mask`` is a boolean mask aligned to ``adata.obs`` by position, not a set
    of index labels -- ``adata.obs.index`` is not guaranteed unique, so selecting
    by label could silently return more rows than requested.

    ``name`` is looked up in ``adata.var_names`` first, then in
    ``adata.obs.columns``, so both a marker (e.g. ``"CD21"``) and a numeric
    observation (e.g. ``"area"``) can be used to color cells.
    """
    if name in adata.var_names:
        sub = adata[obs_mask, name]
        X = sub.layers[layer] if layer is not None else sub.X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        return np.asarray(X).flatten().astype(float)

    if name in adata.obs.columns:
        col = adata.obs.loc[obs_mask, name]
        if not pd.api.types.is_numeric_dtype(col):
            raise ValueError(
                f"adata.obs['{name}'] is not numeric -- expression_on_masks "
                "paints continuous values. Use spacec.pl.plot_masks for "
                "categorical annotations such as cell types."
            )
        return col.to_numpy(dtype=float)

    raise ValueError(f"'{name}' was not found in adata.var_names or adata.obs.columns.")


def _resolve_continuous_values_from_df(df, name, obs_mask):
    """``_resolve_continuous_values``'s counterpart for a plain DataFrame (no X/obs
    split -- ``name`` is just looked up as a column directly).

    ``obs_mask`` is a boolean mask aligned to ``df`` by position, not a set of
    index labels -- see ``_resolve_continuous_values`` for why that matters.
    """
    if name not in df.columns:
        raise ValueError(f"'{name}' was not found in df.columns.")
    col = df.loc[obs_mask, name]
    if not pd.api.types.is_numeric_dtype(col):
        raise ValueError(
            f"df['{name}'] is not numeric -- expression_on_masks paints continuous "
            "values. Use spacec.pl.plot_masks for categorical annotations such as "
            "cell types."
        )
    return col.to_numpy(dtype=float)


def feature_outlier_grid(
    adata,
    reference_imgs,
    feature_cols,
    region_col="unique_region",
    label_col="label",
    half_window=45,
    show_all_contacts_for=("n_contacts",),
    figsize=None,
    style=None,
):
    """
    For each feature in `feature_cols`, plot its distribution alongside a crop of the
    lowest- and highest-scoring cell, so extreme values can be sanity-checked against a real
    example rather than trusting the number alone.

    Masks are read from `adata.uns["masks"]` (see :func:`spacec.hf.hf_attach_masks` /
    :func:`spacec.hf.hf_load_masks`) and matched to observations by an exact `label_col` join
    within a region -- the same convention :func:`expression_on_masks` uses -- rather than a
    pixel-centroid lookup.

    Parameters
    ----------
    adata : AnnData
        Must have `feature_cols`, `region_col`, and `label_col` in `.obs`, and masks attached
        under `adata.uns["masks"]`.
    reference_imgs : dict[str, numpy.ndarray]
        Maps each region to a 2D reference image (e.g. DAPI) of the same shape as its mask, used
        as the crop background.
    feature_cols : sequence of str
        obs columns to plot, one row per feature.
    region_col : str, optional
        Column in `adata.obs` keying `adata.uns["masks"]`/`reference_imgs`. By default
        "unique_region".
    label_col : str, optional
        Column in `adata.obs` holding each cell's mask label. By default "label".
    half_window : int, optional
        Crop half-width/height in pixels around each example cell, by default 45.
    show_all_contacts_for : sequence of str, optional
        Feature names for which every other cell's outline in the crop is also drawn (in
        yellow), not just the example cell's (in red) -- useful for "n_contacts"-style features.
        By default ("n_contacts",).
    figsize : tuple, optional
        Defaults to (9, 3 * len(feature_cols)).
    style : SPACEcStyle or str or pathlib.Path, optional
        Style to use for this call.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _style = _apply_style(style)

    masks_dict = adata.uns.get("masks")
    if not masks_dict:
        raise ValueError(
            "adata.uns['masks'] is missing or empty -- attach segmentation "
            "masks first with spacec.hf.hf_attach_masks (loading them with "
            "spacec.hf.hf_load_masks)."
        )
    masks_dict = {str(key): value for key, value in masks_dict.items()}

    for col in (region_col, label_col):
        if col not in adata.obs.columns:
            raise ValueError(
                f"adata.obs has no '{col}' column, which is required to match "
                "observations to their mask pixels."
            )

    feature_cols = list(feature_cols)
    missing_features = [f for f in feature_cols if f not in adata.obs.columns]
    if missing_features:
        raise ValueError(f"adata.obs is missing feature columns: {missing_features}")

    figsize = figsize or (9, 3 * len(feature_cols))

    with _style.context():
        fig, axes = plt.subplots(len(feature_cols), 3, figsize=figsize, squeeze=False)

        for i, feature in enumerate(feature_cols):
            values = adata.obs.loc[adata.obs[feature].notna(), feature]

            ax_hist = axes[i, 0]
            sns.histplot(values, bins=60, ax=ax_hist, color=_style.accent_color_neutral)
            ax_hist.set_title(feature)
            ax_hist.grid(False)
            sns.despine(ax=ax_hist)

            for col_i, idx, tag in (
                (1, values.idxmin(), "low"),
                (2, values.idxmax(), "high"),
            ):
                row = adata.obs.loc[idx]
                region = str(row[region_col])
                label = int(row[label_col])
                mask = masks_dict[region]
                ref_img = reference_imgs[region]

                ys, xs = np.where(mask == label)
                if ys.size == 0:
                    raise ValueError(
                        f"label {label} (region {region!r}) not found in its attached mask -- "
                        "adata.obs and adata.uns['masks'] are out of sync."
                    )
                cy, cx = int(round(ys.mean())), int(round(xs.mean()))

                y0, y1 = max(cy - half_window, 0), min(
                    cy + half_window, ref_img.shape[0]
                )
                x0, x1 = max(cx - half_window, 0), min(
                    cx + half_window, ref_img.shape[1]
                )
                mask_crop = mask[y0:y1, x0:x1]

                ax = axes[i, col_i]
                ax.imshow(ref_img[y0:y1, x0:x1], cmap="gray")
                if feature in show_all_contacts_for:
                    other_boundaries = (
                        find_boundaries(mask_crop, mode="inner")
                        & (mask_crop != 0)
                        & (mask_crop != label)
                    )
                    overlay = np.zeros(mask_crop.shape + (4,), dtype=float)
                    overlay[other_boundaries] = mcolors.to_rgba("yellow")
                    ax.imshow(overlay)
                ax.contour(mask_crop == label, colors="red", linewidths=1.5)
                ax.set_title(f"{tag}: {row[feature]:.2f} ({region})", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

        fig.tight_layout()

    # Show explicitly so the figure renders regardless of how the call sits in
    # the cell (trailing ';', assigned, in a loop). End the call with ';' or
    # assign the result to avoid a second render of the returned Figure.
    plt.show()
    return fig


def expression_on_masks(
    adata,
    marker,
    subset=None,
    region_col="unique_region",
    label_col="label",
    layer=None,
    masks=None,
    cmap=None,
    background="black",
    vmin=None,
    vmax=None,
    shared_scale=True,
    outline=False,
    n_columns=4,
    figsize=5,
    colorbar_label=None,
    savefig=False,
    output_dir="./",
    output_fname="",
    style=None,
    scalebar: bool = True,
    scalebar_length_px=None,
    um_per_px=None,
    scalebar_show_text: bool = True,
):
    """
    Recolor segmentation masks by a continuous per-cell value.

    The continuous counterpart to :func:`plot_masks`: instead of filling each
    cell with its cell-type color, every pixel of a cell takes that cell's
    marker expression (or numeric ``obs`` value), showing the actual segmented
    cell shape behind each measurement rather than a centroid dot.

    Masks are read from ``adata.uns["masks"]`` (see
    :func:`spacec.hf.hf_attach_masks` / :func:`spacec.hf.hf_load_masks`) and
    matched to observations by an exact ``label_col`` join within a region.
    Cells dropped upstream by ``filter_data``/``remove_noise`` are absent from
    ``adata.obs``, so their pixels fall back to ``background`` -- a direct look
    at what filtering removed.

    ``adata`` also accepts a plain DataFrame (e.g. ``df_nn``, before
    :func:`spacec.hf.make_anndata`), to inspect marker expression -- and decide a
    ``nonFuncAb_list`` -- before building the AnnData at all. In that mode ``masks``
    is required (there's no ``.uns`` to source it from, e.g. pass the dict straight
    from :func:`spacec.hf.hf_load_masks`) and ``layer`` has no meaning.

    Either way, before plotting each region's masks are checked against that region's
    table rows by label identity, not just count: every ``label_col`` value present in
    the table must actually appear as a pixel value in that region's mask (filtered-out
    cells being absent from the *table* is normal; a table label missing from the
    *mask* is not -- that only happens if the wrong mask was loaded for a region).

    Parameters
    ----------
    adata : AnnData or DataFrame
        Annotated data matrix with masks attached under ``adata.uns["masks"]``, or a
        plain DataFrame (requires ``masks`` to be given explicitly).
    marker : str or list of str
        Name(s) to color by. For an AnnData, resolved against ``adata.var_names``
        first, then against numeric columns of ``adata.obs``. For a DataFrame, looked
        up directly as a column.
    subset : str or list of str, optional
        Region(s) to plot. If None, all regions that have a mask are plotted.
    region_col : str, optional
        Column (in ``adata.obs``, or in the DataFrame) keying the masks.
    label_col : str, optional
        Column (in ``adata.obs``, or in the DataFrame) holding each cell's mask label.
    layer : str, optional
        Layer to read marker values from instead of ``adata.X``. Only meaningful when
        ``adata`` is an AnnData; raises if given together with a DataFrame.
    masks : dict[str, numpy.ndarray], optional
        Mapping of region -> mask array, e.g. from :func:`spacec.hf.hf_load_masks`.
        Required when ``adata`` is a DataFrame. When ``adata`` is an AnnData, this
        overrides ``adata.uns["masks"]`` for this call if given; otherwise
        ``adata.uns["masks"]`` is used as before.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for the values.
    background : str, optional
        Color for background pixels and unmatched cells.
    vmin, vmax : float, optional
        Explicit color scale limits.
    shared_scale : bool, optional
        If True, all regions for one marker share a color scale.
    outline : bool, optional
        If True, outline each matched cell.
    n_columns : int, optional
        Maximum number of columns in the subplot grid.
    figsize : int or float, optional
        Size of each subplot in inches.
    colorbar_label : str, optional
        Colorbar label. Defaults to the marker name.
    savefig : bool, optional
        If True, save the figure.
    output_dir : str, optional
        Directory in which to save the figure.
    output_fname : str, optional
        Base filename for the saved figure.
    style : SPACEcStyle or str or pathlib.Path, optional
        Style to use for this call.
    scalebar : bool, optional
        If True, add a scale bar to each panel.
    scalebar_length_px : int, optional
        Explicit scale bar length in pixels.
    um_per_px : float, optional
        Micrometres per pixel.
    scalebar_show_text : bool, optional
        If True, label the scale bar.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing one panel per region and marker combination.

    Notes
    -----
    Masks are drawn in their stored orientation using ``imshow``, unlike
    :func:`plot_masks`, which rotates them by 90 degrees.

    The figure is shown with ``plt.show()`` and also returned. In a notebook,
    end the call with ``;`` or assign the result to avoid rendering it twice.
    """
    _style = _apply_style(style)

    is_df = isinstance(adata, pd.DataFrame)

    if is_df:
        if layer is not None:
            raise ValueError(
                "'layer' is only meaningful when 'adata' is an AnnData object (it "
                "selects an AnnData layer); it has no effect when 'adata' is a "
                "DataFrame."
            )
        if masks is None:
            raise ValueError(
                "'masks' is required when 'adata' is a DataFrame (e.g. from "
                "spacec.hf.hf_load_masks) -- there is no '.uns' to source masks "
                "from."
            )
        table = adata
        masks_dict = masks
        table_desc = "df"
    else:
        table = adata.obs
        masks_dict = masks if masks is not None else adata.uns.get("masks")
        table_desc = "adata.obs"

    if not masks_dict:
        raise ValueError(
            "No masks available -- pass 'masks' directly, or (for an AnnData) "
            "attach segmentation masks first with spacec.hf.hf_attach_masks "
            "(loading them with spacec.hf.hf_load_masks)."
        )

    for col in (region_col, label_col):
        if col not in table.columns:
            raise ValueError(
                f"{table_desc} has no '{col}' column, which is required to match "
                "observations to their mask pixels."
            )

    if not isinstance(n_columns, int) or isinstance(n_columns, bool):
        raise TypeError("n_columns must be a positive integer.")
    if n_columns < 1:
        raise ValueError("n_columns must be at least 1.")
    if figsize <= 0:
        raise ValueError("figsize must be greater than 0.")

    # Normalize mask dictionary keys so region comparisons are consistent.
    masks_dict = {str(key): value for key, value in masks_dict.items()}
    region_values = table[region_col].astype(str)

    if subset is None:
        regions = sorted(set(region_values.unique()).intersection(masks_dict.keys()))
        if not regions:
            raise ValueError(
                f"no region in {table_desc}['{region_col}'] has a mask. "
                f"Valid mask keys: {sorted(masks_dict)}"
            )
    else:
        requested_regions = [subset] if isinstance(subset, str) else list(subset)
        regions = [str(region) for region in requested_regions]

        unknown = [
            requested
            for requested, normalized in zip(requested_regions, regions)
            if normalized not in masks_dict
        ]
        if unknown:
            raise ValueError(
                f"no mask found for region(s) {unknown}. Valid keys: "
                f"{sorted(masks_dict)}"
            )

    # Check the masks actually belong to this data, by label identity rather than
    # count: filtered-out cells being absent from the table is expected (their label
    # just won't be requested), but a table label missing from its own region's mask
    # is not -- every label a table row carries came directly off that region's mask
    # during feature extraction, so it must exist there if the masks are the right
    # ones. A near-zero overlap (wrong region's mask, or a stale/re-segmented file)
    # is what this is catching; a small tolerance below 100% allows for edge noise.
    for region in regions:
        in_region = region_values == str(region)
        table_labels = set(table.loc[in_region, label_col].to_numpy().astype(int))
        if not table_labels:
            continue
        mask_labels = set(np.unique(masks_dict[region])) - {0}
        overlap_frac = len(table_labels & mask_labels) / len(table_labels)
        if overlap_frac < 0.95:
            raise ValueError(
                f"masks for region {region!r} do not look like they belong to this "
                f"data: only {overlap_frac:.0%} of the '{label_col}' values present "
                f"in {table_desc} for this region were found as labels in its mask "
                "(expected ~100%). Check you loaded the right mask for this region."
            )

    markers = [marker] if isinstance(marker, str) else list(marker)
    if not markers:
        raise ValueError("marker must contain at least one marker name.")

    plot_cmap = plt.get_cmap(
        cmap if cmap is not None else _style.sequential_cmap
    ).copy()
    plot_cmap.set_bad(background)

    # Build every panel before plotting so shared limits can be calculated.
    panels = []
    for name in markers:
        for region in regions:
            in_region = region_values == str(region)
            obs_region = table.loc[in_region]

            labels_arr = obs_region[label_col].to_numpy().astype(int)
            if is_df:
                values_arr = _resolve_continuous_values_from_df(
                    table,
                    name,
                    in_region,
                )
            else:
                values_arr = _resolve_continuous_values(
                    adata,
                    name,
                    in_region,
                    layer=layer,
                )

            region_mask = masks_dict[str(region)]
            image = _continuous_mask_image(
                region_mask,
                labels_arr,
                values_arr,
            )

            panels.append((name, region, region_mask, labels_arr, image))

    if not panels:
        raise ValueError("No panels are available to plot.")

    # Calculate color limits independently for each marker.
    limits = {}
    for name in markers:
        marker_images = [panel[4] for panel in panels if panel[0] == name]

        if shared_scale:
            finite_values = [
                image.compressed() for image in marker_images if image.count() > 0
            ]
            finite_values = (
                np.concatenate(finite_values)
                if finite_values
                else np.array([], dtype=float)
            )

            lower = (
                vmin
                if vmin is not None
                else (finite_values.min() if finite_values.size else None)
            )
            upper = (
                vmax
                if vmax is not None
                else (finite_values.max() if finite_values.size else None)
            )

            limits[name] = (lower, upper)
        else:
            limits[name] = (vmin, vmax)

    # Do not allocate empty columns when fewer panels than n_columns exist.
    n_panels = len(panels)
    n_cols = min(n_columns, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize * n_cols, figsize * n_rows),
        squeeze=False,
        layout="constrained",
    )
    flat_axes = axes.ravel()

    for panel, ax in zip(panels, flat_axes):
        name, region, region_mask, labels_arr, image = panel
        panel_vmin, panel_vmax = limits[name]

        im = ax.imshow(
            image,
            cmap=plot_cmap,
            vmin=panel_vmin,
            vmax=panel_vmax,
        )

        if outline:
            # Draw boundaries as a separate RGBA overlay because a masked
            # scalar image cannot carry an independent outline color.
            boundaries = find_boundaries(region_mask, mode="outer") & np.isin(
                region_mask, labels_arr
            )

            overlay = np.zeros(region_mask.shape + (4,), dtype=float)
            overlay[boundaries, :3] = mcolors.to_rgb(_style.edge_color_default)
            overlay[boundaries, 3] = 1.0

            ax.imshow(overlay)

        ax.set_title(f"{name} -- {region}")
        ax.set_axis_off()
        ax.set_aspect("equal")

        if scalebar:
            _add_scalebar(
                ax,
                image.shape[1],
                _style,
                scalebar_length_px=scalebar_length_px,
                um_per_px=um_per_px,
                show_text=scalebar_show_text,
            )

        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=_style.colorbar_fraction,
            pad=_style.colorbar_pad,
            label=(colorbar_label if colorbar_label is not None else name),
        )
        _style_colorbar(cbar, _style)

    # Remove rather than merely hide unused axes so they do not affect layout.
    for ax in flat_axes[n_panels:]:
        ax.remove()

    if savefig:
        _savefig(
            fig,
            output_dir,
            output_fname,
            "_expression_on_masks",
            _style,
        )

    # Show explicitly so the figure renders regardless of how the call sits in
    # the cell (trailing ';', assigned, in a loop). End the call with ';' or
    # assign the result to avoid a second render of the returned Figure.
    plt.show()

    return fig


def create_mask_dict(mask_file_paths, region_names):
    """
    Create a dictionary mapping region names to their corresponding masks.

    Parameters:
    mask_file_paths (list of str): List of file paths to the mask files.
    region_names (list of str): List of region names.

    Returns:
    dict: Dictionary where keys are region names and values are the corresponding masks.
    """
    if len(mask_file_paths) != len(region_names):
        raise ValueError(
            "The number of mask file paths must match the number of region names."
        )

    mask_dict = {}
    for file_path, region_name in zip(mask_file_paths, region_names):
        with open(file_path, "rb") as f:
            mask = pickle.load(f)
        mask_dict[region_name] = mask

    return mask_dict


def create_mask_dict_png(mask_file_paths, region_names):
    """
    Create a dictionary mapping region names to their corresponding masks.

    Parameters:
    mask_file_paths (list of str): List of file paths to the mask files.
    region_names (list of str): List of region names.

    Returns:
    dict: Dictionary where keys are region names and values are combined masks.
    """
    if len(mask_file_paths) != len(region_names):
        raise ValueError(
            "The number of mask file paths must match the number of region names."
        )

    mask_dict = {}
    for file_path, region_name in zip(mask_file_paths, region_names):
        # Read the image in grayscale mode
        mask_image = imread(file_path, as_gray=True)
        if mask_image is None:
            raise ValueError(f"Could not read the image file: {file_path}")

        # Rotate the image 90 degrees to the right
        mask_image = np.rot90(mask_image, k=3)

        # Mirror the image horizontally
        mask_image = np.fliplr(mask_image)

        # Convert the image to 8-bit grayscale
        mask_image = (mask_image * 255).astype(np.uint8)

        # Create a combined mask by assigning unique integer labels to each unique gray level, skipping black (0)
        combined_mask = np.zeros_like(mask_image, dtype=np.int32)
        unique_gray_levels = np.unique(mask_image)
        for i, gray_level in enumerate(unique_gray_levels, start=1):
            if gray_level == 0:
                continue  # Skip black (background)
            combined_mask[mask_image == gray_level] = i

        mask_dict[region_name] = combined_mask

    return mask_dict


def ppa_res_donut(
    adata,
    cat_col,
    key_name="ppa_result",
    palette=None,
    distance_mode="within",  # "within" or "between"
    unit="µm",
    figsize=(10, 10),
    add_guides=True,
    text="example CN",
    label_color="black",
    subset_column=None,
    subset_condition=None,
    group_by=None,  # New parameter to group by a column
    title="Title",
    legend="right",
    legend_ncol=3,
    text_width=20,
    label_angle_step=35,
    savefig=False,
    output_fname="",
    output_dir="./",
    style=None,
):
    """
    Plot donut chart of cell type proportions at different distances.
    Generates a multi-ring donut chart where each ring represents a specific
    distance range from the patches. The segments within each
    ring show the proportion of different categories (e.g., cell types)
    found within that distance range. The function can operate in two modes:
    'within' (cumulative proportions up to a distance) or 'between'
    (proportions within discrete distance intervals). It also supports
    subsetting the data and generating separate plots for different groups.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing the results in `adata.uns`.
    cat_col : str
        Column name in the results DataFrame (accessed via `key_name`)
        that contains the categorical data to plot (e.g., 'cell_type').
    key_name : str, optional
        Key in `adata.uns` where the DataFrame containing distance
        information (specifically 'distance_from_patch' column) is stored.
        Defaults to "ppa_result".
    palette : dict or None, optional
        A dictionary mapping category names (from `cat_col`) to colors -- any category
        not covered is colored grey, with a warning. If None, reuses/creates a persistent
        mapping cached in `adata.uns[cat_col + "_colors"]`, consistent with any other
        function (e.g. `catplot`) already plotting `cat_col`. Defaults to None.
    distance_mode : {'within', 'between'}, optional
        Determines how proportions are calculated for each distance ring:

        - "within": Includes all cells up to the specified distance. Rings
          represent cumulative proportions.
        - "between": Includes only cells between the current distance ring's
          outer radius and the previous ring's outer radius. Rings represent
          proportions in discrete intervals.

        Defaults to "within".
    unit : str, optional
        Unit for distance labels displayed on the plot (e.g., 'µm', 'px').
        Defaults to "µm".
    figsize : tuple, optional
        Size of the figure (width, height) in inches. Defaults to (10, 10).
    add_guides : bool, optional
        Whether to add radial and circular guide lines to the plot background.
        Defaults to True.
    text : str, optional
        Text to display in the center of the donut chart. Defaults to
        "example CN".
    label_color : str, optional
        Color of the distance labels placed on the rings. Defaults to "black".
    subset_column : str or None, optional
        Column name in the results DataFrame to use for subsetting the data
        before plotting. If None, no subsetting is performed based on this.
        Defaults to None.
    subset_condition : str or None, optional
        Value within `subset_column` to filter the data by. Only data points
        where `subset_column` equals `subset_condition` will be used.
        Requires `subset_column` to be set. Defaults to None.
    group_by : str or None, optional
        Column name in the results DataFrame to group the data by. If provided,
        a separate donut chart will be generated for each unique value in this
        column. The function calls itself recursively for each group.
        Defaults to None.
    title : str, optional
        Main title for the plot. If `group_by` is used, the group identifier
        will be appended to this title for each subplot. Defaults to "Title".
    legend : {"right", "below"}, optional
        Legend position. "below" puts it under the donut in ``legend_ncol`` columns, which keeps
        the donut large in narrow figures. Defaults to "right".
    legend_ncol : int, optional
        Number of legend columns when ``legend="below"``. Defaults to 3.
    label_angle_step : float, optional
        Angle in degrees between the distance labels of consecutive rings. The labels are spread
        along the rings (starting at 12 o'clock, going clockwise) so they never overlap, even in
        small donuts where the rings are only a few millimetres apart. 0 stacks them at the top.
        Defaults to 35.
    text_width : int, optional
        Maximum characters per line of the centre text; use a small value (e.g. 10) when the
        donut is small. Defaults to 20.
    savefig : bool, optional
        Whether to save the generated figure(s) to disk. If False, displays
        the plot instead. Defaults to False.
    output_fname : str, optional
        Base filename for the saved plot (without extension). If `group_by`
        is used, the group identifier will be appended. Defaults to "".
    output_dir : str, optional
        Directory where the plot(s) will be saved if `savefig` is True.
        Defaults to "./".

    Returns
    -------
    matplotlib.figure.Figure or dict
        If `group_by` is None, returns the single matplotlib Figure object
        for the plot.
        If `group_by` is provided, returns a dictionary where keys are the
        unique values from the `group_by` column and values are the
        corresponding matplotlib Figure objects.

    Raises
    ------
    ValueError
        If `key_name` is not found in `adata.uns`.
        If the DataFrame accessed by `key_name` is empty.
        If the 'distance_from_patch' column is missing in the DataFrame.
        If `cat_col` is missing in the DataFrame.
        If `subset_column` is provided but missing in the DataFrame.
        If `subset_condition` is provided but not found within `subset_column`.
        If `group_by` column is missing in the DataFrame.
        If no distance values are found in the 'distance_from_patch' column.
        If `distance_mode` is not 'within' or 'between'.

    Notes
    -----
    - The function assumes the results DataFrame stored in `adata.uns[key_name]`
      contains at least the columns specified by `cat_col` and
      'distance_from_patch'.
    - When `group_by` is used, the function makes recursive calls to itself,
      passing the subsetted data for each group. The `subset_column` and
      `subset_condition` parameters within the recursive call are used to
      filter for the specific group, and `group_by` is set to None to prevent
      infinite recursion.
    - The radii of the donut rings increase linearly with the sorted unique
      distance values. The innermost ring corresponds to the smallest distance,
      and the outermost ring corresponds to the largest distance.
    """
    import os
    import textwrap

    import matplotlib.pyplot as plt
    import numpy as np

    _style = _apply_style(style)

    # extract key from adata
    region_results = adata.uns[key_name]

    # check if region_results is empty
    if region_results.shape[0] == 0:
        print(f"Key {key_name} is empty.")
        return

    # check if distance_from_patch column exists
    if "distance_from_patch" not in region_results.columns:
        raise ValueError("distance_from_patch column not found in the dataframe")

    # Check if cat_col exists in region_results
    if cat_col not in region_results.columns:
        raise ValueError(f"Column '{cat_col}' not found in region_results")

    categories = sorted(adata.obs[cat_col].dropna().unique().tolist())
    palette = _resolve_categorical_palette(
        _style, adata, cat_col, categories, palette=palette
    )

    # subset by condition if specified
    if subset_column is not None:
        if subset_column not in region_results.columns:
            raise ValueError(
                f"Column '{subset_column}' does not exist in the DataFrame."
            )
        elif subset_condition not in region_results[subset_column].unique():
            raise ValueError(
                f"Value '{subset_condition}' does not exist in the column '{subset_column}'."
            )
        else:
            region_results = region_results[
                region_results[subset_column] == subset_condition
            ]

    # If group_by is specified, create separate visualizations for each group
    if group_by is not None:
        if group_by not in region_results.columns:
            raise ValueError(
                f"Group by column '{group_by}' not found in region_results"
            )

        # Get unique values in the group_by column
        group_values = region_results[group_by].unique()
        figs = {}

        # Create a figure for each group
        for group_val in group_values:
            print(f"Creating visualization for {group_by} = {group_val}")
            # Filter data for this group
            group_data = region_results[region_results[group_by] == group_val]

            # Skip groups with no data
            if len(group_data) == 0:
                print(f"No data for {group_by} = {group_val}, skipping")
                continue

            # Custom title including the group value
            group_title = f"{title} - {group_by}: {group_val}"

            # Custom filename if saving
            group_filename = (
                f"{output_fname}_{group_by}_{group_val}"
                if output_fname
                else f"{group_by}_{group_val}"
            )

            # Call this function recursively without group_by to avoid infinite recursion
            fig = ppa_res_donut(
                adata=adata,
                cat_col=cat_col,
                key_name=key_name,
                palette=palette,
                distance_mode=distance_mode,
                unit=unit,
                figsize=figsize,
                add_guides=add_guides,
                text=text,
                label_color=label_color,
                subset_column=group_by,  # Use group_by as the subset column
                subset_condition=group_val,  # Use group_val as the subset condition
                group_by=None,  # Don't group again to avoid recursion
                title=group_title,
                legend=legend,
                legend_ncol=legend_ncol,
                text_width=text_width,
                label_angle_step=label_angle_step,
                savefig=savefig,
                output_fname=group_filename,
                output_dir=output_dir,
                style=_style,  # pass resolved style to avoid re-resolution in recursion
            )

            figs[group_val] = fig

        return figs

    # Extract unique distances from the dataframe
    available_distances = sorted(region_results["distance_from_patch"].unique())
    if len(available_distances) == 0:
        raise ValueError("No distance values found in distance_from_patch column")

    # plotting
    fig, ax = plt.subplots(figsize=figsize)

    if add_guides:
        # add grid lines
        plt.plot([0, 0], [1.05, -1.05], color="black", alpha=0.3, zorder=-1)
        plt.plot([1.05, -1.05], [0, 0], color="black", alpha=0.3, zorder=-1)
        # add diagonal lines
        for angle in [45, -45]:
            x_new = 1.05 * np.cos(np.radians(angle))
            y_new = 1.05 * np.sin(np.radians(angle))
            plt.plot(
                [-x_new, x_new], [-y_new, y_new], color="black", alpha=0.3, zorder=-1
            )

    # Collect all categories to ensure consistent wedge ordering
    # largest category first, same order in every ring and in the legend
    all_categories = region_results[cat_col].value_counts().index

    # Process each distance - START FROM SMALLEST (INNERMOST) TO LARGEST (OUTERMOST)
    for i, radius in enumerate(available_distances):  # Not reversed anymore
        # Filter data based on distance_mode
        if distance_mode == "within":
            # Include all cells within the radius
            filtered_results = region_results[
                region_results["distance_from_patch"] <= radius
            ]
        elif distance_mode == "between":
            # Include only cells between this radius and the next smaller one
            if i == 0:  # Smallest radius (innermost ring)
                prev_radius = 0
            else:
                prev_radius = available_distances[i - 1]  # Previous (smaller) radius

            filtered_results = region_results[
                (region_results["distance_from_patch"] <= radius)
                & (region_results["distance_from_patch"] > prev_radius)
            ]
        else:
            raise ValueError("distance_mode must be 'within' or 'between'")

        print(f"Radius {radius}: {filtered_results.shape[0]} cells")

        if filtered_results.shape[0] > 0:
            # Calculate percentages for this specific radius
            percentage_dict = (
                filtered_results[cat_col].value_counts(normalize=True) * 100
            )

            # Create list of values and colors in consistent order
            wedge_values = []
            wedge_colors = []

            # For each possible category, get its percentage (or 0 if not present)
            for category in all_categories:
                if category in percentage_dict.index:
                    wedge_values.append(percentage_dict[category])
                    if category not in palette:
                        raise ValueError(
                            f"No color provided for category {category} in the palette"
                        )
                    wedge_colors.append(palette[category])

            # Only draw if there are values to plot
            if wedge_values:
                # Radius increases with distance - smallest at center, largest at outside
                ring_radius = 0.5 + 0.1 * (i + 1)

                # Draw this ring's pie
                ax.pie(
                    wedge_values,
                    radius=ring_radius,
                    colors=wedge_colors,
                    startangle=90,  # Start at top
                    wedgeprops=dict(width=0.1, edgecolor="w"),  # Make it a ring
                )

    # add labels for each distance ring
    for j, radius in enumerate(available_distances):  # Not reversed anymore
        r_lab = 0.45 + 0.1 * (j + 1)  # label position corresponds to the ring radius
        ang = np.deg2rad(90 - label_angle_step * j)
        plt.text(
            r_lab * np.cos(ang),
            r_lab * np.sin(ang),
            str(radius) + unit,
            horizontalalignment="center",
            verticalalignment="center",
            fontweight="bold",
            fontsize=_style.font_size_tiny,
            color=label_color,
            # white box keeps the label readable on top of the wedges
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            zorder=10,
        )

    # add a circle at the center to transform it in a donut chart
    my_circle = plt.Circle((0, 0), 0.45, color="white")  # Smaller center circle
    p = plt.gcf()
    p.gca().add_artist(my_circle)

    # add a legend based on the colors and keys in palette
    # Only include categories that are present in the data
    present_categories = all_categories
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[key])
        for key in present_categories
        if key in palette
    ]

    if legend == "below":
        plt.legend(
            handles,
            [cat for cat in present_categories if cat in palette],
            bbox_to_anchor=(0.5, -0.02),
            loc="upper center",
            ncol=legend_ncol,
            prop={"size": _style.font_size_small},
            frameon=_style.legend_frameon,
            columnspacing=1.0,
            handlelength=1.0,
            labelspacing=0.3,
        )
    else:
        plt.legend(
            handles,
            [cat for cat in present_categories if cat in palette],
            bbox_to_anchor=(0.94, 0.925),
            loc="upper left",
            prop={"size": _style.font_size_small},
            frameon=_style.legend_frameon,
        )

    # Define the maximum length of a line
    max_line_length = text_width

    # Split the text into multiple lines if it's too long
    wrapped_text = textwrap.fill(text, max_line_length)

    # Add a title in the middle of the white circle
    plt.text(
        0,
        0,
        wrapped_text,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=_style.font_size_medium,
    )

    plt.title(title, size=_style.font_size_large, y=0.96)
    plt.axis("equal")  # Equal aspect ratio ensures circle looks round

    if savefig is None:
        pass
    elif savefig:
        _savefig(None, output_dir, output_fname, "", _style)

    plt.show()

    return fig


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
    Deprecated: moved to the preprocessing module and renamed. Use
    ``sp.pp.marker_score_thres`` instead (which takes ``marker_list`` rather than
    ``col_num``). Kept so code written for SPACEc 0.1.2 keeps running.

    Parameters and return value are those of ``sp.pp.zcount_thres``: the
    noise-removal thresholds as a dict, or ``(thresholds, fig)`` when
    ``qc_dashboard=True`` (the default).
    """
    from ..preprocessing._general import zcount_thres as _pp_zcount_thres

    warnings.warn(
        "'sp.pl.zcount_thres' is deprecated and will be removed in a future version. "
        "Use 'sp.pp.marker_score_thres' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # sp.pp.zcount_thres raises its own deprecation warnings (for the function and for
    # col_num); silence those so the caller sees the single message above.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="'zcount_thres' is deprecated",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore", message="'col_num' is deprecated", category=DeprecationWarning
        )
        return _pp_zcount_thres(
            dfz,
            col_num,
            cut_off=cut_off,
            count_bin=count_bin,
            zsum_bin=zsum_bin,
            figsize=figsize,
            qc_dashboard=qc_dashboard,
            flag_cut_off=flag_cut_off,
            style=style,
            positive_thres=positive_thres,
        )
