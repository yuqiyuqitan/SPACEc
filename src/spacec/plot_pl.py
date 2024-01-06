# load required packages
import os as os
import skimage
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import tensorly as tl
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tensorly.decomposition import non_negative_tucker

from .helperfunctions_hf import *

# Setup
sns.set_style("ticks")


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

    ##### Can subset it here if I do not want normalized per the group
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
            alpha=0.8,
            linewidth=1,
            figsize=fig_sizing,
            rot=90,
            stacked=True,
            edgecolor="black",
        )

    else:
        # first subplot
        ax1 = melt_test_piv.plot.bar(
            alpha=0.8,
            linewidth=1,
            color=[color_dic.get(x) for x in melt_test_piv.columns],
            figsize=fig_sizing,
            rot=90,
            stacked=True,
            edgecolor="black",
        )

    for line in ax1.lines:
        line.set_color("black")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
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
        plt.savefig(
            output_dir + save_name + "_stacked_barplot.pdf",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )
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
):
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
    else:
        # Order by average
        plot_order = (
            melt_per_plot.groupby(per_cat)
            .mean()
            .reset_index()
            .sort_values(by="percentage")[per_cat]
            .to_list()
        )

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
                edgecolor="black",
                linewidth=1,
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
                edgecolor="black",
                linewidth=1,
                palette=color_dic,
            )

        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.3))
        plt.xticks(rotation=90)
        plt.xlabel("")
        plt.ylabel("")
        plt.title(sub_list[0])
        sns.despine()

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
                edgecolor="black",
                linewidth=1,
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
                edgecolor="black",
                linewidth=1,
                palette=color_dic,
            )
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.3))
        # ax.set_yscale(\log\)
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
        sns.despine()

    if output_dir:
        if save_name:
            plt.savefig(
                output_dir + save_name + "_swarm_boxplot.pdf",
                dpi=300,
                transparent=True,
                bbox_inches="tight",
            )
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
            edgecolor="black",
            linewidth=1,
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
            edgecolor="black",
            linewidth=1,
            palette=color_dic,
        )

    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.3))
    plt.xticks(rotation=90)
    plt.xlabel("")
    plt.ylabel("Shannon Diversity")
    plt.title("")
    sns.despine()
    if save == True:
        plt.savefig(
            output_dir + sub_list[0] + "_Shannon.pdf",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

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
            plt.savefig(
                output_dir + sub_list[0] + "_tukey.png",
                format="png",
                dpi=300,
                transparent=True,
                bbox_inches="tight",
            )
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
    plt.savefig(output_dir + "/cell_types_composition_hstack.png", bbox_inches="tight")

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
    plt.savefig(
        output_dir + "/cell_types_composition_hUNstack.png", bbox_inches="tight"
    )

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
    plt.savefig(
        output_dir + "/cell_types_composition_perc_hstack.png", bbox_inches="tight"
    )


##############


def pl_regions_per_sample(data, sample_col, region_col, bar_color="grey"):
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
            f.savefig(output_dir + "neighborhood_" + str(k) + "_id{}.png".format(n))

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
        fc, vmin=-3, vmax=3, cmap="bwr", row_colors=sns.color_palette(palette, len(fc))
    )
    s.savefig(output_dir + "celltypes_perniche_" + "_" + str(k) + ".png", dpi=600)

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
            fc.iloc[plot_specific_neighborhoods, :], vmin=-3, vmax=3, cmap="bwr"
        )
        s.savefig(output_dir + "celltypes_perniche_" + "_" + str(k) + ".png", dpi=600)


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

            # Set the alpha value for red dots
            alpha = 0.7 if group == highlight_group else 1.0

            # Plot the dots
            ax.scatter(x, y, color=color, label=group, alpha=alpha, zorder=2, s=1)

        # Remove legend
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


def pl_cell_types_de(data, pvals, neigh_num, output_dir, figsize=(20, 10)):
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
    # GENERAL GRAPH SETTINGs
    # font size of graph
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    # Settings for graph
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

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

    f.savefig(
        output_dir + "tissue_neighborhood_coeff_pvalue_bar.png",
        format="png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )

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
            f.savefig(output_dir + "community_" + str(k) + "_id{}.png".format(n))

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
        fc, vmin=-3, vmax=3, cmap="bwr", row_colors=sns.color_palette(palette, len(fc))
    )
    s.savefig(output_dir + "celltypes_perniche_" + "_" + str(k) + ".png", dpi=600)

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
            fc.iloc[plot_specific_neighborhoods, :], vmin=-3, vmax=3, cmap="bwr"
        )
        s.savefig(output_dir + "celltypes_perniche_" + "_" + str(k) + ".png", dpi=600)


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

    if colors != None:
        pal = colors
    else:
        pal = sns.color_palette("bright", 50)

    pos = nx.nx_agraph.graphviz_layout(g1, prog="neato")
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
        plt.savefig(
            output_dir + "/" + save_name,
            format="png",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )


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
            plt.savefig(
                save_path + save_name + "_" + str(p) + "_tensor.png",
                format="png",
                dpi=300,
                transparent=True,
                bbox_inches="tight",
            )

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
        plt.savefig(
            output_dir + save_name + ".png",
            format="png",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

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


def pl_cor_subplot(mp, sub_list, output_dir, save_name=None):
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
        plt.savefig(
            output_dir + save_name + "_corrplot.png",
            format="png",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )


def annotate(data, names, **kws):
    r, p = sp.stats.pearsonr(data[names[0]], data[names[1]])
    ax = plt.gca()
    ax.text(
        0.5, 0.8, "r={:.2f}, p={:.2g}".format(r, p), transform=ax.transAxes, fontsize=14
    )


def pl_cor_subplot_new(mp, sub_list, output_dir, save_name=None):
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
        plt.savefig(
            output_dir + save_name + "_corrplot.png",
            format="png",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )


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

    # Settings for graph
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
        plt.savefig(
            output_dir + save_name + ".png",
            format="png",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )


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
    pos = nx.drawing.nx_pydot.graphviz_layout(draw, prog="dot")
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
    # ax.set_yscale(\log\)
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
    style="white",
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
    scatter_kws_ = {"s": size, "alpha": 1}
    scatter_kws_.update(scatter_kws)

    figures = []
    df = df.rename(columns=lambda x: str(x))

    df[hue] = df[hue].astype("category")
    if invert_y:
        y_orig = df[Y].values.copy()
        df[Y] *= -1

    style = {"axes.facecolor": style}
    sns.set_style(style)
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

        plt.xticks(fontsize=ticks_fontsize)  # Increase x-axis label size
        plt.yticks(fontsize=ticks_fontsize)  # Increase y-axis label size

        plt.show()
        figures += [f]

    if invert_y:
        df[Y] = y_orig

    return figures


##########


def pl_comb_num_freq(data_list, plot_order=None, pal_tis=None, figsize=(5, 5)):
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
    # ax.set_yscale(\log\)
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
def pl_zcount_thres(
    dfz, col_num, cut_off=0.01, count_bin=50, zsum_bin=50, figsize=(10, 5)
):
    dfz_copy = dfz
    dfz_copy["Count"] = dfz.iloc[:, : col_num + 1].ge(0).sum(axis=1)
    dfz_copy["z_sum"] = dfz.iloc[:, : col_num + 1].sum(axis=1)
    fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=figsize)
    axes[0].hist(dfz_copy["Count"], bins=count_bin)
    axes[0].set_title("Count")
    axes[0].axvline(
        dfz_copy["Count"].quantile(1 - cut_off),
        color="k",
        linestyle="dashed",
        linewidth=1,
    )
    axes[0].text(
        0.75,
        0.75,
        "Cut off: {:.2f}".format(dfz_copy["Count"].quantile(1 - cut_off)),
        ha="right",
        va="bottom",
        transform=axes[0].transAxes,
    )
    axes[1].hist(dfz_copy["z_sum"], bins=zsum_bin)
    axes[1].title.set_text("Zscore sum")
    axes[1].axvline(
        dfz_copy["z_sum"].quantile(1 - cut_off),
        color="k",
        linestyle="dashed",
        linewidth=1,
    )
    axes[1].text(
        0.75,
        0.75,
        "Cut off: {:.2f}".format(dfz_copy["z_sum"].quantile(1 - cut_off)),
        ha="right",
        va="bottom",
        transform=axes[1].transAxes,
    )


##########


def pl_mono_cluster_spatial(
    df,
    sample_col="Sample",
    cluster_col="Cell Type",
    x="x",
    y="y",
    color_dict=None,
    s=3,
    alpha=0.5,
    figsize=(15, 12),
):
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
):
    # Subset the DataFrame based on region_column and selected_region
    subset_df1 = df[df[region_column] == selected_region]

    # Subset the DataFrame based on subsetting_column and values_list
    subset_df2 = subset_df1[subset_df[subsetting_column].isin(values_list)]

    # Create a 2D density plot
    sns.kdeplot(data=subset_df2, x=x_column, y=y_column, fill=True)

    # Overlay the individual data points as a scatter plot
    sns.scatterplot(
        data=subset_df1, x=x_column, y=y_column, color="lightgrey", alpha=0.5
    )

    # Add labels and title to the plot
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title("2D Density Plot with Overlay")

    # Display the plot
    plt.show()


#######


def pl_create_cluster_celltype_heatmap(dataframe, cluster_column, celltype_column):
    # Create a frequency table using pandas crosstab
    frequency_table = pd.crosstab(dataframe[cluster_column], dataframe[celltype_column])

    # Create the heatmap using seaborn
    plt.figure(figsize=(20, 6))  # Set the size of the heatmap (adjust as needed)
    sns.heatmap(
        frequency_table, cmap="YlGnBu", annot=True, fmt="d"
    )  # cmap sets the color palette

    plt.title("Cluster-Cell Type Heatmap")
    plt.xlabel("Cell Types")
    plt.ylabel("Cluster IDs")
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def pl_catplot_ad(
    adata,
    color,
    unique_region,
    subset=None,
    X="x",
    Y="y",
    invert_y=False,
    size=6,
    alpha=1,
    palette=None, #default is None which means the color comes from the anndata object
    savefig=False,
    output_dir="./",
    output_fname = "",
    figsize=5,
    style="white",
    axis="on",
    scatter_kws={},
    n_columns=4,
    legend_padding=0.2,
    rand_seed = 1
):
    """
    Plots cells in tissue section color coded by either cell type or node allocation.
    adata: anndata containing information
    size: size of point to plot for each cell.
    color: color by "Clusterid" or "Node" respectively.
    unique_region: each region is one independent CODEX image
    legend: to include legend in plot.
    """
    scatter_kws_ = {"s": size, "alpha": alpha}
    scatter_kws_.update(scatter_kws)

    df = pd.DataFrame(adata.obs[[X, Y, color, unique_region]])

    df[color] = df[color].astype("category")
    if invert_y:
        y_orig = df[Y].values.copy()
        df[Y] *= -1
    
    if palette is None:
        if color + '_colors' not in adata.uns.keys():
            ct_colors = hf_generate_random_colors(
                len(adata.obs[color].unique()), 
                rand_seed = rand_seed)
            palette = dict(zip(np.sort(adata.obs[color].unique()), ct_colors))
            adata.uns[color + "_colors"] = ct_colors
        else:
            palette = dict(zip(
                np.sort(adata.obs[color].unique()), 
                adata.uns[color + '_colors']))   

    style = {"axes.facecolor": style}
    sns.set_style(style)
    if subset is None:
        region_list = list(
            df[unique_region].unique().sort_values()
        )  # display all experiments
    else:
        if subset not in list(df[unique_region].unique().sort_values()):
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
        gridspec_kw={"wspace": 1.1, "hspace": 0.4},
    )

    for i_ax, (name, ax) in enumerate(zip(region_list, axes.flatten())):
        data = df[df[unique_region] == name]
        # print(name)
        sns.scatterplot(
            x=X, y=Y, data=data, hue=color, palette=palette, ax=ax, s=size, alpha=alpha
        )
        ax.grid(False)
        if axis == "off":
            ax.axis("off")

        ax.set_title(name)
        ax.set_aspect("equal")

        # Add padding to the legend
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        # frame = legend.get_frame()
        # frame.set_facecolor('white')  # Adjust the legend background color

    for i in range(i_ax + 1, n_rows * n_columns):
        axes.flatten()[i].axis("off")

    # fig.tight_layout(pad = 0.5)

    if savefig:
        fig.savefig(output_dir + output_fname +"_spatial_plot.pdf", bbox_inches="tight")
    # else:
    #    return fig



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
):
    draw = graph
    pos = nx.drawing.nx_pydot.graphviz_layout(draw, prog="dot")
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
        plt.savefig(output_dir + "_CNMap.pdf", bbox_inches="tight")
    else:
        plt.show()


def pl_stacked_bar_plot_ad(
    adata,
    color,
    grouping,
    cell_list,
    output_dir,
    norm=True,
    savefig=False, # new
    output_fname = "", # new
    col_order=None,
    sub_col=None,
    name_cat="celltype",
    fig_sizing=(8, 4),
    plot_order=None,
    palette=None,
    remove_leg=False,
    rand_seed = 1
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
            sub_cell_list = list(
                data.loc[data[color].isin(cell_list)][color].unique()
            )
        else:
            test1 = data.copy()
            sub_cell_list = list(
                data.loc[data[sub_col].isin(cell_list)][color].unique()
            )
    if palette is None:
        if color + '_colors' not in adata.uns.keys():
            ct_colors = hf_generate_random_colors(len(adata.obs[color].unique()), rand_seed = rand_seed)
            palette = dict(zip(np.sort(adata.obs[color].unique()), ct_colors))
            adata.uns[color + "_colors"] = ct_colors
        else:
            palette = dict(zip(np.sort(adata.obs[color].unique()), adata.uns[color + '_colors']))   

    test1[color] = test1[color].astype("category")
    test_freq = test1.groupby(grouping).apply(
        lambda x: x[color].value_counts(normalize=True, sort=False) * 100
    )
    test_freq.columns = test_freq.columns.astype(str)

    ##### Can subset it here if I do not want normalized per the group
    test_freq.reset_index(inplace=True)
    sub_cell_list.append(grouping)
    test_freq = test_freq[sub_cell_list]
    melt_test = pd.melt(
        test_freq, id_vars=[grouping]
    )  # , value_vars=test_freq.columns)
    #melt_test.rename(columns={per_cat: name_cat, "value": "percent"}, inplace=True)
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
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
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
    if savefig:
        plt.savefig(
            output_dir + output_fname + ".pdf",
            format="pdf",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )
    else:
        plt.show()
    return melt_test_piv, plot_order


def pl_swarm_box_ad(
    adata,
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
):
    # extract information form adata
    data = adata.obs

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
    else:
        # Order by average
        plot_order = (
            melt_per_plot.groupby(per_cat)
            .mean()
            .reset_index()
            .sort_values(by="percentage")[per_cat]
            .to_list()
        )

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
                edgecolor="black",
                linewidth=1,
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
                edgecolor="black",
                linewidth=1,
                palette=color_dic,
            )

        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.3))
        plt.xticks(rotation=90)
        plt.xlabel("")
        plt.ylabel("")
        plt.title(sub_list[0])
        sns.despine()

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
                edgecolor="black",
                linewidth=1,
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
                edgecolor="black",
                linewidth=1,
                palette=color_dic,
            )
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.3))
        # ax.set_yscale(\log\)
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
        sns.despine()

    if output_dir:
        if save_name:
            plt.savefig(
                output_dir + save_name + "_swarm_boxplot.png",
                format="png",
                dpi=300,
                transparent=True,
                bbox_inches="tight",
            )
        else:
            print("define save_name")
    else:
        print("plot was not saved - to save the plot specify an output directory")
    return melt_per_plot


def pl_create_pie_charts_ad(
    adata,
    color,
    grouping,
    plot_order=None,
    show_percentages=True,
    palette=None,
    savefig=False,
    output_fname = "",
    output_dir = './',
    rand_seed = 1

):
    """
    Create pie charts for each group based on a grouping column, showing the percentage of total rows based on a
    count column.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        grouping (str): The column name for grouping the data.
        color (str): The column name used for counting occurrences.
        plot_order (list, optional): The order of groups for plotting. Defaults to None.
        show_percentages (bool, optional): Whether to show the percentage numbers on the pie charts. Defaults to True.
        palette (dict, optional): A dictionary to manually set colors for neighborhoods. Defaults to None.

    Returns:
        None
    """
    data = adata.obs

    # Group the data by the grouping column
    grouped_data = data.groupby(grouping)

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
    if palette is None:
        if color + '_colors' not in adata.uns.keys():
            ct_colors = hf_generate_random_colors(len(adata.obs[color].unique()), rand_seed = rand_seed)
            palette = dict(zip(np.sort(adata.obs[color].unique()), ct_colors))
            adata.uns[color + "_colors"] = ct_colors
        else:
            palette = dict(zip(np.sort(adata.obs[color].unique()), adata.uns[color + '_colors']))   

    # Iterate over each group and create a pie chart
    for i, (group, group_df) in enumerate(grouped_data):
        # Count the occurrences of each neighborhood within the group
        neighborhood_counts = group_df[color].value_counts()

        # Calculate the percentage of total rows for each neighborhood
        percentages = neighborhood_counts / group_df.shape[0] * 100

        # Create a color list for neighborhoods in the count column
        colors = [
            palette.get(neighborhood, "gray") for neighborhood in percentages.index
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
    if savefig:
        plt.savefig(
            output_dir + output_fname + "_piechart.pdf",
            format="pdf",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )
    else:
        #Show the plot
        plt.show()


def pl_CN_exp_heatmap_ad(adata, 
                         cluster_col, 
                         cn_col, 
                         palette=None, 
                         figsize = (18,12),
                         savefig=False,
                         output_fname = "",
                         output_dir = './',
                         row_clus = True,
                         col_clus = True,
                         rand_seed = 1
                        ):
    
    data = adata.obs
   
    if palette is None:
        if cn_col + '_colors' not in adata.uns.keys():
            # Create a color dictionary if not provided 
            cn_colors = hf_generate_random_colors(len(adata.obs[cn_col].unique()), rand_seed = rand_seed)
            palette = dict(zip(np.sort(adata.obs[cn_col].unique()), cn_colors))
            adata.uns[cn_col + "_colors"] = cn_colors
        else:
            palette = dict(zip(np.sort(adata.obs[cn_col].unique()), adata.uns[cn_col + '_colors']))   

    neigh_data = pd.DataFrame({
    cn_col:list(palette.keys()),
    'color':list(palette.values())
    })
    neigh_data.set_index(keys=cn_col,inplace=True)
    
    df3 = pd.concat([data,pd.get_dummies(data[cluster_col])],axis=1)
    sum_cols2 = df3[cluster_col].unique()
    values2 = df3[sum_cols2].values
    cell_list = sum_cols2.copy()
    cell_list = cell_list.tolist()
    cell_list.append(cn_col)

    subset = df3[cell_list]
    niche_sub = subset.groupby(cn_col).sum()
    niche_df = niche_sub.apply(lambda x: x/x.sum() * 10, axis=1)
    neigh_clusters = niche_df.to_numpy()

 
    tissue_avgs = values2.mean(axis = 0)
    
        
    fc_2 = np.log2(((neigh_clusters+tissue_avgs)/(neigh_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
    fc_2 = pd.DataFrame(fc_2,columns = sum_cols2)
    fc_2.set_index(niche_df.index, inplace=True)
    s=sns.clustermap(fc_2, vmin =-3,vmax = 3,cmap = 'bwr', figsize=figsize, row_colors=[neigh_data.reindex(fc_2.index)['color']],\
                    cbar_pos=(0.03,0.06,0.03,0.1),
                    )

    s.ax_row_dendrogram.set_visible(row_clus)
    s.ax_col_dendrogram.set_visible(col_clus)
    s.ax_heatmap.set_ylabel("", labelpad=25)
    s.ax_heatmap.tick_params(axis='y', pad=42)
    s.ax_heatmap.yaxis.set_ticks_position("right")
    
    if savefig:
        s.figure.savefig(
            output_dir + output_fname + "_cn_heatmap.pdf"
        )


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
    # Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
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


def pl_plot_scatter_correlation(data, x, y, xlabel=None, ylabel=None, save_path=None):
    g = sns.lmplot(x=x, y=y, data=data, height=5, aspect=1)
    g.map_dataframe(hf_annotate_cor_plot, x=x, y=y, data=data)

    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if save_path:
        plt.savefig(save_path + "_corrplot.pdf", transparent=True, dpi=600, bbox_inches="tight")
    plt.show()


def pl_plot_scatter_correlation_ad(
    adata, x, y, xlabel=None, ylabel=None, save_path=None
):
    data = adata.obs

    g = sns.lmplot(x=x, y=y, data=data, height=5, aspect=1)
    g.map_dataframe(hf_annotate_cor_plot, x=x, y=y, data=data)

    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if save_path:
        plt.savefig(save_path + "_corrplot.pdf", transparent=True, dpi=600, bbox_inches="tight")
    plt.show()


########

def pl_plot_correlation_matrix(cmat):
    # plot correlation matrix as heatmap
    # Create a mask to hide the upper triangle
    mask = np.triu(np.ones_like(cmat, dtype=bool))

    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(
        cmat, annot=True, fmt=".2f", cmap="coolwarm", square=True, mask=mask, ax=ax
    )
    plt.show()


def pl_dumbbell(data, figsize=(10,10), colors = ['#DB444B', '#006BA2']):
    fig, ax = plt.subplots(figsize=figsize, facecolor = "white")
    #plot each country one at a time

    # Create grid 
    # Zorder tells it which layer to put it on. We are setting this to 1 and our data to 2 so the grid is behind the data.
    ax.grid(which="major", axis='both', color='#758D99', alpha=0.6, zorder=1)

    # Remove splines. Can be done one at a time or can slice with a list.
    ax.spines[['top','right','bottom']].set_visible(False)

    # Plot data
    comp_cat = data.columns
    
    # Plot horizontal lines first
    ax.hlines(y=data.index, xmin=data[comp_cat[0]], xmax=data[comp_cat[1]], color='#758D99', zorder=2, linewidth=2, label='_nolegend_', alpha=.8)
    # Plot bubbles next
    ax.scatter(data[comp_cat[0]], data.index, label='2014', s=60, color=colors[0], zorder=3)
    ax.scatter(data[comp_cat[1]], data.index, label='2018', s=60, color=colors[1], zorder=3)

    # Set xlim
    #ax.set_xlim(-3, 3)

    # Reformat x-axis tick labels

    ax.xaxis.set_tick_params(labeltop=True,      # Put x-axis labels on top
                            labelbottom=False,  # Set no x-axis labels on bottom
                            bottom=False,       # Set no ticks on bottom
                            labelsize=11,       # Set tick label size
                            pad=-1)             # Lower tick labels a bit

    ax.axvline(x=0, color='k', linestyle='--')

    # Set Legend
    ax.legend(data.columns, loc=(0,1.076), ncol=2, frameon=False, handletextpad=-.1, handleheight=1) 



def pl_CNmap(adata,
    cnmap_dict,
    cn_col,
    palette = None,
    figsize=(40, 20),
    savefig=False,
    output_fname = "",
    output_dir="./",
    rand_seed = 1
):
    
    graph = cnmap_dict['g']
    tops = cnmap_dict['tops']
    e0 = cnmap_dict['e0']
    e1 = cnmap_dict['e1']
    simp_freqs = cnmap_dict['simp_freqs']
    draw = graph
    pos = nx.drawing.nx_pydot.graphviz_layout(draw, prog="dot")
    height = 8

    # generate color
    cn_colors = hf_generate_random_colors(len(adata.obs[cn_col].unique()), rand_seed = rand_seed)
    if palette is None:
        if cn_col + '_colors' not in adata.uns.keys():
            palette = dict(zip(np.sort(adata.obs[cn_col].unique()), cn_colors))
            adata.uns[cn_col + "_colors"] = cn_colors
        else:
            palette = dict(zip(np.sort(adata.obs[cn_col].unique()), adata.uns[cn_col + '_colors']))  

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
        plt.savefig(output_dir + output_fname + "_CNMap.pdf", bbox_inches="tight")
    else:
        plt.show()


def pl_coordinates_on_image(df, 
                            overlay_data, 
                            color= None, 
                            x ='x', y = 'y',
                            fig_width=20, fig_height=20, dot_size = 10, 
                            convert_to_grey=True, 
                            scale=False,
                            cmap='inferno',
                            savefig = False,
                            output_dir = "./",
                            output_fname = ""):
    # Create a new figure with increased size
    plt.figure(figsize=(fig_width, fig_height))
    
    if convert_to_grey:
        # Convert the image to grayscale
        overlay_data = skimage.color.rgb2gray(overlay_data)
        plt.imshow(overlay_data, cmap='gray')
    
    else:
        plt.imshow(overlay_data)
    
    image_width, image_height = overlay_data.shape[1], overlay_data.shape[0]
    
    
    # Plot the coordinates on top of the image
    # colorscale by area
    
    if color != None:
        
        if scale:
            # minmax scale the variable
            df[color] = (df[color] - df[color].min())/(df[color].max() - df[color].min())
            # change dot size based on variable
            plt.scatter(df[x], df[y], s=df[color]*30, c=df[color], cmap=cmap)
        else:
            plt.scatter(df[x], df[y], c=df[color], s=dot_size, cmap=cmap)   
            
    else:
        plt.scatter(df[x], df['y'], s=dot_size)
    
    # add colorbar
    plt.colorbar()
    
    
    # set axis limits
    plt.xlim(0, image_width)
    plt.ylim(image_height, 0)

    # Show the plot
    if savefig:
        plt.savefig(output_dir + output_fname +"_seg_masks_overlay.pdf", bbox_inches="tight")
    else:
        plt.show()


def pl_count_patch_proximity_res(adata, 
                           x, 
                           hue,
                           palette="Set3",
                           order = True,
                           key_name = 'ppa_result',
                           savefig = False,
                           output_dir = "./",
                           output_fname = ""
                          ):
    
    region_results = adata.uns[key_name]
    ax = sns.countplot(x=x, 
                   hue=hue, 
                   data=region_results, 
                   palette=palette, 
                   order=region_results[x].value_counts().index)
    
    tick_positions = range(len(region_results[x].value_counts()))
    tick_labels = region_results[x].value_counts().index
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90)
    if savefig:
        plt.savefig(output_dir + output_fname +"_count_ppa_result.pdf", bbox_inches="tight")
    else:
        plt.show()
