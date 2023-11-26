# load required packages
import glob
import os
from itertools import product
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import norm, zscore

# read the data frame output from the segmentation functions
def pp_read_segdf(segfile_list,
                seg_method, 
               region_list=None, #optional information #please make sure the length of each list matches
               meta_list=None #optional information
              ):
    if region_list is not None:
        if len(region_list) != len(segfile_list):
            sys.exit("length of each list does not match!")
    elif meta_list is not None:
        if len(meta_list) != len(segfile_list):
            sys.exit("length of each list does not match!")
    
    df = pd.DataFrame()
    #concat old dataframes
    for i in range(len(segfile_list)):
        tmp = pd.read_csv(segfile_list[i], index_col = 0)
        tmp['region_num'] = str(i)
        if region_list is not None:
            tmp['unique_region'] = str(region_list[i])
        if meta_list is not None:
             tmp['condition'] = str(meta_list[i])
        df = pd.concat([df, tmp], axis = 0)

    if(seg_method == 'cellseg'):
        #See resultant dataframe
        df.columns = df.columns.str.split(':').str[-1].tolist()
        df = df.reset_index().rename(columns={'index':'first_index'})
        df.columns = df.columns.str.split(':').str[-1].tolist()
        df.rename(columns={'size': 'area'}, inplace=True)
    return df


def pp_filter_data(df, 
                   nuc_thres=1,
                   size_thres=1,
                   nuc_marker="DAPI",
                   cell_size = "area",
                   region_column = "region_num",
                   color_by = None,
                   palette = "Paired",
                   alpha=0.8, size=0.4, # dot style
                   log_scale=False):
    
    if color_by == None:
        color_by = region_column
    
    df_nuc = df[
        (df[nuc_marker] > nuc_thres) * df[cell_size] > size_thres
    ]
    per_keep = len(df_nuc) / len(df)
    
    
    
    
    # Create a figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Boxplot
    sns.boxplot(data=df.loc[:, [cell_size, nuc_marker]],
                orient='h',
                ax=ax1)
    ax1.set_title("Cell size and nuclear marker intensity")
    
    

    # Plot 1
    sns.scatterplot(
        x=df[nuc_marker],
        y=df[cell_size],
        hue=df[color_by],
        palette=palette,
        alpha=alpha,
        ax=ax2,
        legend=False
    )

    if log_scale == True:
        ax2.set_xscale("log")
        ax2.set_yscale("log")

    ax2.axhline(size_thres, color="k", linestyle="dashed", linewidth=1)
    ax2.axvline(nuc_thres, color="k", linestyle="dashed", linewidth=1)
    ax2.legend(bbox_to_anchor=(1.2, 1), loc="upper left", borderaxespad=0)
    ax2.set_title("Before filtering")

    # Plot 2
    sns.scatterplot(
        x=df_nuc[nuc_marker],
        y=df_nuc[cell_size],
        hue=df_nuc[color_by],
        palette=palette,
        alpha=alpha,
        ax=ax3
    )

    if log_scale == True:
        ax3.set_xscale("log")
        ax3.set_yscale("log")

    ax3.axhline(size_thres, color="k", linestyle="dashed", linewidth=1)
    ax3.axvline(nuc_thres, color="k", linestyle="dashed", linewidth=1)
    ax3.legend(bbox_to_anchor=(1.2, 1), loc="upper left", borderaxespad=0)
    ax3.set_title("After filtering")

    plt.show()
    # show plot
    plt.tight_layout()
    plt.show()
    
    # print the percentage of cells that are kept
    print("Percentage of cells kept: ", per_keep * 100, "%")
    #print(f"Number of cells removed per region:\n{df.groupby(region_column).size() - df_nuc.groupby(region_column).size()}")
    
    # print five point statistics for cell size and nuclear marker intensity before filtering
    #print("BEFORE FILTERING:")

    #print("Five point statistics for cell size and nuclear marker intensity:")
    #print(df.loc[:, [cell_size, nuc_marker]].describe())

    
    # print five point statistics for cell size and nuclear marker intensity after filtering
    #print("AFTER FILTERING:")

    #print("Five point statistics for cell size and nuclear marker intensity:")
    #print(df_nuc.loc[:, [cell_size, nuc_marker]].describe())
    
    return df_nuc


def pp_format(data, list_out, list_keep, method="zscore", ArcSin_cofactor=150):
    # original function:
    # #Drop column list
    # list1 = [col for col in data.columns if 'blank' in col]
    # list_out1 = list1+list_out

    # #Drop columns not interested in
    # dfin = data.drop(list_out1, axis = 1)

    # #save columns for later
    # df_loc = dfin.loc[:,list_keep]

    # #dataframe for normalization
    # dfz = dfin.drop(list_keep, axis = 1)

    # #zscore of the column markers
    # dfz1 = pd.DataFrame(zscore(dfz,0),index = dfz.index,columns = [i for i in dfz.columns])

    # #Add back labels for normalization type
    # dfz_all = pd.concat([dfz1, df_loc], axis=1, join='inner')

    # print("the number of regions = "+str(len(dfz_all.region_num.unique())))

    # return dfz_all
    list = ["zscore", "double_zscore", "MinMax", "ArcSin"]

    if method not in list:
        print("Please select methods from zscore, double_zscore, MinMax, ArcSin!")
        exit()

    ##ArcSin transformation
    if method == "ArcSin":
        # Drop column list
        list1 = [col for col in data.columns if "blank" in col]
        list_out1 = list1 + list_out

        # Drop columns not interested in
        dfin = data.drop(list_out1, axis=1)

        # save columns for later
        df_loc = dfin.loc[:, list_keep]

        # dataframe for normalization
        dfas = dfin.drop(list_keep, axis=1)

        # parameters seit in function
        # Only decrease the background if the median is higher than the background
        dfa = dfas.apply(lambda x: np.arcsinh(x / ArcSin_cofactor))

        # Add back labels for normalization type
        dfz_all = pd.concat([dfa, df_loc], axis=1, join="inner")

        return dfz_all

    ##Double Z normalization
    elif method == "double_zscore":
        # Drop column list
        list1 = [col for col in data.columns if "blank" in col]
        list_out1 = list1 + list_out

        # Drop columns not interested in
        dfin = data.drop(list_out1, axis=1)

        # save columns for later
        df_loc = dfin.loc[:, list_keep]

        # dataframe for normalization
        dfz = dfin.drop(list_keep, axis=1)

        # zscore of the column markers
        dfz1 = pd.DataFrame(
            zscore(dfz, 0), index=dfz.index, columns=[i for i in dfz.columns]
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
        # Drop column list
        list1 = [col for col in data.columns if "blank" in col]
        list_out1 = list1 + list_out

        # Drop columns not interested in
        dfin = data.drop(list_out1, axis=1)

        # save columns for later
        df_loc = dfin.loc[:, list_keep]

        # dataframe for normalization
        dfmm = dfin.drop(list_keep, axis=1)

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
        # Drop column list
        list1 = [col for col in data.columns if "blank" in col]
        list_out1 = list1 + list_out

        # Drop columns not interested in
        dfin = data.drop(list_out1, axis=1)

        # save columns for later
        df_loc = dfin.loc[:, list_keep]

        # dataframe for normalization
        dfz = dfin.drop(list_keep, axis=1)

        # zscore of the column markers
        dfz1 = pd.DataFrame(
            zscore(dfz, 0), index=dfz.index, columns=[i for i in dfz.columns]
        )

        # Add back labels for normalization type
        dfz_all = pd.concat([dfz1, df_loc], axis=1, join="inner")

        # print("the number of regions = "+str(len(dfz_all.region_num.unique())))

        return dfz_all


# Only useful for "classic CODEX" where samples are covered by multiple regions
# Could also be used for montages of multiple samples (tiles arraged in grid)
def pp_xycorr(data, y_rows, x_columns, X_pix, Y_pix):
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
def pp_remove_noise(df, col_num, z_sum_thres, z_count_thres):
    df_z_1_copy = df.copy()
    df_z_1_copy["Count"] = df_z_1_copy.iloc[:, : col_num + 1].ge(0).sum(axis=1)
    df_z_1_copy["z_sum"] = df_z_1_copy.iloc[:, : col_num + 1].sum(axis=1)
    cc = df_z_1_copy[
        (df_z_1_copy["z_sum"] > z_sum_thres) | (df_z_1_copy["Count"] > z_count_thres)
    ]
    df_want = df_z_1_copy[
        ~((df_z_1_copy["z_sum"] > z_sum_thres) | (df_z_1_copy["Count"] > z_count_thres))
    ]
    percent_removed = np.round(1 - (df_want.shape[0]/ df_z_1_copy.shape[0]), decimals = 3)
    print(str(percent_removed*100) + "% cells are removed.")
    df_want.drop(columns=["Count", "z_sum"], inplace=True)
    df_want.reset_index(inplace=True, drop=True)
    return df_want, cc


def pp_clust_leid(adata, res=1, Matrix_plot=True):
    # Compute the neighborhood relations of single cells the range 2 to 100 and usually 10
    sc.pp.neighbors(adata, n_neighbors=10)

    # Perform leiden clustering - improved version of louvain clustering
    sc.tl.leiden(adata, resolution=res, key_added="leiden")

    # UMAP computation
    sc.tl.umap(adata)

    plt.rcParams["legend.markerscale"] = 1
    sc.pl.umap(adata, color=["leiden"])

    m_list = adata.var.index.to_list()

    # Create matrix plot with mean expression per each cluster
    if Matrix_plot == True:
        sc.pl.matrixplot(adata, m_list, "leiden", standard_scale="var")

    return adata, m_list


# Function to remove segmentation artifacts
def pp_remove_segmentation_artifacts(
    df, size_thres=0, nuc_thres=0, cellsize_column="area", nuc_marker_column="Hoechst1"
):
    df_copy = df.copy()

    if size_thres > 0:
        df_copy = df_copy[df_copy[cellsize_column] > size_thres]

    if nuc_thres > 0:
        df_copy = df_copy[df_copy[nuc_marker_column] > nuc_thres]

    return df_copy

