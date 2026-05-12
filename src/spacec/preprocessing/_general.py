# load required packages
import glob
import os
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from scipy.stats import norm, zscore


# read the data frame output from the segmentation functions
def read_segdf(
    segfile_list,
    seg_method,
    region_list=None,  # optional information #please make sure the length of each list matches
    meta_list=None,  # optional information
):
    """
    Read the data frame output from the segmentation functions.

    Parameters
    ----------
    segfile_list : list
        List of segmented csv files to be read.
    seg_method : str
        The segmentation method used.
    region_list : list, optional
        List of regions, by default None. Please make sure the length of each list matches.
    meta_list : list, optional
        List of metadata, by default None. Please make sure the length of each list matches.

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
    elif meta_list is not None:
        if len(meta_list) != len(segfile_list):
            sys.exit("length of each list does not match!")

    df = pd.DataFrame()
    # concat old dataframes
    for i in range(len(segfile_list)):
        tmp = pd.read_csv(segfile_list[i], index_col=0)
        tmp["region_num"] = str(i)
        if region_list is not None:
            tmp["unique_region"] = str(region_list[i])
        if meta_list is not None:
            tmp["condition"] = str(meta_list[i])
        df = pd.concat([df, tmp], axis=0)

    if seg_method == "cellseg":
        # See resultant dataframe
        df.columns = df.columns.str.split(":").str[-1].tolist()
        df = df.reset_index().rename(columns={"index": "first_index"})
        df.columns = df.columns.str.split(":").str[-1].tolist()
        df.rename(columns={"size": "area"}, inplace=True)
    return df


def filter_data(
    df,
    nuc_thres=1,
    size_thres=1,
    nuc_marker="DAPI",
    cell_size="area",
    region_column="region_num",
    color_by=None,
    palette="Paired",
    alpha=0.8,
    size=0.4,  # dot style
    log_scale=False,
    plot=False,
):
    """
    Filter data based on nuclear threshold and size threshold, and visualize the data before and after filtering.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be filtered.
    nuc_thres : int, optional
        The nuclear threshold, by default 1.
    size_thres : int, optional
        The size threshold, by default 1.
    nuc_marker : str, optional
        The nuclear marker, by default "DAPI".
    cell_size : str, optional
        The cell size, by default "area".
    region_column : str, optional
        The region column, by default "region_num".
    color_by : str, optional
        The column to color by, by default None.
    palette : str, optional
        The color palette, by default "Paired".
    alpha : float, optional
        The alpha for the scatter plot, by default 0.8.
    size : float, optional
        The size for the scatter plot, by default 0.4.
    log_scale : bool, optional
        Whether to use log scale for the scatter plot, by default False.

    Returns
    -------
    df_nuc : pandas.DataFrame
        The filtered DataFrame.
    """
    if color_by == None:
        color_by = region_column

    df_nuc = df[(df[nuc_marker] > nuc_thres) * df[cell_size] > size_thres]
    per_keep = len(df_nuc) / len(df)

    # Create a figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Boxplot
    # sns.boxplot(data=df.loc[:, [cell_size, nuc_marker]], orient="h", ax=ax1)
    # ax1.set_title("Cell size and nuclear marker intensity")
    if plot:
        print("Plotting scatterplots now, if your data is large, it will take awhile!")
        sns.scatterplot(
            x=df[nuc_marker],
            y=df[cell_size],
            hue=df[color_by],
            palette=palette,
            alpha=alpha,
            ax=ax2,
            legend=False,
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
            ax=ax3,
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

    return df_nuc


def format(data, list_out, list_keep, method="zscore", ArcSin_cofactor=150):
    """
    This function formats the data based on the specified method. It supports four methods: "zscore", "double_zscore", "MinMax", and "ArcSin".

    Parameters
    ----------
    data : DataFrame
        The input data to be formatted.
    list_out : list
        The list of columns to be dropped from the data.
    list_keep : list
        The list of columns to be kept in the data.
    method : str, optional
        The method to be used for normalizing the data. It can be "zscore", "double_zscore", "MinMax", or "ArcSin". By default, it is "zscore".
    ArcSin_cofactor : int, optional
        The cofactor to be used in the ArcSin transformation. By default, it is 150.

    Returns
    -------
    DataFrame
        The formatted data.

    Raises
    ------
    ValueError
        If the specified method is not supported.
    """

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
def remove_noise(df, col_num, z_sum_thres, z_count_thres):
    """
    Removes noisy cells from the dataset based on the given thresholds.

    Parameters
    ----------
    df : DataFrame
        The input data from which noisy cells are to be removed.
    col_num : int
        The column number up to which the operation is performed.
    z_sum_thres : float
        The threshold for the sum of z-scores. Cells with a sum of z-scores greater than this threshold are considered noisy.
    z_count_thres : int
        The threshold for the count of z-scores. Cells with a count of z-scores greater than this threshold are considered noisy.

    Returns
    -------
    df_want : DataFrame
        The cleaned data with noisy cells removed.
    cc : DataFrame
        The data of the noisy cells that were removed from the original data.

    """
    df_z_1_copy = df.copy()
    df_z_1_copy["Count"] = df_z_1_copy.iloc[:, : col_num + 1].ge(0).sum(axis=1)
    df_z_1_copy["z_sum"] = df_z_1_copy.iloc[:, : col_num + 1].sum(axis=1)
    cc = df_z_1_copy[
        (df_z_1_copy["z_sum"] > z_sum_thres) | (df_z_1_copy["Count"] > z_count_thres)
    ]
    df_want = df_z_1_copy[
        ~((df_z_1_copy["z_sum"] > z_sum_thres) | (df_z_1_copy["Count"] > z_count_thres))
    ]
    percent_removed = np.round(
        1 - (df_want.shape[0] / df_z_1_copy.shape[0]), decimals=3
    )
    print(str(percent_removed * 100) + "% cells are removed.")
    df_want.drop(columns=["Count", "z_sum"], inplace=True)
    df_want.reset_index(inplace=True, drop=True)
    return df_want, cc


class ImageProcessor:
    """
    A class used to process images and compute channel means and sums.

    ...

    Attributes
    ----------
    flatmasks : ndarray
        2D numpy array containing masks for each cell.

    Methods
    -------
    update_adjacency_value(adjacency_matrix, original, neighbor):
        Updates the adjacency matrix based on the original and neighbor values.
    update_adjacency_matrix(plane_mask_flattened, width, height, adjacency_matrix, index):
        Updates the adjacency matrix based on the flattened plane mask.
    compute_channel_means_sums_compensated(image):
        Computes the channel means and sums for each cell and compensates them.
    """

    def __init__(self, flatmasks):
        """
        Constructs all the necessary attributes for the ImageProcessor object.

        Parameters
        ----------
            flatmasks : ndarray
                2D numpy array containing masks for each cell.
        """
        self.flatmasks = flatmasks

    def update_adjacency_value(self, adjacency_matrix, original, neighbor):
        # This function is copied from CellSeg
        """
        Updates the adjacency matrix based on the original and neighbor values.

        Parameters
        ----------
            adjacency_matrix : ndarray
                2D numpy array representing the adjacency matrix.
            original : int
                Original value.
            neighbor : int
                Neighbor value.

        Returns
        -------
            bool
                True if the original and neighbor values are different and not zero, False otherwise.
        """
        border = False

        if original != 0 and original != neighbor:
            border = True
            if neighbor != 0:
                adjacency_matrix[int(original - 1), int(neighbor - 1)] += 1
        return border

    def update_adjacency_matrix(
        self, plane_mask_flattened, width, height, adjacency_matrix, index
    ):
        # This function uses code from CellSeg
        """
        Updates the adjacency matrix based on the flattened plane mask.

        Parameters
        ----------
            plane_mask_flattened : ndarray
                1D numpy array representing the flattened plane mask.
            width : int
                Width of the plane mask.
            height : int
                Height of the plane mask.
            adjacency_matrix : ndarray
                2D numpy array representing the adjacency matrix.
            index : int
                Index of the current cell in the flattened plane mask.
        """
        mod_value_width = index % width
        origin_mask = plane_mask_flattened[index]
        left, right, up, down = False, False, False, False

        if mod_value_width != 0:
            left = self.update_adjacency_value(
                adjacency_matrix, origin_mask, plane_mask_flattened[index - 1]
            )
        if mod_value_width != width - 1:
            right = self.update_adjacency_value(
                adjacency_matrix, origin_mask, plane_mask_flattened[index + 1]
            )
        if index >= width:
            up = self.update_adjacency_value(
                adjacency_matrix, origin_mask, plane_mask_flattened[index - width]
            )
        if index <= len(plane_mask_flattened) - 1 - width:
            down = self.update_adjacency_value(
                adjacency_matrix, origin_mask, plane_mask_flattened[index + width]
            )

        if left or right or up or down:
            adjacency_matrix[int(origin_mask - 1), int(origin_mask - 1)] += 1

    def compute_channel_means_sums_compensated(self, image, device=None):
        """
        Computes and compensates channel means and sums for each cell in a multi-channel image using
        vectorized operations and GPU acceleration when available.

        This method processes a multi-channel image and its corresponding cell masks to:
        1. Calculate mean intensities for each channel in each cell
        2. Build an adjacency matrix representing cell neighborhoods
        3. Compensate for potential intensity bleeding between adjacent cells using least squares optimization

        Parameters
        ----------
        image : np.ndarray
            A 3D array of shape (height, width, n_channels) containing the multi-channel image data.
            Values should be in float32 format.
        device: str|None
            None (default) will select the compute device automatically.
            Can be forced to `cuda`, `mps`, or `cpu`.

        Returns
        -------
        compensated_means : np.ndarray
            A 2D array of shape (n_masks, n_channels) containing the compensated mean intensities
            for each cell (mask) across all channels.
        means : np.ndarray
            A 2D array of shape (n_masks, n_channels) containing the original (uncompensated)
            mean intensities for each cell across all channels.
        channel_counts : np.ndarray
            A 1D array of length n_masks containing the number of pixels in each cell mask.

        Notes
        -----
        The method uses the instance's `flatmasks` attribute which should be a 2D array where
        each unique positive integer represents a different cell mask (0 represents background).

        The compensation process involves:
        - Computing mean intensities per channel for each cell
        - Creating an adjacency matrix representing cell neighborhoods
        - Solving a least squares optimization problem to adjust for intensity bleeding
        - Using GPU acceleration (CUDA or MPS) when available

        The adjacency matrix is built considering:
        - Direct neighbors (left, right, up, down)
        - Diagonal contributions for border pixels
        - Self-connections for cells with borders

        See Also
        --------
        torch.linalg.lstsq : The underlying least squares solver used for compensation
        numpy.bincount : Used for efficient computation of channel sums

        Examples
        --------
        >>> processor = ImageProcessor(masks)
        >>> image_data = np.random.rand(100, 100, 3).astype(np.float32)
        >>> comp_means, orig_means, counts = processor.compute_channel_means_sums_compensated(image_data)
        """
        height, width, n_channels = image.shape
        M = self.flatmasks  # shape: (mask_height, mask_width)
        mask_height, mask_width = M.shape
        n_masks = len(np.unique(M)) - 1
        if n_masks == 0:
            zeros = np.zeros((n_masks, n_channels), dtype=np.float32)
            zeros_counts = np.zeros(n_masks, dtype=np.float32)
            return zeros, zeros, zeros_counts

        # Reshape the image and mask for vectorized operations
        squashed_image = image.reshape(-1, n_channels)
        flat_mask = M.flatten()

        valid = flat_mask > 0
        mask_ids = flat_mask[valid].astype(np.int32) - 1
        image_valid = squashed_image[valid]

        channel_sums = np.zeros((n_masks, n_channels), dtype=np.float32)
        for ch in range(n_channels):
            channel_sums[:, ch] = np.bincount(
                mask_ids, weights=image_valid[:, ch], minlength=n_masks
            )
        counts = np.bincount(mask_ids, minlength=n_masks)
        channel_counts = np.tile(counts, (n_channels, 1)).T

        means = np.true_divide(
            channel_sums,
            channel_counts,
            out=np.zeros_like(channel_sums, dtype=np.float32),
            where=channel_counts != 0,
        )
        means = means.astype(np.float32)

        # Build adjacency matrix using vectorized numpy operations.
        adj = np.zeros((n_masks, n_masks), dtype=np.float32)
        h, w = M.shape

        # Left neighbors
        orig = M[:, 1:]
        left_val = M[:, :-1]
        cond = (orig != 0) & (left_val != 0) & (orig != left_val)
        rows = (orig[cond] - 1).astype(np.int32)
        cols = (left_val[cond] - 1).astype(np.int32)
        np.add.at(adj, (rows, cols), 1)

        # Right neighbors
        orig = M[:, :-1]
        right_val = M[:, 1:]
        cond = (orig != 0) & (right_val != 0) & (orig != right_val)
        rows = (orig[cond] - 1).astype(np.int32)
        cols = (right_val[cond] - 1).astype(np.int32)
        np.add.at(adj, (rows, cols), 1)

        # Up neighbors
        orig = M[1:, :]
        up_val = M[:-1, :]
        cond = (orig != 0) & (up_val != 0) & (orig != up_val)
        rows = (orig[cond] - 1).astype(np.int32)
        cols = (up_val[cond] - 1).astype(np.int32)
        np.add.at(adj, (rows, cols), 1)

        # Down neighbors
        orig = M[:-1, :]
        down_val = M[1:, :]
        cond = (orig != 0) & (down_val != 0) & (orig != down_val)
        rows = (orig[cond] - 1).astype(np.int32)
        cols = (down_val[cond] - 1).astype(np.int32)
        np.add.at(adj, (rows, cols), 1)

        # Diagonal contributions for pixels bordering a different cell
        has_border = np.zeros_like(M, dtype=bool)
        has_border[:, 1:] |= (M[:, 1:] != 0) & (M[:, 1:] != M[:, :-1])
        has_border[:, :-1] |= (M[:, :-1] != 0) & (M[:, :-1] != M[:, 1:])
        has_border[1:, :] |= (M[1:, :] != 0) & (M[1:, :] != M[:-1, :])
        has_border[:-1, :] |= (M[:-1, :] != 0) & (M[:-1, :] != M[1:, :])
        diag_ids = (M[has_border][M[has_border] != 0] - 1).astype(np.int32)
        np.add.at(adj, (diag_ids, diag_ids), 1)

        diag = np.diag(adj)
        denom = np.maximum(diag, 1)
        adj = adj / (denom[:, None] * 2)
        np.fill_diagonal(adj, 1)

        # Run the least squares solver using torch on CUDA if available, otherwise use MPS on Mac, else CPU.
        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        adjacency_matrix_torch = torch.from_numpy(adj).to(device)
        means_torch = torch.from_numpy(means).to(device)
        results_torch = torch.linalg.lstsq(adjacency_matrix_torch, means_torch).solution
        results = (
            results_torch.cpu().numpy()
        )  # Move results back to CPU for further computations
        compensated_means = np.maximum(results, 0)

        return compensated_means, means, channel_counts[:, 0]


def compensate_cell_matrix(df, image_dict, masks, overwrite=True, device=None):
    """
    Compensate cell matrix by computing channel means and sums.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to which the compensated means will be added.
    image_dict : dict
        Dictionary containing images for each channel.
    masks : ndarray
        3D numpy array containing masks for each cell.
    overwrite : bool, optional
        If True, overwrite existing columns in df. If False, add new columns to df. Default is True.
    device: str|None
        None (default) will select the compute device automatically (for `compute_channel_means_sums_compensated`).
        Can be forced to `cuda`, `mps`, or `cpu`.

    Returns
    -------
    DataFrame
        The DataFrame with added compensated means.

    Notes
    -----
    The function computes the channel means and sums for each cell, compensates them, and adds them to the DataFrame.
    The compensated means are added to the DataFrame with column names from the keys of the image_dict.
    If overwrite is True, existing columns in the DataFrame are overwritten. If overwrite is False, new columns are added to the DataFrame.
    """

    print(
        "This function uses code from Lee, M.Y., Bedia, J.S., Bhate, S.S. et al. CellSeg: a robust, pre-trained nucleus segmentation and pixel quantification software for highly multiplexed fluorescence images. BMC Bioinformatics 23, 46 (2022) please consider citing the original work."
    )

    masks = masks.squeeze()
    image_list = [
        image_dict[channel_name]
        for channel_name in image_dict.keys()
        if channel_name != "segmentation_channel"
    ]

    # Stack the 2D numpy arrays along the third dimension to create a 3D numpy array
    image = np.stack(image_list, axis=-1)

    # Now you can use `image` as the input for the function
    processor = ImageProcessor(masks)
    (
        compensated_means,
        means,
        channel_counts,
    ) = processor.compute_channel_means_sums_compensated(image, device=device)

    # Get the keys
    keys = [
        channel for channel in image_dict.keys() if channel != "segmentation_channel"
    ]

    if overwrite:
        new_cols = pd.DataFrame(compensated_means, columns=keys, index=df.index)
        df.update(new_cols)
    else:
        new_keys = [k + "_compensated" for k in keys]
        new_cols = pd.DataFrame(compensated_means, columns=new_keys, index=df.index)
        df = pd.concat([df, new_cols], axis=1)

    return df
