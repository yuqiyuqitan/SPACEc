# load required packages
import os
import time

import concave_hull
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st
import skimage
import skimage.color
import skimage.exposure
import skimage.filters.rank
import skimage.io as io
import skimage.morphology
import skimage.transform
import statsmodels.api as sm
from concave_hull import concave_hull_indexes
from joblib import Parallel, delayed
from scipy import stats
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm
from yellowbrick.cluster import KElbowVisualizer

from ..helperfunctions._general import *

# Tools
############################################################


############
"""
The function tl_cell_types_de performs differential enrichment analysis for various cell subsets between different neighborhoods using linear regression.
It takes in several inputs such as cell type frequencies, neighborhood numbers, and patient information.
The function first normalizes overall cell type frequencies and then neighborhood-specific cell type frequencies. Next, a linear regression model is fitted to find the coefficients and p-values for the group coefficient.
Finally, the function returns a dataframe with the coefficients and p-values for each cell subset. The p-values can be corrected for multiple testing after the function has been executed.
"""


def tl_cell_types_de(
    ct_freq, all_freqs, neighborhood_num, nbs, patients, group, cells, cells1
):
    # data prep
    # normalized overall cell type frequencies
    X_cts = hf_normalize(
        ct_freq.reset_index().set_index("patients").loc[patients, cells]
    )

    # normalized neighborhood specific cell type frequencies
    df_list = []

    for nb in nbs:
        cond_nb = (
            all_freqs.loc[all_freqs[neighborhood_num] == nb, cells1]
            .rename({col: col + "_" + str(nb) for col in cells}, axis=1)
            .set_index("patients")
        )
        df_list.append(hf_normalize(cond_nb))

    X_cond_nb = pd.concat(df_list, axis=1).loc[patients]

    # differential enrichment for all cell subsets
    changes = {}
    # nbs =[0, 2, 3, 4, 6, 7, 8, 9]
    for col in cells:
        for nb in nbs:
            # build a design matrix with a constant, group 0 or 1 and the overall frequencies
            X = pd.concat(
                [
                    X_cts[col],
                    group.astype("int"),
                    pd.Series(np.ones(len(group)), index=group.index.values),
                ],
                axis=1,
            ).values
            if col + "_%d" % nb in X_cond_nb.columns:
                # set the neighborhood specific ct freqs as the outcome
                Y = X_cond_nb[col + "_%d" % nb].values
                X = X[~pd.isna(Y)]
                Y = Y[~pd.isna(Y)]
                # fit a linear regression model
                results = sm.OLS(Y, X).fit()
                # find the params and pvalues for the group coefficient
                changes[(col, nb)] = (results.pvalues[1], results.params[1])

    # make a dataframe with coeffs and pvalues
    dat = pd.DataFrame(changes).loc[1].unstack()
    dat = (
        pd.DataFrame(np.nan_to_num(dat.values), index=dat.index, columns=dat.columns)
        .T.sort_index(ascending=True)
        .loc[:, X_cts.columns]
    )
    pvals = (
        (pd.DataFrame(changes).loc[0].unstack())
        .T.sort_index(ascending=True)
        .loc[:, X_cts.columns]
    )

    # this is where you should correct pvalues for multiple testing

    return dat, pvals


#########


def tl_Create_neighborhoods(
    df, n_num, cluster_col, X, Y, regions, sum_cols=None, keep_cols=None, ks=[20]
):
    if sum_cols == None:
        sum_cols = df[cluster_col].unique()

    if keep_cols == None:
        keep_cols = df.columns.values.tolist()

    Neigh = Neighborhoods(
        df, ks, cluster_col, sum_cols, keep_cols, X, Y, regions, add_dummies=True
    )
    windows = Neigh.k_windows()

    return (windows, sum_cols)


######


def tl_Chose_window_size(
    windows, n_num, n_neighborhoods, sum_cols, n2_name="neigh_ofneigh"
):
    # Choose the windows size to continue with
    w = windows[n_num]

    k_centroids = {}

    km = MiniBatchKMeans(n_clusters=n_neighborhoods, random_state=0)
    labels = km.fit_predict(w[sum_cols].values)
    k_centroids[n_num] = km.cluster_centers_
    w[n2_name] = labels

    return (w, k_centroids)


#######


def tl_calculate_neigh_combs(w, l, n_num, threshold=0.85, per_keep_thres=0.85):
    w.loc[:, l]

    # need to normalize by number of neighborhoods or k chosen for the neighborhoods
    xm = w.loc[:, l].values / n_num

    # Get the neighborhood combinations based on the threshold
    simps = hf_get_thresh_simps(xm, threshold)
    simp_freqs = simps.value_counts(normalize=True)
    simp_sums = np.cumsum(simp_freqs)

    # See the percent to keep threshold or percent of neigbhorhoods that fall above a certain threshold
    test_sums_thres = simp_sums[simp_sums < per_keep_thres]
    test_len = len(test_sums_thres)
    per_values_above = simp_sums[test_len] - simp_sums[test_len - 1]
    print(test_len, per_values_above)

    w["combination"] = [tuple(l[a] for a in s) for s in simps]
    w["combination_num"] = [tuple(a for a in s) for s in simps]

    # this shows what proportion (y) of the total cells are assigned to the top x combinations
    plt.figure(figsize=(7, 3))
    plt.plot(simp_sums.values)
    plt.title(
        "proportion (y) of the total cells are assigned to the top x combinations"
    )
    plt.show()

    # this shows what proportion (y) of the total cells are assigned to the top x combinations
    plt.figure(figsize=(7, 3))
    plt.plot(test_sums_thres.values)
    plt.title(
        "proportion (y) of the total cells are assigned to the top x combinations - thresholded"
    )
    plt.show()
    # plt.xticks(range(0,350,35),range(0,350,35),rotation = 90,fontsize = 10)

    return (simps, simp_freqs, simp_sums)


#######


def tl_build_graph_CN_comb_map(simp_freqs, thresh_freq=0.001):
    g = nx.DiGraph()

    # selected_simps = simp_sums[simp_sums<=thresh_cumulative].index.values
    selected_simps = simp_freqs[simp_freqs >= thresh_freq].index.values
    selected_simps

    """
    this builds the graph for the CN combination map
    """
    for e0 in selected_simps:
        for e1 in selected_simps:
            if (set(list(e0)) < set(list(e1))) and (len(e1) == len(e0) + 1):
                g.add_edge(e0, e1)

    tops = (
        simp_freqs[simp_freqs >= thresh_freq]
        .sort_values(ascending=False)
        .index.values.tolist()[:20]
    )

    return (g, tops, e0, e1)


#######


def tl_spatial_context_stats(
    n_num,
    patient_ID_component1,
    patient_ID_component2,
    windows,
    total_per_thres=0.9,
    comb_per_thres=0.005,
    tissue_column="Block type",
    subset_list=["Resection"],
    plot_order=["Resection", "Biopsy"],
    pal_tis={"Resection": "blue", "Biopsy": "orange"},
    subset_list_tissue1=["Resection"],
    subset_list_tissue2=["Biopsy"],
):
    data_compare = windows[n_num]

    # Prepare IDs this could for example be the combination of patient ID and tissue type. Apart from that, the function assigns a number to each name from the neighborhood column
    data_compare = prepare_neighborhood_df(
        data_compare,
        neighborhood_column=None,
        patient_ID_component1=patient_ID_component1,
        patient_ID_component2=patient_ID_component2,
    )  # this is a helper function

    data_compare["donor_tis"].unique()

    simp_df_tissue1 = hf_simp_rep(
        data=data_compare,
        patient_col="donor_tis",
        tissue_column=tissue_column,
        subset_list_tissue=subset_list_tissue1,
        ttl_per_thres=total_per_thres,
        comb_per_thres=comb_per_thres,
        thres_num=1,
    )
    print(simp_df_tissue1)

    simp_df_tissue2 = hf_simp_rep(
        data=data_compare,
        patient_col="donor_tis",
        tissue_column=tissue_column,
        subset_list_tissue=subset_list_tissue2,
        ttl_per_thres=total_per_thres,
        comb_per_thres=comb_per_thres,
        thres_num=1,
    )
    print(simp_df_tissue2)

    ##### Compare the organization at high level to see if differences in combinations - more or less structured/compartmentalized
    data_simp = [simp_df_tissue1, simp_df_tissue2]
    df_num_count = pl_comb_num_freq(data_list=data_simp)
    print(df_num_count)

    return (simp_df_tissue1, simp_df_tissue2)


###########


def tl_xycorr(df, sample_col, y_rows, x_columns, X_pix, Y_pix):
    # Make a copy for xy correction
    df_XYcorr = df.copy()

    df_XYcorr["Xcorr"] = 0
    df_XYcorr["Ycorr"] = 0

    for sample in df_XYcorr[sample_col].unique():
        df_sub = df_XYcorr.loc[df_XYcorr[sample_col] == sample]
        region_num = df_sub.region.max().astype(int)

        # first value of tuple is y and second is x
        d = list(product(range(0, y_rows, 1), range(0, x_columns, 1)))
        e = list(range(1, region_num + 1, 1))
        dict_corr = {}
        dict_corr = dict(zip(e, d))

        # Adding the pixels with the dictionary
        for x in range(1, region_num + 1, 1):
            df_XYcorr["Xcorr"].loc[
                (df_XYcorr["region"] == x) & (df_XYcorr[sample_col] == sample)
            ] = (
                df_XYcorr["x"].loc[
                    (df_XYcorr["region"] == x) & (df_XYcorr[sample_col] == sample)
                ]
                + dict_corr[x][1] * X_pix
            )

        for x in range(1, region_num + 1, 1):
            df_XYcorr["Ycorr"].loc[
                (df_XYcorr["region"] == x) & (df_XYcorr[sample_col] == sample)
            ] = (
                df_XYcorr["y"].loc[
                    (df_XYcorr["region"] == x) & (df_XYcorr[sample_col] == sample)
                ]
                + dict_corr[x][0] * Y_pix
            )

    return df_XYcorr


###############


def tl_get_distances(df, cell_list, cell_type_col):
    names = cell_list
    cls = {}
    for i, cname in enumerate(names):
        cls[i] = df[["x", "y"]][df[cell_type_col] == cname].to_numpy()
        cls[i] = cls[i][~np.isnan(cls[i]).any(axis=1), :]

    dists = {}

    for i in range(5):
        for j in range(0, i):
            dists[(j, i)] = cdist(cls[j], cls[i])
            dists[(i, j)] = dists[(j, i)]
    return cls, dists


###############
# clustering


def clustering(
    adata,
    clustering="leiden",
    marker_list=None,
    resolution=1,
    n_neighbors=10,
    reclustering=False,
    key_added=None,
    **cluster_kwargs,
):
    if clustering not in ["leiden", "louvain"]:
        print("Invalid clustering options. Please select from leiden or louvain!")
        exit()

    if key_added is None:
        key_added = clustering + "_" + str(resolution)

    # input a list of markers for clustering
    # reconstruct the anndata
    if marker_list is not None:
        if len(list(set(marker_list) - set(adata.var_names))) > 0:
            print("Marker list not all in adata var_names! Using intersection instead!")
            marker_list = list(set(marker_list) & set(adata.var_names))
            print("New marker_list: " + " ".join(marker_list))
        adata_tmp = adata
        adata = adata[:, marker_list]
    # Compute the neighborhood relations of single cells the range 2 to 100 and usually 10
    if reclustering:
        print("Clustering")
        if clustering == "leiden":
            sc.tl.leiden(
                adata, resolution=resolution, key_added=key_added, **cluster_kwargs
            )
        else:
            sc.tl.louvain(
                adata, resolution=resolution, key_added=key_added, **cluster_kwargs
            )
    else:
        print("Computing neighbors and UMAP")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        # UMAP computation
        sc.tl.umap(adata)
        print("Clustering")
        # Perform leiden clustering - improved version of louvain clustering
        if clustering == "leiden":
            sc.tl.leiden(
                adata, resolution=resolution, key_added=key_added, **cluster_kwargs
            )
        else:
            sc.tl.louvain(
                adata, resolution=resolution, key_added=key_added, **cluster_kwargs
            )

    if marker_list is None:
        return adata
    else:
        adata_tmp.obs[key_added] = adata.obs[key_added].values
        # append other data
        adata_tmp.obsm = adata.obsm
        adata_tmp.obsp = adata.obsp
        adata_tmp.uns = adata.uns
        return adata_tmp


###############
# Patch analysis


def tl_generate_voronoi_plots(
    df,
    output_path,
    grouping_col="Community",
    tissue_col="tissue",
    region_col="unique_region",
    x_col="x",
    y_col="y",
):
    """
    Generate Voronoi plots for unique combinations of tissue and region.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing the data.
        output_path (str): Output path to save the plots.
        grouping_col (str): Column that contains group label that is used to color the voronoi diagrams
        tissue_col (str): Column that contains tissue labels
        region_col (str): Column that contains region labels
        x_col (str): Column that contains x coordinates
        y_col (str): Column that contains y coordinates

    Returns:
        None
    """

    unique_tissues = df[tissue_col].unique()
    unique_regions = df[region_col].unique()

    combinations = list(itertools.product(unique_tissues, unique_regions))

    for tissue, region in combinations:
        subset_df = df[(df[tissue_col] == tissue) & (df[region_col] == region)]
        sorted_df = subset_df.sort_values(grouping_col)
        unique_values = sorted_df[grouping_col].unique()

        specific_output = os.path.join(output_path, tissue)
        os.makedirs(specific_output, exist_ok=True)
        specific_output = os.path.join(specific_output, region)
        os.makedirs(specific_output, exist_ok=True)

        for group in unique_values:
            start = time.time()

            output_filename = group + "_plot.png"
            output_path2 = os.path.join(specific_output, output_filename)

            color_dict = {}
            for value in unique_values:
                color_dict[value] = "black"
            color_dict[group] = "white"

            X = sorted_df[x_col]
            Y = sorted_df[y_col]
            np.random.seed(1234)
            points = np.c_[X, Y]

            vor = Voronoi(points)
            regions, vertices = hf_voronoi_finite_polygons_2d(vor)
            groups = sorted_df[grouping_col].values

            fig, ax = plt.subplots()
            ax.set_ylim(0, max(Y))
            ax.set_xlim(0, max(X))
            ax.axis("off")

            for i, region in tqdm(
                enumerate(regions), total=len(regions), desc="Processing regions"
            ):
                group = groups[i]
                color = color_dict.get(group, "gray")
                polygon = vertices[region]
                ax.fill(*zip(*polygon), color=color)

            ax.plot(points[:, 0], points[:, 1], "o", color="black", zorder=1, alpha=0)

            fig.set_size_inches(9.41, 9.07 * 1.02718006795017)
            fig.savefig(
                output_path2, bbox_inches="tight", pad_inches=0, dpi=129.0809327846365
            )
            plt.close(fig)

            end = time.time()
            print(end - start)


def tl_generate_masks_from_images(
    image_folder, mask_output, image_type=".tif", filter_size=5, threshold_value=10
):
    """
    Generate binary masks from CODEX images.

    Parameters:
        image_folder (str): Directory that contains the images that are used to generate the masks
        mask_output (str): Directory to store the generated masks
        image_type (str): File type of image. By default ".tif"
        filter_size (num): Size for filter disk during mask generation
        threshold_value (num): Threshold value for binary mask generation

    Returns:
        None
    """
    folders_list = hf_list_folders(image_folder)
    print(folders_list)
    for folder in tqdm(folders_list, desc="Processing folders"):
        direc = image_folder + "/" + folder
        print(direc)

        filelist = os.listdir(direc)
        filelist = [f for f in filelist if f.endswith(image_type)]
        print(filelist)

        output_dir = mask_output + folder
        os.makedirs(output_dir, exist_ok=True)

        for f in tqdm(filelist, desc="Processing files"):
            path = os.path.join(direc, f)
            print(path)

            tl_generate_mask(
                path=path,
                output_dir=output_dir,
                filename="/" + f,
                filter_size=filter_size,
                threshold_value=threshold_value,
            )


def tl_generate_info_dataframe(
    df,
    voronoi_output,
    mask_output,
    filter_list=None,
    info_cols=["tissue", "donor", "unique_region", "region", "array"],
):
    """
    Generate a filtered DataFrame based on specific columns and values.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.
        voronoi_output (str): Path to the Voronoi output directory.
        mask_output (str): Path to the mask output directory.
        info_cols (list): columns to extract from input df
        filter_list (list, optional): List of values to filter.

    Returns:
        pandas.DataFrame: Filtered DataFrame.
    """
    df_info = df[info_cols].drop_duplicates()
    df_info["folder_names"] = df_info["array"]
    df_info["region"] = df_info["region"].astype(int)
    df_info["region_long"] = ["reg00" + str(region) for region in df_info["region"]]
    df_info["voronoi_path"] = (
        voronoi_output + df_info["tissue"] + "/" + df_info["unique_region"]
    )
    df_info["mask_path"] = mask_output + df_info["folder_names"] + "/"

    if filter_list != None:
        # remove unwanted folders
        df_info = df_info[~df_info["folder_names"].isin(filter_list)]

    else:
        print("no filter used")

    return df_info


###


def tl_process_files(voronoi_path, mask_path, region):
    """
    Process files based on the provided paths and region.

    Parameters:
        voronoi_path (str): Path to the Voronoi files.
        mask_path (str): Path to the mask files.
        region (str): Region identifier.

    Returns:
        None
    """
    png_files_list = hf_get_png_files(voronoi_path)
    tiff_file_path = hf_find_tiff_file(mask_path, region)

    if tiff_file_path:
        print(f"Matching TIFF file found: {tiff_file_path}")
    else:
        print("No matching TIFF file found.")

    for f in tqdm(png_files_list, desc="Processing files"):
        print(f)
        tl_apply_mask(f, tiff_file_path, f + "_cut.png")


###


def tl_process_data(df_info, output_dir_csv):
    """
    Process data based on the information provided in the DataFrame.

    Parameters:
        df_info (pandas.DataFrame): DataFrame containing the information.
        output_dir_csv (str): Output directory for CSV results.

    Returns:
        pandas.DataFrame: Concatenated DataFrame of results.
        list: List of contours.
    """
    DF_list = []
    contour_list = []

    for index, row in df_info.iterrows():
        voronoi_path = row["voronoi_path"]
        mask_path = row["mask_path"]
        region = row["region_long"]
        donor = row["donor"]
        unique_region = row["unique_region"]

        png_files_list = hf_get_png_files(voronoi_path)
        png_files_list = [
            filename for filename in png_files_list if not filename.endswith("cut.png")
        ]

        tiff_file_path = hf_find_tiff_file(mask_path, region)

        if tiff_file_path:
            print(f"Matching TIFF file found: {tiff_file_path}")
        else:
            print("No matching TIFF file found.")

        for f in png_files_list:
            print(f)
            g = f + "_cut" + ".png"
            print(g)
            tl_apply_mask(f, tiff_file_path, g)

            output_dir_csv_tmp = output_dir_csv + "/" + donor + "_" + unique_region
            os.makedirs(output_dir_csv_tmp, exist_ok=True)

            image_dir = output_dir_csv + "/" + donor + "_" + unique_region
            os.makedirs(image_dir, exist_ok=True)
            print(f"Path created: {image_dir}")

            image_dir = os.path.join(image_dir, os.path.basename(os.path.normpath(g)))
            path = g

            df, contour = tl_analyze_image(
                path,
                invert=False,
                output_dir=image_dir,
            )

            df["group"] = hf_extract_filename(g)
            df["unique_region"] = unique_region

            DF_list.append(df)
            contour_list.append(contour)

    results_df = pd.concat(DF_list)
    contour_list_results_df = pd.concat(DF_list)

    results_df.to_csv(os.path.join(output_dir_csv, "results.csv"))

    return results_df, contour_list


###


def tl_analyze_image(
    path,
    output_dir,
    invert=False,
    properties_list=[
        "label",
        "centroid",
        "area",
        "perimeter",
        "solidity",
        "coords",
        "axis_minor_length",
        "axis_major_length",
        "orientation",
        "slice",
    ],
):
    """
    Analyze an image by performing connected component analysis on patches and storing their information.

    The function applies image processing techniques such as Gaussian smoothing, thresholding, and connected component
    labeling to identify and analyze patches within the image. It extracts region properties of these patches,
    calculates their circularity, and stores the coordinates of their contour. The resulting information is saved
    in a DataFrame along with a visualization plot.

    Parameters:
        path (str): Path to the input image.
        output_dir (str): Directory to save the output plot.
        invert (bool, optional): Flag indicating whether to invert the image (default is False).
        properties_list: (list of str): Define properties to be measured (see SciKit Image), by default "label", "centroid", "area", "perimeter", "solidity", "coords", "axis_minor_length", "axis_major_length", "orientation", "slice"

    Returns:
        tuple: A tuple containing the DataFrame with region properties, including patch contour coordinates, and
               the list of contour coordinates for each patch.
    """
    image = skimage.io.imread(path)

    if image.ndim == 2:
        print("2D array")
    else:
        image = image[:, :, 0]

    if invert:
        print(
            "The original background color was white. The image was inverted for further analysis."
        )
        # image = 255 - image
    else:
        print("no inversion")

    smooth = skimage.filters.gaussian(image, sigma=1.5)
    thresh = smooth > skimage.filters.threshold_otsu(smooth)

    blobs_labels = skimage.measure.label(thresh, background=0)

    properties = skimage.measure.regionprops(blobs_labels)

    props_table = skimage.measure.regionprops_table(
        blobs_labels,
        properties=(properties_list),
    )

    prop_df = pd.DataFrame(props_table)

    prop_df["circularity"] = (4 * np.pi * prop_df["area"]) / (prop_df["perimeter"] ** 2)

    # Store the contour of each patch in the DataFrame
    contour_list = []
    for index in range(1, blobs_labels.max()):
        label_i = properties[index].label
        contour = skimage.measure.find_contours(blobs_labels == label_i, 0.5)[0]
        contour_list.append(contour)

    contour_list_df = pd.DataFrame({"contours": contour_list})

    prop_df = pd.concat([prop_df, contour_list_df], axis=1)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(1, 2, 1)
    plt.imshow(thresh, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(blobs_labels, cmap="nipy_spectral")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(output_dir)
    plt.close()

    return prop_df, contour_list


###


def tl_apply_mask(image_path, mask_path, output_path):
    """
    Apply a mask to an image and save the resulting masked image.

    Parameters:
        image_path (str): Path to the input image.
        mask_path (str): Path to the mask image.
        output_path (str): Path to save the masked image.

    Returns:
        None
    """
    # Load the image and the mask
    image = io.imread(image_path)
    mask = io.imread(mask_path, as_gray=True)
    mask = np.flip(mask, axis=0)

    width = 941
    height = 907
    image = skimage.transform.resize(image, (height, width))

    if image.ndim == 2:
        print("2D array")
    else:
        image = image[:, :, :3]

        # Convert to grayscale
        image = skimage.color.rgb2gray(image)

    # Convert to 8-bit
    image = skimage.img_as_ubyte(image)

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    # Ensure the mask is binary
    mask = mask > 0

    # Apply the mask to the image
    masked_image = image.copy()
    masked_image[~mask] = 0

    # Check if the image has an alpha channel (transparency)
    if masked_image.ndim == 2:
        print("2D array")
    else:
        masked_image = masked_image[:, :, :3]

    # Save the masked image
    io.imsave(output_path, skimage.img_as_ubyte(masked_image))


###


def tl_generate_mask(
    path, output_dir, filename="mask.png", filter_size=5, threshold_value=5
):
    """
    Generate a mask from a maximum projection of an input image.

    Parameters:
        path (str): Path to the input image.
        output_dir (str): Directory to save the generated mask and quality control plot.
        filename (str, optional): Name of the generated mask file (default is "mask.png").
        filter_size (int, optional): Size of the filter disk used for image processing (default is 5).
        threshold_value (int, optional): Threshold value for binary conversion (default is 5).

    Returns:
        None
    """
    # Load the image
    image = io.imread(path)

    # Perform Z projection using Maximum Intensity
    z_projection = np.max(image, axis=0)

    # Resize the image
    width = 941
    height = 907
    resized_image = skimage.transform.resize(
        z_projection, (height, width, 3), preserve_range=True
    )
    print("Resized image shape:", resized_image.shape)

    # Remove alpha channel if present
    if resized_image.shape[-1] == 4:
        resized_image = resized_image[:, :, :3]

    # Convert to grayscale
    gray_image = skimage.color.rgb2gray(resized_image)

    # Assuming gray_image has pixel values outside the range [0, 1]
    # Normalize the pixel values to the range [0, 1]
    gray_image_normalized = (gray_image - gray_image.min()) / (
        gray_image.max() - gray_image.min()
    )

    # Convert to 8-bit
    gray_image_8bit = skimage.img_as_ubyte(gray_image_normalized)

    # Apply maximum filter
    max_filtered = skimage.filters.rank.maximum(
        gray_image_8bit, skimage.morphology.disk(filter_size)
    )

    # Apply minimum filter
    min_filtered = skimage.filters.rank.minimum(
        max_filtered, skimage.morphology.disk(filter_size)
    )

    # Apply median filter
    median_filtered = skimage.filters.rank.median(
        min_filtered, skimage.morphology.disk(filter_size)
    )

    # Manual Thresholding
    binary = median_filtered > threshold_value

    # Convert to mask
    mask = skimage.morphology.closing(binary, skimage.morphology.square(3))

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    axes[0, 0].imshow(gray_image, cmap="gray")
    axes[0, 0].set_title("Grayscale Image")

    axes[0, 1].imshow(gray_image_8bit, cmap="gray")
    axes[0, 1].set_title("8-bit Image")

    axes[0, 2].imshow(max_filtered, cmap="gray")
    axes[0, 2].set_title("Maximum Filtered")

    axes[1, 0].imshow(min_filtered, cmap="gray")
    axes[1, 0].set_title("Minimum Filtered")

    axes[1, 1].imshow(median_filtered, cmap="gray")
    axes[1, 1].set_title("Median Filtered")

    axes[1, 2].imshow(mask, cmap="gray")
    axes[1, 2].set_title("Mask")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_dir + filename + "_QC_plot.png", dpi=300, format="png")

    plt.show()

    # Save the result
    io.imsave(output_dir + filename, mask)


#####


def tl_test_clustering_resolutions(
    adata, clustering="leiden", n_neighbors=10, resolutions=[1]
):
    """
    Test different resolutions for reclustering using Louvain or Leiden algorithm.

    Parameters:
        adata (AnnData): Anndata object containing the data.
        clustering (str, optional): Clustering algorithm to use (default is 'leiden').
        n_neighbors (int, optional): Number of nearest neighbors (default is 10).
        resolutions (list, optional): List of resolutions to test (default is [1]).

    Returns:
        None
    """
    for res in tqdm(resolutions, desc="Testing resolutions"):
        if "leiden" in clustering:
            clustering(
                adata,
                clustering="leiden",
                n_neighbors=n_neighbors,
                res=res,
                reclustering=True,
            )
        else:
            clustering(
                adata,
                clustering="louvain",
                n_neighbors=n_neighbors,
                res=res,
                reclustering=True,
            )

        sc.pl.umap(adata, color=f"{clustering}_{res}", legend_loc="on data")


###############
# clustering


def tl_clustering_ad(
    adata,
    clustering="leiden",
    marker_list=None,
    res=1,
    n_neighbors=10,
    reclustering=False,
):
    if clustering not in ["leiden", "louvain"]:
        print("Invalid clustering options. Please select from leiden or louvain!")
        exit()
    # input a list of markers for clustering
    # reconstruct the anndata
    if marker_list is not None:
        if len(list(set(marker_list) - set(adata.var_names))) > 0:
            print("Marker list not all in adata var_names! Using intersection instead!")
            marker_list = list(set(marker_list) & set(adata.var_names))
            print("New marker_list: " + " ".join(marker_list))
        adata_tmp = adata
        adata = adata[:, marker_list]
    # Compute the neighborhood relations of single cells the range 2 to 100 and usually 10
    if reclustering:
        print("Clustering")
        if clustering == "leiden":
            sc.tl.leiden(adata, resolution=res, key_added="leiden_" + str(res))
        else:
            sc.tl.louvain(adata, resolution=res, key_added="louvain" + str(res))
    else:
        print("Computing neighbors and UMAP")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        # UMAP computation
        sc.tl.umap(adata)
        print("Clustering")
        # Perform leiden clustering - improved version of louvain clustering
        if clustering == "leiden":
            sc.tl.leiden(adata, resolution=res, key_added="leiden_" + str(res))
        else:
            sc.tl.louvain(adata, resolution=res, key_added="louvain" + str(res))

    if marker_list is None:
        return adata
    else:
        if clustering == "leiden":
            adata_tmp.obs["leiden_" + str(res)] = adata.obs["leiden_" + str(res)].values
        else:
            adata_tmp.obs["leiden_" + str(res)] = adata.obs[
                "louvain_" + str(res)
            ].values
        # append other data
        adata_tmp.obsm = adata.obsm
        adata_tmp.obsp = adata.obsp
        adata_tmp.uns = adata.uns
        return adata_tmp


def neighborhood_analysis(
    adata,
    unique_region,
    cluster_col,
    X="x",
    Y="y",
    k=35,
    n_neighborhoods=30,
    elbow=False,
    metric="distortion",
):
    """
    Compute for Cellular neighborhood
    adata:  anndata containing information
    unique_region: each region is one independent CODEX image
    cluster_col:  columns to compute CNs on, typicall 'celltype'
    X:  X
    Y: Y
    k: k neighbors to compute
    n_neighborhoods: number of neighborhoods one ends ups with
    elbow: whether to see the optimal elbow plots or not
    metric:
    """
    df = pd.DataFrame(adata.obs[[X, Y, cluster_col, unique_region]])

    cells = pd.concat([df, pd.get_dummies(df[cluster_col])], axis=1)
    sum_cols = cells[cluster_col].unique()
    values = cells[sum_cols].values

    neighborhood_name = "CN" + "_k" + str(k) + "_n" + str(n_neighborhoods)
    centroids_name = "Centroid" + "_k" + str(k) + "_n" + str(n_neighborhoods)

    n_neighbors = k

    cells[unique_region] = cells[unique_region].astype("str")
    cells["cellid"] = cells.index.values
    cells.reset_index(inplace=True)

    keep_cols = [X, Y, unique_region, cluster_col]

    # Get each region
    tissue_group = cells[[X, Y, unique_region]].groupby(unique_region)
    exps = list(cells[unique_region].unique())
    tissue_chunks = [
        (time.time(), exps.index(t), t, a)
        for t, indices in tissue_group.groups.items()
        for a in np.array_split(indices, 1)
    ]

    tissues = [
        hf_get_windows(job, n_neighbors, exps=exps, tissue_group=tissue_group, X=X, Y=Y)
        for job in tissue_chunks
    ]

    # Loop over k to compute neighborhoods
    out_dict = {}

    for neighbors, job in zip(tissues, tissue_chunks):
        chunk = np.arange(len(neighbors))  # indices
        tissue_name = job[2]
        indices = job[3]
        window = (
            values[neighbors[chunk, :k].flatten()]
            .reshape(len(chunk), k, len(sum_cols))
            .sum(axis=1)
        )
        out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)

    windows = {}

    window = pd.concat(
        [
            pd.DataFrame(
                out_dict[(exp, k)][0],
                index=out_dict[(exp, k)][1].astype(int),
                columns=sum_cols,
            )
            for exp in exps
        ],
        axis=0,
    )
    window = window.loc[cells.index.values]
    window = pd.concat([cells[keep_cols], window], axis=1)
    windows[k] = window

    # Fill in based on above
    k_centroids = {}

    # producing what to plot
    windows2 = windows[k]
    windows2[cluster_col] = cells[cluster_col]

    if elbow != True:
        km = MiniBatchKMeans(n_clusters=n_neighborhoods, random_state=0)

        labels = km.fit_predict(windows2[sum_cols].values)
        k_centroids[str(k)] = km.cluster_centers_
        adata.obs[neighborhood_name] = labels
        adata.uns[centroids_name] = k_centroids

    else:
        km = MiniBatchKMeans(random_state=0)

        X = windows2[sum_cols].values

        labels = km.fit_predict(X)
        k_centroids[str(k)] = km.cluster_centers_
        adata.obs[neighborhood_name] = labels
        adata.uns[centroids_name] = k_centroids

        visualizer = KElbowVisualizer(
            km, k=(n_neighborhoods), timings=False, metric=metric
        )
        visualizer.fit(X)  # Fit the data to the visualizer
        visualizer.show()  # Finalize and render the figure

    return adata


def cn_map(
    adata,
    cn_col,
    unique_region,
    palette=None,
    k=75,
    X="x",
    Y="y",
    threshold=0.85,
    per_keep_thres=0.85,
    sub_list=None,
    sub_col=None,
    rand_seed=1,
):
    ks = [k]
    cells_df = pd.DataFrame(adata.obs)
    cells_df = cells_df[[X, Y, unique_region, cn_col]]
    cells_df.reset_index(inplace=True)
    sum_cols = cells_df[cn_col].unique()
    keep_cols = cells_df.columns

    cn_colors = hf_generate_random_colors(
        len(adata.obs[cn_col].unique()), rand_seed=rand_seed
    )

    if palette is None:
        if cn_col + "_colors" not in adata.uns.keys():
            palette = dict(zip(np.sort(adata.obs[cn_col].unique()), cn_colors))
            adata.uns[cn_col + "_colors"] = cn_colors
        else:
            palette = dict(
                zip(np.sort(adata.obs[cn_col].unique()), adata.uns[cn_col + "_colors"])
            )

    Neigh = Neighborhoods(
        cells_df,
        ks,
        cn_col,
        sum_cols,
        keep_cols,
        X,
        Y,
        reg=unique_region,
        add_dummies=True,
    )
    windows = Neigh.k_windows()
    w = windows[k]
    if sub_list:
        # convert sub_list to list if only str is provided
        if isinstance(sub_list, str):
            sub_list = [sub_list]
        w = w[w[sub_col].isin(sub_list)]
    l = list(palette.keys())
    simps, simp_freqs, simp_sums = tl_calculate_neigh_combs(
        w, l, k, threshold=threshold, per_keep_thres=per_keep_thres  # color palette
    )
    g, tops, e0, e1 = tl_build_graph_CN_comb_map(simp_freqs)
    return {"g": g, "tops": tops, "e0": e0, "e1": e1, "simp_freqs": simp_freqs}


def tl_format_for_squidpy(adata, x_col, y_col):
    # Extract the count data from your original AnnData object
    counts = adata.X

    # Extract the spatial coordinates from the 'obs' metadata
    spatial_coordinates = adata.obs[[x_col, y_col]].values

    # Create a new AnnData object with the expected format
    new_adata = ad.AnnData(counts, obsm={"spatial": spatial_coordinates})

    return new_adata


def tl_corr_cell_ad(
    adata, per_categ, grouping_col, rep, sub_column, normed=True, sub_list2=None
):
    """
    Perform correlation analysis on a pandas DataFrame and plot correlation scatter plots.

    Parameters
    ----------
    data : pandas DataFrame
        The input DataFrame.
    per_categ : str
        The categorical column in the DataFrame to be used.
    grouping_col : str
        The grouping column in the DataFrame.
    rep : str
        The replicate column in the DataFrame.
    sub_column : str
        The subcategory column in the DataFrame.
    normed : bool, optional
        If the percentage should be normalized. Default is True.
    sub_list2 : list, optional
        A list of subcategories to be considered. Default is None.

    Returns
    -------
    cmat : pandas DataFrame
        The correlation matrix DataFrame.
    cc : pandas DataFrame
        The DataFrame after pivoting and formatting for correlation function.
    """
    data = adata.obs
    cmat, cc = tl_corr_cell(
        data,
        per_categ,
        grouping_col=grouping_col,
        rep=rep,
        sub_column=sub_column,
        normed=normed,
        sub_list2=sub_list2,
    )

    return cmat, cc


def calculate_triangulation_distances(df_input, id, x_pos, y_pos, cell_type, region):
    # Perform Delaunay triangulation
    points = df_input[[x_pos, y_pos]].values
    tri = Delaunay(points)
    indices = tri.simplices

    # Get interactions going both directions
    edges = set()
    for simplex in indices:
        for i in range(3):
            for j in range(i + 1, 3):
                edges.add(tuple(sorted([simplex[i], simplex[j]])))
    edges = np.array(list(edges))

    # Create dataframe from edges
    rdelaun_result = pd.DataFrame(edges, columns=["ind1", "ind2"])
    rdelaun_result[["x1", "y1"]] = df_input.iloc[rdelaun_result["ind1"]][
        [x_pos, y_pos]
    ].values
    rdelaun_result[["x2", "y2"]] = df_input.iloc[rdelaun_result["ind2"]][
        [x_pos, y_pos]
    ].values

    # Annotate results with cell type and region information
    df_input["XYcellID"] = (
        df_input[x_pos].astype(str) + "_" + df_input[y_pos].astype(str)
    )
    rdelaun_result["cell1ID"] = (
        rdelaun_result["x1"].astype(str) + "_" + rdelaun_result["y1"].astype(str)
    )
    rdelaun_result["cell2ID"] = (
        rdelaun_result["x2"].astype(str) + "_" + rdelaun_result["y2"].astype(str)
    )

    annotated_result = pd.merge(
        rdelaun_result, df_input, left_on="cell1ID", right_on="XYcellID"
    )
    annotated_result = annotated_result.rename(
        columns={cell_type: "celltype1", id: "celltype1_index"}
    )
    annotated_result = annotated_result.drop(columns=[x_pos, y_pos, region, "XYcellID"])

    annotated_result = pd.merge(
        annotated_result,
        df_input,
        left_on="cell2ID",
        right_on="XYcellID",
        suffixes=(".x", ".y"),
    )
    annotated_result = annotated_result.rename(
        columns={cell_type: "celltype2", id: "celltype2_index"}
    )
    annotated_result = annotated_result.drop(columns=[x_pos, y_pos, "XYcellID"])

    # Calculate distance
    annotated_result["distance"] = np.sqrt(
        (annotated_result["x2"] - annotated_result["x1"]) ** 2
        + (annotated_result["y2"] - annotated_result["y1"]) ** 2
    )

    # Reorder columns
    annotated_result = annotated_result[
        [
            region,
            "celltype1_index",
            "celltype1",
            "x1",
            "y1",
            "celltype2_index",
            "celltype2",
            "x2",
            "y2",
            "distance",
        ]
    ]
    annotated_result.columns = [
        region,
        "celltype1_index",
        "celltype1",
        "celltype1_X",
        "celltype1_Y",
        "celltype2_index",
        "celltype2",
        "celltype2_X",
        "celltype2_Y",
        "distance",
    ]

    return annotated_result


# Define the process_region function at the top level
def process_region(df, unique_region, id, x_pos, y_pos, cell_type, region):
    subset = df[df[region] == unique_region].copy()
    subset["uniqueID"] = (
        subset[id].astype(str)
        + "-"
        + subset[x_pos].astype(str)
        + "-"
        + subset[y_pos].astype(str)
    )
    subset["XYcellID"] = subset[x_pos].astype(str) + "_" + subset[y_pos].astype(str)
    result = calculate_triangulation_distances(
        df_input=subset,
        id=id,
        x_pos=x_pos,
        y_pos=y_pos,
        cell_type=cell_type,
        region=region,
    )
    return result


def get_triangulation_distances(
    df_input, id, x_pos, y_pos, cell_type, region, num_cores=None, correct_dtype=True
):
    if correct_dtype == True:
        # change columns to pandas string
        df_input[cell_type] = df_input[cell_type].astype(str)
        df_input[region] = df_input[region].astype(str)

    # Check if x_pos and y_pos are integers, and if not, convert them
    if not issubclass(df_input[x_pos].dtype.type, np.integer):
        print("This function expects integer values for xy coordinates.")
        print(
            x_pos
            + " and "
            + y_pos
            + " will be changed to integer. Please check the generated output!"
        )
        df_input[x_pos] = df_input[x_pos].astype(int).values
        df_input[y_pos] = df_input[y_pos].astype(int).values

    # Get unique regions
    unique_regions = df_input[region].unique()

    # Select only necessary columns
    df_input = df_input.loc[:, [id, x_pos, y_pos, cell_type, region]]

    # Set up parallelization
    if num_cores is None:
        num_cores = os.cpu_count() // 2  # default to using half of available cores

    # Parallel processing using joblib
    results = Parallel(n_jobs=num_cores)(
        delayed(process_region)(df_input, reg, id, x_pos, y_pos, cell_type, region)
        for reg in unique_regions
    )

    triangulation_distances = pd.concat(results)

    return triangulation_distances


def shuffle_annotations(df_input, cell_type, region, permutation):
    # Set the seed for reproducibility
    np.random.seed(permutation + 1234)

    # Create a copy to avoid modifying the original dataframe
    df_shuffled = df_input.copy()

    # Shuffle annotations within each region
    for region_name in df_shuffled[region].unique():
        region_mask = df_shuffled[region] == region_name
        shuffled_values = df_shuffled.loc[region_mask, cell_type].sample(frac=1).values
        df_shuffled.loc[region_mask, "random_annotations"] = shuffled_values

    return df_shuffled


def tl_iterate_tri_distances(
    df_input, id, x_pos, y_pos, cell_type, region, num_cores=None, num_iterations=1000
):
    unique_regions = df_input[region].unique()
    # Use only the necessary columns
    df_input = df_input[[id, x_pos, y_pos, cell_type, region]]

    if num_cores is None:
        num_cores = os.cpu_count() // 2  # Default to using half of available cores

    # Define a helper function to process each region and iteration
    def process_iteration(region_name, iteration):
        # Filter by region
        subset = df_input[df_input[region] == region_name].copy()
        # Create unique IDs
        subset.loc[:, "uniqueID"] = (
            subset[id].astype(str)
            + "-"
            + subset[x_pos].astype(str)
            + "-"
            + subset[y_pos].astype(str)
        )
        subset.loc[:, "XYcellID"] = (
            subset[x_pos].astype(str) + "_" + subset[y_pos].astype(str)
        )

        # Shuffle annotations
        shuffled = shuffle_annotations(subset, cell_type, region, iteration)

        # Get triangulation distances
        results = get_triangulation_distances(
            df_input=shuffled,
            id=id,
            x_pos=x_pos,
            y_pos=y_pos,
            cell_type="random_annotations",
            region=region,
            num_cores=num_cores,
            correct_dtype=False,
        )

        # Summarize results
        per_cell_summary = (
            results.groupby(["celltype1_index", "celltype1", "celltype2"])
            .distance.mean()
            .reset_index(name="per_cell_mean_dist")
        )

        per_celltype_summary = (
            per_cell_summary.groupby(["celltype1", "celltype2"])
            .per_cell_mean_dist.mean()
            .reset_index(name="mean_dist")
        )
        per_celltype_summary[region] = region_name
        per_celltype_summary["iteration"] = iteration

        return per_celltype_summary

    # TODO: remove nans valid here a good idea (attempt to fix windows unpickle issue)?
    unique_regions = [r for r in unique_regions if r != np.nan]

    # Parallel processing for each region and iteration
    results = Parallel(n_jobs=num_cores)(
        delayed(process_iteration)(region_name, iteration)
        for region_name in unique_regions
        for iteration in range(1, num_iterations + 1)
    )

    # Combine all results
    iterative_triangulation_distances = pd.concat(results, ignore_index=True)
    # iterative_triangulation_distances = iterative_triangulation_distances.dropna()
    return iterative_triangulation_distances


def tl_iterate_tri_distances_ad(
    adata,
    id,
    x_pos,
    y_pos,
    cell_type,
    region,
    num_cores=None,
    num_iterations=1000,
    key_name=None,
    correct_dtype=True,
):
    df_input = pd.DataFrame(adata.obs)
    df_input[id] = df_input.index

    if correct_dtype == True:
        # change columns to pandas string
        df_input[cell_type] = df_input[cell_type].astype(str)
        df_input[region] = df_input[region].astype(str)

    # Check if x_pos and y_pos are integers, and if not, convert them
    if not issubclass(df_input[x_pos].dtype.type, np.integer):
        print("This function expects integer values for xy coordinates.")
        print("Class will be changed to integer. Please check the generated output!")
        df_input[x_pos] = df_input[x_pos].astype(int).values
        df_input[y_pos] = df_input[y_pos].astype(int).values

    unique_regions = df_input[region].unique()
    # Use only the necessary columns
    df_input = df_input.loc[:, [id, x_pos, y_pos, cell_type, region]]

    if num_cores is None:
        num_cores = os.cpu_count() // 2  # Default to using half of available cores

    # Define a helper function to process each region and iteration
    def process_iteration(region_name, iteration):
        # Filter by region
        subset = df_input.loc[df_input[region] == region_name, :].copy()
        subset.loc[:, "uniqueID"] = (
            subset[id].astype(str)
            + "-"
            + subset[x_pos].astype(str)
            + "-"
            + subset[y_pos].astype(str)
        )
        subset.loc[:, "XYcellID"] = (
            subset[x_pos].astype(str) + "_" + subset[y_pos].astype(str)
        )

        # Shuffle annotations
        shuffled = shuffle_annotations(subset, cell_type, region, iteration)

        # Get triangulation distances
        results = get_triangulation_distances(
            df_input=shuffled,
            id=id,
            x_pos=x_pos,
            y_pos=y_pos,
            cell_type="random_annotations",
            region=region,
            num_cores=num_cores,
            correct_dtype=False,
        )

        # Summarize results
        per_cell_summary = (
            results.groupby(["celltype1_index", "celltype1", "celltype2"])
            .distance.mean()
            .reset_index(name="per_cell_mean_dist")
        )

        per_celltype_summary = (
            per_cell_summary.groupby(["celltype1", "celltype2"])
            .per_cell_mean_dist.mean()
            .reset_index(name="mean_dist")
        )
        per_celltype_summary[region] = region_name
        per_celltype_summary["iteration"] = iteration

        return per_celltype_summary

    # Parallel processing for each region and iteration
    results = Parallel(n_jobs=num_cores)(
        delayed(process_iteration)(region_name, iteration)
        for region_name in unique_regions
        for iteration in range(1, num_iterations + 1)
    )

    # Combine all results
    iterative_triangulation_distances = pd.concat(results, ignore_index=True)

    # append result to adata
    if key_name is None:
        key_name = "iTriDist_" + str(num_iterations)
    adata.uns[key_name] = iterative_triangulation_distances
    print("Save iterative triangulation distance output to anndata.uns " + key_name)

    return iterative_triangulation_distances


def add_missing_columns(
    triangulation_distances, metadata, shared_column="unique_region"
):
    # Find the difference in columns
    missing_columns = set(metadata.columns) - set(triangulation_distances.columns)
    # Add missing columns to triangulation_distances with NaN values
    for column in missing_columns:
        triangulation_distances[column] = pd.NA
        # Create a mapping from unique_region to tissue in metadata
        region_to_tissue = pd.Series(
            metadata[column].values, index=metadata["unique_region"]
        ).to_dict()

        # Apply this mapping to the triangulation_distances dataframe to create/update the tissue column
        triangulation_distances[column] = triangulation_distances["unique_region"].map(
            region_to_tissue
        )

        # Handle regions with no corresponding tissue in the metadata by filling in a default value
        triangulation_distances[column].fillna("Unknown", inplace=True)
    return triangulation_distances


# Calculate p-values and log fold differences
# def calculate_pvalue(row):
#    try:
#        return st.ttest_ind(row['expected'], row['observed'], alternative='two-sided', equal_var = False).pvalue
#    except ValueError:  # This handles cases with insufficient data
#        return np.nan


# Calculate p-values and log fold differences
def calculate_pvalue(row):
    try:
        return st.mannwhitneyu(
            row["expected"], row["observed"], alternative="two-sided"
        ).pvalue
    except ValueError:  # This handles cases with insufficient data
        return np.nan


def tl_identify_interactions(
    triangulation_distances,
    iterative_triangulation_distances,
    metadata,
    min_observed=10,
    distance_threshold=128,
    comparison="tissue",
):
    # Reformat observed dataset
    triangulation_distances_long = add_missing_columns(
        triangulation_distances, metadata, shared_column="unique_region"
    )

    observed_distances = (
        triangulation_distances_long.query("distance <= @distance_threshold")
        .groupby(
            ["celltype1_index", "celltype1", "celltype2", comparison, "unique_region"]
        )
        .agg(mean_per_cell=("distance", "mean"))
        .reset_index()
        .groupby(["celltype1", "celltype2", comparison])
        .agg(observed=("mean_per_cell", list), observed_mean=("mean_per_cell", "mean"))
        .reset_index()
    )

    # Reformat expected dataset
    iterated_triangulation_distances_long = add_missing_columns(
        iterative_triangulation_distances, metadata, shared_column="unique_region"
    )

    expected_distances = (
        iterated_triangulation_distances_long.query("mean_dist <= @distance_threshold")
        .groupby(["celltype1", "celltype2", comparison])
        .agg(expected=("mean_dist", list), expected_mean=("mean_dist", "mean"))
        .reset_index()
    )

    # Drop comparisons with low numbers of observations
    observed_distances["keep"] = observed_distances["observed"].apply(
        lambda x: len(x) > min_observed
    )
    observed_distances = observed_distances[observed_distances["keep"]]

    expected_distances["keep"] = expected_distances["expected"].apply(
        lambda x: len(x) > min_observed
    )
    expected_distances = expected_distances[expected_distances["keep"]]

    # concatenate observed and expected distances
    distance_pvals = expected_distances.merge(
        observed_distances, on=["celltype1", "celltype2", comparison], how="left"
    )

    distance_pvals = expected_distances.merge(
        observed_distances, on=["celltype1", "celltype2", comparison], how="left"
    )
    distance_pvals["pvalue"] = distance_pvals.apply(calculate_pvalue, axis=1)
    distance_pvals["logfold_group"] = np.log2(
        distance_pvals["observed_mean"] / distance_pvals["expected_mean"]
    )
    distance_pvals["interaction"] = (
        distance_pvals["celltype1"] + " --> " + distance_pvals["celltype2"]
    )

    # drop na from distance_pvals
    # distance_pvals = distance_pvals.dropna()

    return distance_pvals


def filter_interactions(distance_pvals, pvalue=0.05, logfold_group_abs=0.1):
    # calculate absolute logfold difference
    distance_pvals["logfold_group_abs"] = distance_pvals["logfold_group"].abs()

    # Creating pairs
    distance_pvals["pairs"] = (
        distance_pvals["celltype1"] + "_" + distance_pvals["celltype2"]
    )

    # Filter significant p-values and other specified conditions
    distance_pvals_sig = distance_pvals[
        (distance_pvals["pvalue"] < pvalue)
        & (distance_pvals["celltype1"] != distance_pvals["celltype2"])
        & (~distance_pvals["observed_mean"].isna())
        & (distance_pvals["logfold_group_abs"] > logfold_group_abs)
    ]

    # calculate absolute logfold difference
    distance_pvals["logfold_group_abs"] = distance_pvals["logfold_group"].abs()

    # Filter significant p-values and other specified conditions
    distance_pvals_sig = distance_pvals[
        (distance_pvals["pvalue"] < pvalue)
        & (distance_pvals["celltype1"] != distance_pvals["celltype2"])
        & (~distance_pvals["observed_mean"].isna())
    ]

    # Assuming distance_pvals_interesting2 is a pandas DataFrame with the same structure as the R dataframe.
    # pair_to = distance_pvals_sig["interaction"].unique()
    pairs = distance_pvals_sig["pairs"].unique()

    # Filtering data
    data = distance_pvals_sig[~distance_pvals_sig["interaction"].isna()]

    # Subsetting data
    distance_pvals_sig_sub = data[data["pairs"].isin(pairs)]
    distance_pvals_sig_sub_reduced = distance_pvals_sig_sub.loc[
        :, ["condition", "logfold_group", "pairs"]
    ].copy()

    # set pairs as index
    distance_pvals_sig_sub_reduced = distance_pvals_sig_sub_reduced.set_index("pairs")

    # sort logfold_group into two columns by tissue
    dist_table = distance_pvals_sig_sub_reduced.pivot(
        columns="condition", values="logfold_group"
    )
    dist_table.dropna(inplace=True)

    return dist_table, distance_pvals_sig_sub


def identify_interactions(
    adata,
    id,
    x_pos,
    y_pos,
    cell_type,
    region,
    comparison,
    iTriDist_keyname=None,
    triDist_keyname=None,
    min_observed=10,
    distance_threshold=128,
    num_cores=None,
    num_iterations=1000,
    key_name=None,
    correct_dtype=False,
):
    df_input = pd.DataFrame(adata.obs)
    df_input[id] = df_input.index

    # change columns to pandas string
    df_input[cell_type] = df_input[cell_type].astype(str)
    df_input[region] = df_input[region].astype(str)

    print("Computing for observed distances between cell types!")
    triangulation_distances = get_triangulation_distances(
        df_input=df_input,
        id=id,
        x_pos=x_pos,
        y_pos=y_pos,
        cell_type=cell_type,
        region=region,
        num_cores=num_cores,
        correct_dtype=correct_dtype,
    )
    if triDist_keyname is None:
        triDist_keyname = "triDist"
    adata.uns["triDist_keyname"] = triangulation_distances
    print("Save triangulation distances output to anndata.uns " + triDist_keyname)

    print("Permuting data labels to obtain the randomly distributed distances!")
    print("this step can take awhile")
    iterative_triangulation_distances = tl_iterate_tri_distances(
        df_input=df_input,
        id=id,
        x_pos=x_pos,
        y_pos=y_pos,
        cell_type=cell_type,
        region=region,
        num_cores=num_cores,
        num_iterations=num_iterations,
    )

    # append result to adata
    if triDist_keyname is None:
        triDist_keyname = "iTriDist_" + str(num_iterations)
    adata.uns[triDist_keyname] = iterative_triangulation_distances
    print(
        "Save iterative triangulation distance output to anndata.uns " + triDist_keyname
    )

    metadata = df_input.loc[:, ["unique_region", "condition"]].copy()
    # Reformat observed dataset
    triangulation_distances_long = add_missing_columns(
        triangulation_distances, metadata, shared_column=region
    )

    observed_distances = (
        triangulation_distances_long.query("distance <= @distance_threshold")
        .groupby(["celltype1_index", "celltype1", "celltype2", comparison, region])
        .agg(mean_per_cell=("distance", "mean"))
        .reset_index()
        .groupby(["celltype1", "celltype2", comparison])
        .agg(observed=("mean_per_cell", list), observed_mean=("mean_per_cell", "mean"))
        .reset_index()
    )

    # Reformat expected dataset
    iterated_triangulation_distances_long = add_missing_columns(
        iterative_triangulation_distances, metadata, shared_column=region
    )

    expected_distances = (
        iterated_triangulation_distances_long.query("mean_dist <= @distance_threshold")
        .groupby(["celltype1", "celltype2", comparison])
        .agg(expected=("mean_dist", list), expected_mean=("mean_dist", "mean"))
        .reset_index()
    )

    # Drop comparisons with low numbers of observations
    observed_distances["keep"] = observed_distances["observed"].apply(
        lambda x: len(x) > min_observed
    )
    observed_distances = observed_distances[observed_distances["keep"]]

    expected_distances["keep"] = expected_distances["expected"].apply(
        lambda x: len(x) > min_observed
    )
    expected_distances = expected_distances[expected_distances["keep"]]

    # concatenate observed and expected distances
    distance_pvals = expected_distances.merge(
        observed_distances, on=["celltype1", "celltype2", comparison], how="left"
    )

    distance_pvals = expected_distances.merge(
        observed_distances, on=["celltype1", "celltype2", comparison], how="left"
    )
    distance_pvals["pvalue"] = distance_pvals.apply(calculate_pvalue, axis=1)
    distance_pvals["logfold_group"] = np.log2(
        distance_pvals["observed_mean"] / distance_pvals["expected_mean"]
    )
    distance_pvals["interaction"] = (
        distance_pvals["celltype1"] + " --> " + distance_pvals["celltype2"]
    )

    # drop na from distance_pvals
    # distance_pvals = distance_pvals.dropna()

    return distance_pvals


# Function for patch identification
## Adjust clustering parameter to get the desired number of clusters
def apply_dbscan_clustering(df, min_samples=10):
    """
    Apply DBSCAN clustering to a dataframe and update the cluster labels in the original dataframe.

    Args:
    df (pandas.DataFrame): The dataframe to be clustered.
    eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    None
    """
    # Initialize a new column for cluster labels
    df["cluster"] = -1

    # Apply DBSCAN clustering
    hdbscan = HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=5,
        cluster_selection_epsilon=0.0,
        max_cluster_size=None,
        metric="euclidean",
        alpha=1.0,
        cluster_selection_method="eom",
        allow_single_cluster=False,
    )
    labels = hdbscan.fit_predict(df[["x", "y"]])

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # Update the cluster labels in the original dataframe
    df.loc[df.index, "cluster"] = labels


# plot points and identify points in radius of selected points
def plot_selected_neighbors_with_shapes(
    full_df,
    selected_df,
    target_df,
    radius,
    plot=True,
    identification_column="community",
):
    # Get unique clusters from the full DataFrame
    unique_clusters = full_df[identification_column].unique()

    # DataFrame to store points within the circle but from a different cluster
    all_in_circle_diff_cluster = []

    # Loop through selected points
    for _, row in selected_df.iterrows():
        # Calculate distances from each point in the target DataFrame to the selected point
        distances = np.linalg.norm(
            target_df[["x", "y"]].values - np.array([row["x"], row["y"]]), axis=1
        )

        # Identify points within the circle and from a different cluster
        in_circle = distances <= radius
        diff_cluster = target_df[identification_column] != row[identification_column]
        in_circle_diff_cluster = target_df[in_circle & diff_cluster]

        # Append the result to the list
        all_in_circle_diff_cluster.append(in_circle_diff_cluster)

        # Plot the points with a different shape if plot is True
        if plot:
            plt.scatter(
                in_circle_diff_cluster["x"],
                in_circle_diff_cluster["y"],
                facecolors="none",
                edgecolors="red",
                marker="*",
                s=100,
                zorder=5,
                label="Cell within proximity",
            )

    # Concatenate the list of DataFrames into a single result DataFrame
    all_in_circle_diff_cluster = pd.concat(
        all_in_circle_diff_cluster, ignore_index=True
    )

    # Plot selected points in yellow and draw circles around them if plot is True
    if plot:
        plt.scatter(
            selected_df["x"],
            selected_df["y"],
            color="yellow",
            label="Selected Points",
            s=100,
            edgecolor="black",
            zorder=6,
        )
        for _, row in selected_df.iterrows():
            circle = plt.Circle(
                (row["x"], row["y"]),
                radius,
                color="red",
                fill=False,
                linestyle="--",
                alpha=0.5,
            )
            plt.gca().add_patch(circle)

        # Set plot labels and title
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Points with Radius {radius} for Selected Points")
        plt.grid(True)
        plt.axis("equal")

        # Place the legend outside the plot
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(),
            by_label.keys(),
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        plt.tight_layout()
        plt.show()

    # Remove duplicates from the final DataFrame
    all_in_circle_diff_cluster = all_in_circle_diff_cluster.drop_duplicates()

    return all_in_circle_diff_cluster


# generate a dataframe with the points in proximity to the selected points
## points are selected based on the coordinates of a 2D concave hull
### set edge_neighbours to 3 to select the 3 nearest points to the hull or 1 to disable functionality
def identify_points_in_proximity(
    df,
    full_df,
    identification_column,
    cluster_column="cluster",
    x_column="x",
    y_column="y",
    radius=200,
    edge_neighbours=3,
    plot=True,
    concave_hull_length_threshold=50,
):
    result_list = []

    # Loop through clusters in the DataFrame
    for cluster in set(df[cluster_column]) - {-1}:
        # Filter DataFrame for the current cluster
        subset = df.loc[df[cluster_column] == cluster]
        points = subset[[x_column, y_column]].values

        # Compute concave hull indexes
        idxes = concave_hull_indexes(
            points[:, :2],
            length_threshold=concave_hull_length_threshold,
        )

        # Get hull points from the DataFrame
        hull_points = pd.DataFrame(points[idxes], columns=["x", "y"])

        # Find nearest neighbors of hull points in the original DataFrame
        nbrs = NearestNeighbors(n_neighbors=edge_neighbours).fit(
            df[[x_column, y_column]]
        )
        distances, indices = nbrs.kneighbors(hull_points[["x", "y"]])

        hull_nearest_neighbors = df.iloc[indices.flatten()]

        # Plot selected neighbors and get the DataFrame with different clusters in the circle
        prox_points = plot_selected_neighbors_with_shapes(
            full_df=full_df,
            selected_df=hull_nearest_neighbors,
            target_df=full_df,
            radius=radius,
            plot=plot,
            identification_column=identification_column,
        )

        # Add a 'patch_id' column to identify the cluster
        prox_points["patch_id"] = cluster

        # Append the result to the list
        result_list.append(prox_points)

    # Concatenate the list of DataFrames into a single result DataFrame
    if len(result_list) > 0:
        result = pd.concat(result_list)
    else:
        result = pd.DataFrame(columns=["x", "y", "patch_id", identification_column])

    return result


# This function answers the what is in proximity of this group.
def patch_proximity_analysis(
    adata,
    region_column,
    patch_column,
    group,
    min_samples=80,
    x_column="x",
    y_column="y",
    radius=128,
    edge_neighbours=3,
    plot=True,
    savefig=False,
    output_dir="./",
    output_fname="",
    key_name="ppa_result",
):
    df = adata.obs

    for col in df.select_dtypes(["category"]).columns:
        df[col] = df[col].astype(str)

    # list to store results for each region
    region_results = []

    for region in df[region_column].unique():
        df_region = df[df[region_column] == region].copy()

        df_community = df_region[df_region[patch_column] == group].copy()

        apply_dbscan_clustering(df_community, min_samples=min_samples)

        # plot patches
        if plot:
            df_filtered = df_community[df_community["cluster"] != -1]
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(
                df_filtered["x"],
                df_filtered["y"],
                c=df_filtered["cluster"],
                cmap="tab20",
                alpha=0.5,
            )
            ax.set_title(f"DBSCAN Clusters for {region}_{group}")
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.grid(True)
            ax.axis("equal")
            if savefig:
                fig.savefig(
                    output_dir + output_fname + "_patch_proximity.pdf",
                    bbox_inches="tight",
                )
            else:
                plt.show()

        results = identify_points_in_proximity(
            df=df_community,
            full_df=df_region,
            cluster_column="cluster",
            identification_column=patch_column,
            x_column=x_column,
            y_column=y_column,
            radius=radius,
            edge_neighbours=edge_neighbours,
            plot=plot,
        )

        print(f"Finished {region}_{group}")

        # append to region_results
        region_results.append(results)

    # Concatenate all results into a single DataFrame
    final_results = pd.concat(region_results)

    # generate new column named unique_patch_ID that combines the region, group and patch ID
    final_results["unique_patch_ID"] = (
        final_results[region_column]
        + "_"
        + final_results[patch_column]
        + "_"
        + "patch_no_"
        + final_results["patch_id"].astype(str)
    )

    adata.uns[key_name] = final_results

    return final_results


def ml_train(
    adata_train,
    label,
    test_size=0.33,
    random_state=0,
    model="svm",
    nan_policy_y="raise",
    showfig=True,
):
    X = pd.DataFrame(adata_train.X)
    y = adata_train.obs[label].values

    if nan_policy_y == "omit":
        y_msk = ~y.isna()
        X = X[y_msk]
        y = y[y_msk]
    elif nan_policy_y == "raise":
        pass
    else:
        raise ValueError("nan_policy_y must be either 'omit' or 'raise'")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(y.unique().sort_values())

    print("Training now!")
    svc = SVC(kernel="linear", probability=True)
    svc.fit(X_train, y_train)
    pred = []
    y_prob = svc.predict_proba(X_test)
    y_prob = pd.DataFrame(y_prob)
    y_prob.columns = svc.classes_

    svm_label = y_prob.idxmax(axis=1, skipna=True)
    target_names = svc.classes_
    print("Evaluating now!")
    svm_eval = classification_report(
        y_true=y_test, y_pred=svm_label, target_names=target_names, output_dict=True
    )
    if showfig:
        sns.heatmap(pd.DataFrame(svm_eval).iloc[:-1, :].T, annot=True)
        plt.show()

    return svc


def ml_predict(adata_val, svc, save_name="svm_pred", return_prob_mat=False):
    print("Classifying!")
    X_val = pd.DataFrame(adata_val.X)
    y_prob_val = svc.predict_proba(X_val)
    y_prob_val = pd.DataFrame(y_prob_val)
    y_prob_val.columns = svc.classes_
    svm_label_val = y_prob_val.idxmax(axis=1, skipna=True)
    svm_label_val.index = X_val.index
    print("Saving cell type labels to adata!")
    adata_val.obs[save_name] = svm_label_val.values
    if return_prob_mat:
        print("Returning probability matrix!")
        y_prob_val.columns = svc.classes_
        svm_label_val = y_prob_val.idxmax(axis=1, skipna=True)
        return svm_label_val
