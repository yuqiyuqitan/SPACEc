# this file contains legacy code that is no longer used in the project
# it is kept here for reference purposes

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
            
            

###############
# Patch analysis





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