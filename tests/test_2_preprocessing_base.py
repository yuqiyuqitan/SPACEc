import pathlib
from tempfile import TemporaryDirectory

TEST_DIR = pathlib.Path(__file__).parent


def test_2_preprocessing():
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")

    import spacec as sp

    data_path = TEST_DIR / "data"  # where the data is stored
    overlay_path = TEST_DIR / "data/processed/tonsil/1"

    # where you want to store the output
    with TemporaryDirectory() as output_dir:
        # output_dir = pathlib.Path("tests/_out")
        # output_dir.mkdir(exist_ok=True, parents=True)

        output_path = pathlib.Path(output_dir)

        # ## 2.1 Load data

        # Read and concatenate the csv files (outputs from the cell segmentation algorithms).

        # read in segmentation csv files
        df_seg = sp.pp.read_segdf(
            segfile_list=[
                data_path / "processed/cellseg/reg010_X01_Y01_Z01_compensated.csv",
                data_path / "processed/cellseg/reg001_X01_Y01_Z01_compensated.csv",
            ],
            seg_method="cellseg",
            region_list=["reg010", "reg001"],
            meta_list=["tonsillitis", "tonsil"],
        )

        # 2.2 Filter cells by DAPI intensity and area

        # Identify the lowest 1% for cell size and nuclear marker intensity to get a better idea of potential segmentation artifacts.

        # print smallest 1% of cells by area
        one_percent_area = np.percentile(df_seg.area, 1)

        # If necessary filter the dataframe to remove too small objects or cells without a nucleus.

        df_filt = sp.pp.filter_data(
            df_seg,
            nuc_thres=1,
            size_thres=one_percent_area,
            nuc_marker="DAPI",
            cell_size="area",
            region_column="region_num",
            color_by="region_num",
            log_scale=False,
        )

        # Normalize data with one of the four available methods (zscore as default)

        # This is to normalize the data
        dfz = sp.pp.format(
            data=df_filt,
            list_out=["cell_id", "tile_num", "z", "x_tile", "y_tile"],
            # in case of other segmentation methods: ['eccentricity', 'perimeter', 'convex_area', 'axis_major_length', 'axis_minor_length', "first_index", "filename", "label"]
            list_keep=[
                "DAPI",
                "x",
                "y",
                "area",
                "region_num",
                "region",
                "unique_region",
                "condition",
            ],  # This is a list of meta information that you would like to keep but don't want to normalize
            method="zscore",
        )  # choose from "zscore", "double_zscore", "MinMax", "ArcSin"

        # ## 2.4 Remove noisy cells

        # This section is used to remove noisy cells. This is very important to ensure proper identification of the cells via clustering.

        # get the column index for the last marker
        col_num_last_marker = dfz.columns.get_loc("GATA3")

        # This function helps to figure out what the cut-off should be
        # This is to remove top 1 % of all cells that are highly expressive for all antibodies
        sp.pl.zcount_thres(
            dfz=dfz,
            col_num=col_num_last_marker,  # last antibody index
            cut_off=0.01,  # top 1% of cells
            count_bin=50,
        )

        # This step removes the remaining noisy cells from the analysis

        df_nn, cc = sp.pp.remove_noise(
            df=dfz,
            col_num=col_num_last_marker,  # this is the column index that has the last protein feature
            z_count_thres=51,  # number obtained from the function above
            z_sum_thres=63,  # number obtained from the function above
        )

        # ## 2.5 Save denoise data

        # #### Save as dataframe

        # Save the df as a backup. We strongly recommend the Anndata format for further analysis!
        df_nn.to_csv(output_path / "df_nn_demo.csv")

        # #### Save as anndata

        # inspect which markers work, and drop the ones that did not work from the clustering step
        # make an anndata to be compatible with the downstream clustering step
        adata = sp.hf.make_anndata(
            df_nn=df_nn,
            col_sum=col_num_last_marker,  # this is the column index that has the last protein feature # the rest will go into obs
            nonFuncAb_list=[],  # Remove the antibodies that are not working
        )

        # save the anndata object to a file
        adata.write_h5ad(output_path / "adata_nn_demo.h5ad")

        # ## 2.6 Show the spatial distribution for size (Optional)

        import pickle

        with open(overlay_path / "overlay_tonsil1.pickle", "rb") as f:
            overlay_data1 = pickle.load(f)

        sp.pl.coordinates_on_image(
            df=df_nn.loc[df_nn["unique_region"] == "reg010", :],
            overlay_data=overlay_data1,
            color="area",
            scale=False,  # whether to scale to 1 or not
            dot_size=2,
            convert_to_grey=True,
            fig_width=5,
            fig_height=5,
        )


if __name__ == "__main__":
    test_2_preprocessing()
