import pathlib
from tempfile import TemporaryDirectory

TEST_DIR = pathlib.Path(__file__).parent


def test_qptiff_converter():
    import matplotlib

    matplotlib.use("Agg")

    print("TEST_DIR:", TEST_DIR)
    data_path = TEST_DIR / "data"

    with TemporaryDirectory() as output_dir:
        data_dir = data_path / "raw"
        output_path = pathlib.Path(output_dir)
        file = "S1121_Scan1_small.qptiff"
        file_path = data_dir / file

        import spacec as sp

        # # 0.1 Downscale CODEX images
        # First, specify parameters and load image. A histogram of marker expression levels should appear. Select a lower_cutoff value very close to background and an upper_cutoff value in the range of positive expression and input these values into the lower_cutoff and upper_cutoff parameters in the following cell.
        resized_im = sp.hf.downscale_tissue(file_path=file_path)

        # %% [markdown]
        # # 0.2 Segment individual tissue pieces

        # %%
        tissueframe = sp.tl.label_tissue(
            resized_im, lower_cutoff=0.012, upper_cutoff=0.02
        )

        # %%
        tissueframe.head()

        # # 0.3 Rename tissue number (optional)

        # Optional: manually clean up automatic tissue region assignments. A tissue region often consists of multiple pieces of tissue that are not connected. Occasionally, the above algorithm will assign a piece of tissue to the incorrect region. The next two cells allow the user to manually correct such region mislabelings.
        #
        # Running the first cell generates two plots. The first plot shows the tissue piece labels. The second plot shows the final region assignments for the tissue pieces. These two plots can be used to identify the ID of the tissue piece you want to reassign and the region ID you want to assign it to.
        #
        # The second cell takes two parameters:
        #
        # tissue_id - (int) this is the tissue ID of the piece of tissue you want to reassign
        #
        # new_region_id - (int) this is the ID of the new region you want to assign this tissue piece to
        #
        # Running the second cell relabels the region assignment of the specified tissue piece.

        # %%
        sp.pl.tissue_lables(tissueframe=tissueframe, region="region1")

        # %%
        # Rename the regions based on annotations
        rename_dict = {1: 1}

        for k in rename_dict.keys():
            tissueframe["region1"][tissueframe["tissue"] == k] = rename_dict[k]

        # rename so ordered 1 through 8
        tiss_num = {
            list(tissueframe["region1"].unique())[i]: i + 1
            for i in range(len(tissueframe["region1"].unique()))
        }
        tissueframe["region"] = tissueframe["region1"].map(tiss_num).copy()
        tiss_num

        sp.pl.tissue_lables(tissueframe=tissueframe, region="region")

        tissueframe["region"].value_counts()

        # # 0.4 Extract labeled tissues into subimages into indiviudal tiffstack

        sp.tl.save_labelled_tissue(
            filepath=file_path, tissueframe=tissueframe, output_dir=str(output_path)
        )


if __name__ == "__main__":
    test_cell_segmentation()
