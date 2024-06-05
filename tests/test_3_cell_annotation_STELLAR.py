import pathlib
from tempfile import TemporaryDirectory

TEST_DIR = pathlib.Path(__file__).parent


def test_3_cell_annotation_STELLAR():
    # import standard packages

    # import standard packages
    import os
    import sys
    from pathlib import Path

    import anndata
    import matplotlib.pyplot as plt
    import pandas as pd
    import scanpy as sc
    import seaborn as sns
    from git import Repo

    # %%
    import spacec as sp

    # %%
    root_path = pathlib.Path("..")

    data_path = TEST_DIR / "data"  # where the data is stored
    processed_path = TEST_DIR / "data/processed/tonsil/1"

    # where you want to store the output
    with TemporaryDirectory() as output_dir:
        # output_dir = pathlib.Path("tests/_out")
        # output_dir.mkdir(exist_ok=True, parents=True)

        output_path = pathlib.Path(output_dir)

        # ## Data Explanation
        # Annotated tonsil data is used as training & test data. </br>
        # Tonsillitis data is used as validation data.

        # Load training data
        adata = sc.read(processed_path / "adata_nn_2000.h5ad")
        adata_train = adata[adata.obs["condition"] == "tonsil"]
        adata_val = adata[adata.obs["condition"] == "tonsillitis"]

        # ## 3.1 Training

        import numpy as np

        np.isnan(adata_train.X).sum()

        root_path = TEST_DIR  # replace with your path

        # STELLAR path
        STELLAR_path = Path(root_path / "example_data/STELLAR/")

        # Test if the path exists, if not create it
        if not STELLAR_path.exists():
            STELLAR_path.mkdir(exist_ok=True, parents=True)
            repo_url = "https://github.com/snap-stanford/stellar.git"
            Repo.clone_from(repo_url, STELLAR_path)

        adata_new = sp.tl.adata_stellar(
            adata_train,
            adata_val,
            celltype_col="cell_type",
            x_col="x",
            y_col="y",
            sample_rate=0.5,
            distance_thres=50,
            STELLAR_path=STELLAR_path,
        )

        sp.pl.catplot(
            adata_new,
            color="stellar_pred",  # specify group column name here (e.g. celltype_fine)
            unique_region="condition",  # specify unique_regions here
            X="x",
            Y="y",  # specify x and y columns here
            n_columns=1,  # adjust the number of columns for plotting here (how many plots do you want in one row?)
            palette=None,  # default is None which means the color comes from the anndata.uns that matches the UMAP
            savefig=False,  # save figure as pdf
            output_fname="",  # change it to file name you prefer when saving the figure
            output_dir=output_dir,  # specify output directory here (if savefig=True)
        )

        sp.pl.create_pie_charts(
            adata_new,
            color="stellar_pred",
            grouping="condition",
            show_percentages=False,
            palette=None,  # default is None which means the color comes from the anndata.uns that matches the UMAP
            savefig=False,  # change it to true if you want to save the figure
            output_fname="",  # change it to file name you prefer when saving the figure
            output_dir=output_dir,  # output directory for the figure
        )


if __name__ == "__main__":
    test_3_cell_annotation_ml()
