import pathlib
from tempfile import TemporaryDirectory

TEST_DIR = pathlib.Path(__file__).parent


def test_5_distance_permutation_analysis():
    # Set up environment
    import matplotlib
    import scanpy as sc

    import spacec as sp

    matplotlib.use("Agg")

    processed_path = TEST_DIR / "data/processed/tonsil/1"

    with TemporaryDirectory() as output_dir:
        sc.settings.set_figure_params(dpi=80, facecolor="white")

        # %%
        # Load data
        adata = sc.read(processed_path / "adata_nn_2000.h5ad")
        adata

        # %% [markdown]
        # ## 5.1 Identify potential interactions

        # %%
        distance_pvals, triangulation_distances_dict = sp.tl.identify_interactions(
            adata=adata,  # AnnData object
            cellid="index",  # column that contains the cell id (set index if the cell id is the index of the dataframe)
            x_pos="x",  # x coordinate column
            y_pos="y",  # y coordinate column
            cell_type="cell_type",  # column that contains the cell type information
            region="unique_region",  # column that contains the region information
            num_iterations=100,  # number of iterations for the permutation test
            num_cores=1,  # number of CPU threads to use
            min_observed=10,  # minimum number of observed interactions to consider a cell type pair
            comparison="condition",  # column that contains the condition information we want to compare
            distance_threshold=20 / 0.5085,
        )  # distance threshold in px (20 µm)

        distance_pvals.head()

        distance_pvals_filt = sp.tl.remove_rare_cell_types(
            adata,
            distance_pvals,
            cell_type_column="cell_type",
            min_cell_type_percentage=1,
        )

        # %%
        # Identify significant cell-cell interactions
        # dist_table_filt is a simplified table used for plotting
        # dist_data_filt contains the filtered raw data with more information about the pairs
        dist_table_filt, dist_data_filt = sp.tl.filter_interactions(
            distance_pvals=distance_pvals_filt,
            pvalue=0.05,
            logfold_group_abs=0.1,
            comparison="condition",
        )

        print(dist_table_filt.shape)
        dist_data_filt.head()

        sp.pl.plot_top_n_distances(
            dist_table_filt,
            dist_data_filt,
            n=5,
            colors=None,
            dodge=False,
            savefig=False,
            output_fname="",
            output_dir="./",
            figsize=(5, 5),
            unit="px",
            errorbars=True,
        )

        # %%
        sp.pl.dumbbell(
            data=dist_table_filt, figsize=(8, 12), colors=["#DB444B", "#006BA2"]
        )


if __name__ == "__main__":
    test_5_distance_permutation_analysis()
