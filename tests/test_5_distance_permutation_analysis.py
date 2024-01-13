import pathlib
from tempfile import TemporaryDirectory

TEST_DIR = pathlib.Path(__file__).parent


def test_5_distance_permutation_analysis():

    # Set up environment
    import scanpy as sc
    import spacec as sp

    import matplotlib
    matplotlib.use('Agg')

    processed_path = TEST_DIR / "data/processed/tonsil/1"

    with TemporaryDirectory() as output_dir:
        sc.settings.set_figure_params(dpi=80, facecolor='white')

        # %%
        # Load data
        adata = sc.read(processed_path / "adata_nn_demo_annotated_cn.h5ad")
        adata

        # %% [markdown]
        # ## 5.1 Identify potential interactions

        # %%
        distance_pvals = sp.tl.tl_identify_interactions_ad(
            adata = adata, 
            id = "index", 
            x_pos = "x", 
            y_pos = "y", 
            cell_type = "celltype", 
            region = "unique_region",
            num_iterations=100,
            num_cores=10, 
            min_observed = 10,
            comparison = 'condition')
        distance_pvals.head()

        # %%
        # Identify significant cell-cell interactions
        # dist_table_filt is a simplified table used for plotting
        # dist_data_filt contains the filtered raw data with more information about the pairs
        dist_table_filt, dist_data_filt = sp.tl.tl_filter_interactions(
            distance_pvals = distance_pvals,
            pvalue = 0.05,
            logfold_group_abs = 0.1)

        print(dist_table_filt.shape)
        dist_data_filt.head()

        # %%
        sp.pl.pl_dumbbell(
            data = dist_table_filt, figsize=(10,10), colors = ['#DB444B', '#006BA2'])


if __name__ == '__main__':
    test_5_distance_permutation_analysis()