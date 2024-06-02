import pathlib
from tempfile import TemporaryDirectory

TEST_DIR = pathlib.Path(__file__).parent


def test_6_patch_proximity_analysis():
    # Set up environment
    import matplotlib
    import scanpy as sc

    import spacec as sp

    matplotlib.use("Agg")
    sc.settings.set_figure_params(dpi=80, facecolor="white")

    processed_path = TEST_DIR / "data/processed/tonsil/1"

    adata = sc.read(processed_path / "adata_nn_2000.h5ad")
    adata

    # %%
    # this region result is also saved to adata.uns
    # this region result is also saved to adata.uns
    region_results = sp.tl.patch_proximity_analysis(
        adata, 
        region_column = "unique_region", 
        patch_column = "CN_k20_n6_annot", 
        group="Marginal Zone",
        min_cluster_size=5, 
        x_column='x', y_column='y', 
        radius = (1000), # to get the distance in µm
        edge_neighbours = 3,
        key_name = 'ppa_result_50',
        plot = True)

    # this region result is also saved to adata.uns
    region_results = sp.tl.patch_proximity_analysis(
        adata, 
        region_column = "unique_region", 
        patch_column = "CN_k20_n6_annot", 
        group="Marginal Zone",
        min_cluster_size=5, 
        x_column='x', y_column='y', 
        radius = (1000), # to get the distance in µm
        edge_neighbours = 3,
        key_name = 'ppa_result_100',
        plot = False)

    # this region result is also saved to adata.uns
    region_results = sp.tl.patch_proximity_analysis(
        adata, 
        region_column = "unique_region", 
        patch_column = "CN_k20_n6_annot", 
        group="Marginal Zone",
        min_cluster_size=5, 
        x_column='x', y_column='y', 
        radius = (1000), # to get the distance in µm
        edge_neighbours = 3,
        key_name = 'ppa_result_150',
        plot = False)

    # this region result is also saved to adata.uns
    region_results = sp.tl.patch_proximity_analysis(
        adata, 
        region_column = "unique_region", 
        patch_column = "CN_k20_n6_annot", 
        group="Marginal Zone",
        min_cluster_size=5, 
        x_column='x', y_column='y', 
        radius = (1000), # to get the distance in µm
        edge_neighbours = 3,
        key_name = 'ppa_result_200',
        plot = False)

    # this region result is also saved to adata.uns
    region_results = sp.tl.patch_proximity_analysis(
        adata, 
        region_column = "unique_region", 
        patch_column = "CN_k20_n6_annot", 
        group="Marginal Zone",
        min_cluster_size=5, 
        x_column='x', y_column='y', 
        radius = (1000), # to get the distance in µm
        edge_neighbours = 3,
        key_name = 'ppa_result_250',
        plot = False)

    # %%
    # plot the result to see the cell types enriched around the edge of the patches
    sp.pl.ppa_res_donut(adata, 
                palette=None,
                cat_col = "cell_type",
                key_names = ['ppa_result_50', 'ppa_result_100', 'ppa_result_150', 'ppa_result_200', 'ppa_result_250'],
                radii = [5, 10, 15, 20, 25],
                unit = 'µm',
                figsize = (10,10),  
                add_guides = True,
                text = 'Cell types around Marginal Zone',
                label_color='white',
                subset_column = 'condition',
                subset_condition = 'tonsil',
                title='Tonsil patch proximity analysis')


if __name__ == "__main__":
    test_6_patch_proximity_analysis()
