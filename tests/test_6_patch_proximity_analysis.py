import pathlib
from tempfile import TemporaryDirectory

TEST_DIR = pathlib.Path(__file__).parent


def test_6_patch_proximity_analysis():

    # Set up environment
    import scanpy as sc
    import spacec as sp

    import matplotlib
    matplotlib.use('Agg')
    sc.settings.set_figure_params(dpi=80, facecolor='white')

    processed_path = TEST_DIR / "data/processed/tonsil/1"

    adata = sc.read(processed_path / 'adata_nn_demo_annotated_cn.h5ad')
    adata

    # %%
    # this region result is also saved to adata.uns
    region_results = sp.tl.tl_patch_proximity_analysis(
        adata, 
        region_column = "unique_region", 
        patch_column = "CN_k20_n6_annot", 
        group="Germinal center",
        min_samples=5, # TODO: this break if we don't have no clusters!
        x_column='x', y_column='y', 
        radius = 128,
        edge_neighbours = 3, 
        key_name = 'ppa_result',
        plot = True)

    # %%
    # plot the result to see the cell types enriched around the edge of the patches
    sp.pl.count_patch_proximity_res(
        adata, 
        x="celltype_fine", 
        hue="condition", 
        palette="Set3",
        order = True,
        key_name = 'ppa_result',
        savefig = False)


if __name__ == '__main__':
    test_6_patch_proximity_analysis()