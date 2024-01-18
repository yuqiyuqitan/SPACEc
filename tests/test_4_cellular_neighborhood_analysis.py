import pathlib
from tempfile import TemporaryDirectory

TEST_DIR = pathlib.Path(__file__).parent


def test_4_cellular_neighborhood_analysis():

    # Set up environment
    import scanpy as sc
    import spacec as sp
    
    import matplotlib
    matplotlib.use('Agg')

    processed_path = TEST_DIR / "data/processed/tonsil/1"

    with TemporaryDirectory() as output_dir:

        sc.settings.set_figure_params(dpi=80, facecolor='white')

        # output_dir = pathlib.Path("tests/_out")
        # output_dir.mkdir(exist_ok=True, parents=True)

        output_path = pathlib.Path(output_dir)


        # Loading the denoise/filtered anndata from notebook 3 [cell type or cluster annotation is necessary for the step]
        adata = sc.read(processed_path / 'adata_nn_demo_annotated.h5ad')
        adata

        # compute for CNs
        # tune k and n_neighborhoods to obtain the best result
        adata = sp.tl.neighborhood_analysis_ad(
            adata,
            unique_region = "unique_region", 
            cluster_col = "celltype_fine", 
            X = 'x', Y = 'y',
            k = 20, # k nearest neighbors
            n_neighborhoods = 6, #number of CNs
            elbow = False)

        # to better visualize the CN, we choose a CN palette
        # but if you set palette = None in the following function, it will randomly generate a palette for you
        cn_palette = {
            0: '#204F89',
            1: '#3C5FD7',
            2: '#829868',
            3: '#FDA9AA',
            4: '#E623B1',
            5: '#44CB63'
        }

        # plot CN to see what cell types are enriched per CN so that we can annotate them better
        sp.pl.cn_exp_heatmap_ad(
            adata, 
            cluster_col = "celltype_fine",
            cn_col = "CN_k20_n6",
            palette=cn_palette,
            figsize=(10,8),
            savefig = False,
            output_dir = output_path,
            rand_seed = 1
        )

        # %%
        # plot the CN in the spatial coordinates, using the same color palette
        sp.pl.catplot_ad(
            adata, 
            color = "CN_k20_n6", 
            unique_region = "unique_region",
            X='x', Y='y',
            palette= cn_palette, 
            savefig=False,
            output_dir=output_path,
        )

        # %%
        # Define neighborhood annotation for every cluster ID
        neighborhood_annotation = {
            0: 'Immune Priming Zone',
            1: 'Parafollicular T cell Zone',
            2: 'Marginal Zone',
            3: 'Germinal center',
            4: 'Marginal Zone B-DC Enriched',# we dont have the DC markers to separate the inner pink from the outer pink; but they are both encountering immature B cells
            5: "Epithelium", 
        }

        adata.obs['CN_k20_n6_annot'] = (
            adata.obs['CN_k20_n6']
            .map(neighborhood_annotation)
            .astype('category')
        )

        # %%
        # match the color of the annotated CN to the original CN
        cn_annt_palette = {
            neighborhood_annotation[key]: value 
            for key, value in cn_palette.items()}

        # replotting with CN annotation
        sp.pl.cn_exp_heatmap_ad(
            adata, 
            cluster_col = "celltype", 
            cn_col = "CN_k20_n6_annot",
            palette = cn_annt_palette, #if None, there is randomly generated in the code 
            savefig=True,
            figsize=(12,10),
            output_fname = "",
            output_dir = output_path,
        )

        # %%
        adata.write_h5ad(output_path / 'adata_nn_demo_annotated_cn.h5ad')

        # %% [markdown]
        # ## 4.2 Spatial context maps
        # To plot the spatial context between cellular neighborhoods or communities spatial context maps are a useful visualization. The analysis uses a similar sliding window approach as used for the detection of CNs/Communities but takes these broader groups as input. The resulting vectors are used to analyze which CNs/Communities tend to form interfaces. These interfaces are shown in the graph. Colored squares show single CNs or combinations of them, edges connect parent nodes with daughter nodes. The black circles indicate the abundance of cells falling into these spatial groups. 

        # %%
        # We will look at the spatial context maps sepataely for each condition
        adata_tonsil = adata[adata.obs['condition'] == 'tonsil']
        adata_tonsillitis = adata[adata.obs['condition'] == 'tonsillitis']

        # %% [markdown]
        # #### tonsil

        # %%
        cnmap_dict_tonsil = sp.tl.cn_map_ad(
            adata = adata_tonsil, # adata object
            cn_col = "CN_k20_n6",# column with CNs
            palette = None, # color dictionary
            unique_region = 'region_num',# column with unique regions
            k = 70, # number of neighbors
            X='x', Y='y',  # coordinates
            threshold = 0.85, # threshold for percentage of cells in CN
            per_keep_thres = 0.85,) # threshold for percentage of cells in CN

        # %%
        # Compute for the frequency of the CNs and paly around with the threshold
        sp.pl.cn_map(
            cnmap_dict = cnmap_dict_tonsil,
            adata = adata_tonsil,
            cn_col = "CN_k20_n6",
            palette = None,
            figsize=(40, 20),
            savefig=False,
            output_fname = "",
            output_dir= output_path
        )

        # %% [markdown]
        # ### tonsillitis

        # %%
        cnmap_dict_tonsillitis = sp.tl.cn_map_ad(
            adata = adata_tonsillitis, # adata object
            cn_col = "CN_k20_n6",# column with CNs
            palette = None, # color dictionary
            unique_region = 'region_num',# column with unique regions
            k = 70, # number of neighbors
            X='x', Y='y',  # coordinates
            threshold = 0.85, # threshold for percentage of cells in CN
            per_keep_thres = 0.85,) # threshold for percentage of cells in CN

        # %%
        sp.pl.cn_map(
            cnmap_dict = cnmap_dict_tonsillitis,
            adata = adata_tonsillitis,
            cn_col = "CN_k20_n6",
            palette = None,
            figsize=(40, 20),
            savefig=False,
            output_fname = "",
            output_dir= output_path
        )

if __name__ == '__main__':
    test_4_cellular_neighborhood_analysis()

