import pathlib
from tempfile import TemporaryDirectory

TEST_DIR = pathlib.Path(__file__).parent


def test_3_clustering():
    # ## Set up environment

    # import standard packages
    import pathlib

    import matplotlib
    import pandas as pd
    import scanpy as sc

    import spacec as sp

    matplotlib.use("Agg")

    # data_path = TEST_DIR / 'data' # where the data is stored
    overlay_path = TEST_DIR / "data/processed/tonsil/1"

    with TemporaryDirectory() as output_dir:
        # output_dir = pathlib.Path("tests/_out")
        # output_dir.mkdir(exist_ok=True, parents=True)

        output_path = pathlib.Path(output_dir)

        sc.settings.set_figure_params(dpi=80, facecolor="white")

        # Loading the denoise/filtered anndata from notebook 2
        adata = sc.read(overlay_path / "adata_nn_demo.h5ad")

        # ## 3.1 Clustering

        clustering_random_seed = 0

        # This step can be long if you have large phenocycler images

        # Use this cell-type specific markers for cell type annotation
        marker_list = [
            "FoxP3",
            "HLA-DR",
            "EGFR",
            "CD206",
            "BCL2",
            "panCK",
            "CD11b",
            "CD56",
            "CD163",
            "CD21",
            "CD8",
            "Vimentin",
            "CCR7",
            "CD57",
            "CD34",
            "CD31",
            "CXCR5",
            "CD3",
            "CD38",
            "LAG3",
            "CD25",
            "CD16",
            "CLEC9A",
            "CD11c",
            "CD68",
            "aSMA",
            "CD20",
            "CD4",
            "Podoplanin",
            "CD15",
            "betaCatenin",
            "PAX5",
            "MCT",
            "CD138",
            "GranzymeB",
            "IDO-1",
            "CD45",
            "CollagenIV",
            "Arginase-1",
        ]

        # interative clustering
        sp.tl.launch_interactive_clustering(adata=adata, output_dir="output_path")

        # clustering
        adata = sp.tl.clustering(
            adata,
            clustering="leiden",  # can choose between leiden and louvian
            n_neighbors=10,
            resolution=0.4,  # clustering resolution
            reclustering=False,  # if true, no computing the neighbors
            marker_list=marker_list,  # if it is None, all variable names are used for clustering
            # seed=clustering_random_seed
        )

        # visualization of clustering with UMAP
        sc.pl.umap(adata, color=["leiden_0.4", "unique_region"], wspace=0.5)

        sc.pl.dotplot(adata, marker_list, "leiden_0.4", dendrogram=True)

        # tentative annotation based on the marker
        cluster_to_ct_dict = {
            "0": "B_CD4_vessel_mix",  # *further subcluster
            "1": "B",
            "2": "B",  # *DC candidate
            "3": "CD4_CD8T_mix",  # *further subcluster
            "4": "CD4_CD8T_mix",  # *further subcluster
            "5": "GCB",
            "6": "epithelia",
            "7": "DC",
            "8": "Vessel",
            "9": "Plasma",
            "10": "M2",
            "11": "CD4T",  # * Treg candidate
            "12": "cDC1",
            "13": "M1",
            "14": "Mast cell",
        }

        # ## 3.3 Sub-clustering (optional)

        # ### Round 1 subclustering

        # subclustering cluster 0, 3, 4 sequentially (could be optional for your own data)
        sc.tl.leiden(
            adata,
            seed=clustering_random_seed,
            restrict_to=("leiden_0.4", ["0"]),
            resolution=0.15,
            key_added="leiden_0.4_subcluster_0",
        )
        sc.tl.leiden(
            adata,
            seed=clustering_random_seed,
            restrict_to=("leiden_0.4_subcluster_0", ["3"]),
            resolution=0.1,
            key_added="leiden_0.4_subcluster_3",
        )
        sc.tl.leiden(
            adata,
            seed=clustering_random_seed,
            restrict_to=("leiden_0.4_subcluster_3", ["4"]),
            resolution=0.1,
            key_added="leiden_0.4_subcluster_4",
        )
        sc.pl.umap(adata, color=["leiden_0.4_subcluster_4"])

        sc.pl.dotplot(adata, marker_list, "leiden_0.4_subcluster_4", dendrogram=True)

        # annotate the clusters based on marker gene expression
        # ML annotation of clusters is coming up
        cluster_to_ct_dict = {
            "0,0": "B",
            "0,1": "CD4T",
            "1": "B",
            "2": "B",
            "3,0": "CD4T",
            "3,1": "CD8T",
            "4,0": "CD4T",
            "4,1": "CD8T",
            "5": "GCB",
            "6": "epithelia",
            "7": "DC",
            "8": "Vessel",
            "9": "Plasma",
            "10": "M2",
            "11": "Treg",
            "12": "cDC1",
            "13": "M1",
            "14": "Mast cell",
        }
        adata.obs["celltype"] = (
            adata.obs["leiden_0.4_subcluster_4"]
            .map(cluster_to_ct_dict)
            .astype("category")
        )

        # just to check if the celltype conversion is successful
        sc.pl.umap(adata, color=["leiden_0.4_subcluster_4", "celltype"], wspace=0.5)

        # ### Round 2 subclustering

        ## subclustering cluster 0, 3, 4 sequentially (could be optional for your own data)
        sc.tl.leiden(
            adata,
            seed=clustering_random_seed,
            restrict_to=("leiden_0.4_subcluster_4", ["2"]),
            resolution=0.4,
            key_added="leiden_0.4_subcluster_2",
        )  # 2,4 DC; 2,0 B

        sc.tl.leiden(
            adata,
            seed=clustering_random_seed,
            restrict_to=("leiden_0.4_subcluster_2", ["5"]),
            resolution=0.1,
            key_added="leiden_0.4_subcluster_5",
        )
        sc.tl.leiden(
            adata,
            seed=clustering_random_seed,
            restrict_to=("leiden_0.4_subcluster_3", ["0,0"]),
            resolution=0.4,
            key_added="leiden_0.4_subcluster_0sub",
        )  # 0,0,3 noise, 0,0,1 CD4T, 0.0,4 vessel
        sc.pl.dotplot(adata, marker_list, "leiden_0.4_subcluster_0sub")

        # ## 3.4 Annotate cell types

        # annotate the clusters based on marker gene expression
        # ML annotation of clusters is coming up
        cluster_to_ct_dict = {
            "0,0,0": "B",
            "0,0,1": "CD4T",
            "0,0,2": "B",
            "0,0,3": "noise",
            "0,0,4": "Vessel",
            "0,0,5": "Vessel",
            "0,1": "CD4T",
            "1": "B",
            "2": "B",
            "2,0": "B",
            "2,1": "B",
            "2,2": "B",
            "2,3": "B",
            "2,4": "DC",
            "2,5": "B",
            "2,6": "B",
            "3,0": "CD4T",
            "3,1": "CD8T",
            "4": "CDX",
            "4,0": "CD4T",
            "4,1": "CD8T",
            "4,2": "CD8T",
            "5": "GCB",
            "6": "epithelia",
            "7": "DC",
            "8": "Vessel",
            "9": "Plasma",
            "10": "M2",
            "11,0": "Treg",
            "11,1": "CD4T",
            "11,2": "Treg",
            "12": "cDC1",
            "13": "M1",
            "14": "Mast cell",
        }
        adata.obs["celltype_fine"] = (
            adata.obs["leiden_0.4_subcluster_0sub"]
            .map(cluster_to_ct_dict)
            .astype("category")
        )

        # remove noise cell here
        adata = adata[adata.obs["celltype_fine"] != "noise"]

        # just to check if the celltype conversion is successful
        sc.pl.umap(
            adata, color=["leiden_0.4_subcluster_0sub", "celltype_fine"], wspace=0.5
        )

        # ### 3.4 Save AnnData

        adata.write(output_path / "adata_nn_demo_annotated.h5ad")

        # ## 3.5 Single-cell visualzation

        sp.pl.catplot(
            adata,
            color="celltype_fine",  # specify group column name here (e.g. celltype_fine)
            unique_region="condition",  # specify unique_regions here
            X="x",
            Y="y",  # specify x and y columns here
            n_columns=2,  # adjust the number of columns for plotting here (how many plots do you want in one row?)
            palette=None,  # default is None which means the color comes from the anndata.uns that matches the UMAP
            savefig=False,  # save figure as pdf
            output_fname="",  # change it to file name you prefer when saving the figure
            output_dir=output_path,  # specify output directory here (if savefig=True)
        )

        # cell type percentage tab and visualization [much few]
        ct_perc_tab, _ = sp.pl.stacked_bar_plot(
            adata=adata,  # adata object to use
            color="celltype_fine",  # column containing the categories that are used to fill the bar plot
            grouping="condition",  # column containing a grouping variable (usually a condition or cell group)
            cell_list=[
                "GCB",
                "Treg",
            ],  # list of cell types to plot, you can also see the entire cell types adata.obs['celltype_fine'].unique()
            palette=None,  # default is None which means the color comes from the anndata.uns that matches the UMAP
            savefig=False,  # change it to true if you want to save the figure
            output_fname="",  # change it to file name you prefer when saving the figure
            output_dir=output_path,  # output directory for the figure
            norm=False,  # if True, then whatever plotted will be scaled to sum of 1
        )

        sp.pl.create_pie_charts(
            adata,
            color="celltype_fine",
            grouping="condition",
            show_percentages=False,
            palette=None,  # default is None which means the color comes from the anndata.uns that matches the UMAP
            savefig=False,  # change it to true if you want to save the figure
            output_fname="",  # change it to file name you prefer when saving the figure
            output_dir=output_path,  # output directory for the figure
        )

        # ## Additional visualization

        # One can also use tissuemap to visualize the cell location in original CODEX space and create tissue based annotation
        df = pd.DataFrame(adata.obs)  # make sure there are x and y in the dataframe
        df.to_csv(output_path / "adata_nn_demo_annotated.csv")


if __name__ == "__main__":
    test_3_clustering()
