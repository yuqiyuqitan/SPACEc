import pathlib
from tempfile import TemporaryDirectory

TEST_DIR = pathlib.Path(__file__).parent


def test_3_cell_annotation_ml():
    # import standard packages
    import os
    import pathlib

    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import scanpy as sc
    import seaborn as sns

    matplotlib.use("Agg")

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
        adata = sc.read(processed_path / "adata_nn_demo_annotated.h5ad")
        adata_train = adata[adata.obs["condition"] == "tonsil"]
        adata_val = adata[adata.obs["condition"] == "tonsillitis"]

        # ## 3.1 Training

        import numpy as np

        np.isnan(adata_train.X).sum()

        svc = sp.tl.ml_train(
            adata_train=adata_train, label="celltype", nan_policy_y="omit"
        )

        sp.tl.ml_predict(adata_val=adata_val, svc=svc, save_name="svm_pred")

        sc.pl.umap(adata_val, color="svm_pred")

        # Since we also know the cell type annotation of the adata_val, we can check in this case
        from sklearn.metrics import classification_report

        y_true = adata_val.obs["celltype"].values
        y_pred = adata_val.obs["svm_pred"].values
        nan_mask = ~y_true.isna()
        y_true = y_true[nan_mask]
        y_pred = y_pred[nan_mask]

        svm_eval = classification_report(
            y_true=y_true, y_pred=y_pred, target_names=svc.classes_, output_dict=True
        )
        sns.heatmap(pd.DataFrame(svm_eval).iloc[:-1, :].T, annot=True)
        plt.show()

        # ### 3.4 Save model

        import pickle

        filename = "svc_model.sav"
        pickle.dump(svc, open(output_path / filename, "wb"))
        # adata_val.write(output_dir + "adata_nn_ml_demo_annotated.h5ad")

        # ## 3.5 Single-cell visualzation

        sp.pl.catplot(
            adata_val,
            color="svm_pred",  # specify group column name here (e.g. celltype_fine)
            unique_region="condition",  # specify unique_regions here
            X="x",
            Y="y",  # specify x and y columns here
            n_columns=1,  # adjust the number of columns for plotting here (how many plots do you want in one row?)
            palette=None,  # default is None which means the color comes from the anndata.uns that matches the UMAP
            savefig=False,  # save figure as pdf
            output_fname="",  # change it to file name you prefer when saving the figure
            output_dir=output_dir,  # specify output directory here (if savefig=True)
        )

        # cell type percentage tab and visualization [much few]
        ct_perc_tab, _ = sp.pl.stacked_bar_plot(
            adata=adata_val,  # adata object to use
            color="svm_pred",  # column containing the categories that are used to fill the bar plot
            grouping="condition",  # column containing a grouping variable (usually a condition or cell group)
            cell_list=[
                "GCB",
                "Treg",
            ],  # list of cell types to plot, you can also see the entire cell types adata.obs['celltype_fine'].unique()
            palette=None,  # default is None which means the color comes from the anndata.uns that matches the UMAP
            savefig=False,  # change it to true if you want to save the figure
            output_fname="",  # change it to file name you prefer when saving the figure
            output_dir=output_dir,  # output directory for the figure
            norm=False,  # if True, then whatever plotted will be scaled to sum of 1
        )

        sp.pl.create_pie_charts(
            adata_val,
            color="svm_pred",
            grouping="condition",
            show_percentages=False,
            palette=None,  # default is None which means the color comes from the anndata.uns that matches the UMAP
            savefig=False,  # change it to true if you want to save the figure
            output_fname="",  # change it to file name you prefer when saving the figure
            output_dir=output_dir,  # output directory for the figure
        )


if __name__ == "__main__":
    test_3_cell_annotation_ml()
