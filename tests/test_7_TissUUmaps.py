import pathlib
from tempfile import TemporaryDirectory

import pytest

TEST_DIR = pathlib.Path(__file__).parent


@pytest.mark.skip(reason="Makes pytest hang after completion")
def test_7_TissUUmaps():
    # Set up environment
    import matplotlib
    import scanpy as sc

    import spacec as sp

    matplotlib.use("Agg")
    sc.settings.set_figure_params(dpi=80, facecolor="white")

    processed_path = TEST_DIR / "data/processed/tonsil/1"

    adata = sc.read(processed_path / "adata_nn_demo_annotated_cn.h5ad")
    adata

    image_list, csv_paths = sp.tl.tm_viewer(
        adata,
        images_pickle_path=processed_path / "seg_output_tonsil1.pickle",
        directory=processed_path / "cache",
        region_column="unique_region",
        region="reg001",
        xSelector="y",
        ySelector="x",
        color_by="celltype",
        keep_list=None,
        open_viewer=True,
    )

    csv_paths = sp.tl.tm_viewer_catplot(
        adata,
        directory=processed_path / "cache",
        region_column="unique_region",
        xSelector="y",
        ySelector="x",
        color_by="celltype",
        keep_list=None,
        open_viewer=True,
    )


if __name__ == "__main__":
    test_7_TissUUmaps()
