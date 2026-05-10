import pathlib
from tempfile import TemporaryDirectory

import pytest

TEST_DIR = pathlib.Path(__file__).parent

pytestmark = pytest.mark.slow


def test_preprocessing_compensation():
    print("Testing compensation")

    import spacec as sp

    data_path = TEST_DIR / "data"  # where the data is stored
    overlay_path = TEST_DIR / "data/processed/tonsil/1"

    with TemporaryDirectory() as output_dir:
        output_path = pathlib.Path(output_dir)

        import pickle

        import pandas as pd

        with open(overlay_path / "seg_output_tonsil1.pickle", "rb") as f:
            seg_output1 = pickle.load(f)

        reg1 = pd.read_csv(
            data_path / "processed/cellseg/reg010_X01_Y01_Z01_compensated.csv"
        )

        reg1 = reg1.head(30)

        df1_c = sp.pp.compensate_cell_matrix(
            reg1,  # dataframe with the first region
            image_dict=seg_output1["image_dict"],  # dictionary with images
            masks=seg_output1["masks"],  # masks from the second region
            overwrite=False,
            device="cpu",
        )  # either overwrite your channels in your dataframe or add compensated channels as new columns

        df1_c = sp.pp.compensate_cell_matrix(
            reg1,  # dataframe with the first region
            image_dict=seg_output1["image_dict"],  # dictionary with images
            masks=seg_output1["masks"],  # masks from the second region
            overwrite=True,
            device="cpu",
        )  # either overwrite your channels in your dataframe or add compensated channels as new columns

        df1_c.to_csv(output_path / "df_nn_demo_comp.csv")


@pytest.mark.gpu
def test_preprocessing_compensation_gpu_autoselect():
    print("Testing compensation")

    import spacec as sp

    data_path = TEST_DIR / "data"  # where the data is stored
    overlay_path = TEST_DIR / "data/processed/tonsil/1"

    with TemporaryDirectory() as output_dir:
        output_path = pathlib.Path(output_dir)

        import pickle

        import pandas as pd

        with open(overlay_path / "seg_output_tonsil1.pickle", "rb") as f:
            seg_output1 = pickle.load(f)

        reg1 = pd.read_csv(
            data_path / "processed/cellseg/reg010_X01_Y01_Z01_compensated.csv"
        )

        reg1 = reg1.head(30)

        df1_c = sp.pp.compensate_cell_matrix(
            reg1,  # dataframe with the first region
            image_dict=seg_output1["image_dict"],  # dictionary with images
            masks=seg_output1["masks"],  # masks from the second region
            overwrite=False,
            device=None,  # select automatically
        )  # either overwrite your channels in your dataframe or add compensated channels as new columns

        df1_c = sp.pp.compensate_cell_matrix(
            reg1,  # dataframe with the first region
            image_dict=seg_output1["image_dict"],  # dictionary with images
            masks=seg_output1["masks"],  # masks from the second region
            overwrite=True,
            device=None,  # select automatically
        )  # either overwrite your channels in your dataframe or add compensated channels as new columns

        df1_c.to_csv(output_path / "df_nn_demo_comp.csv")


if __name__ == "__main__":
    test_preprocessing_compensation()
