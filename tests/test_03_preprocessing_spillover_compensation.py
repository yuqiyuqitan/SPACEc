import pathlib
import pickle
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from scipy import ndimage
from skimage.measure import regionprops_table

TEST_DIR = pathlib.Path(__file__).parent


@pytest.fixture(scope="module")
def tonsil1_seg_output():
    overlay_path = TEST_DIR / "data/processed/tonsil/1"
    with open(overlay_path / "seg_output_tonsil1.pickle", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def tonsil1_features_csv(tmp_path_factory, tonsil1_seg_output):
    """Derive a features CSV for the tonsil1 mask fixture, since none is
    committed. Computed the same way extract_features() would (regionprops
    for label/x/y/area, per-label mean intensity for a few marker
    channels), so it is self-consistent with the mask -- unlike the old
    version of this test, which paired this mask with an unrelated
    region's CSV (reg010 cellseg data) and sliced .head(30) purely to make
    the row count match the mask's label count.
    """
    masks = tonsil1_seg_output["masks"]
    image_dict = tonsil1_seg_output["image_dict"]

    props = regionprops_table(masks, properties=("label", "centroid", "area"))
    df = pd.DataFrame(props).rename(columns={"centroid-0": "y", "centroid-1": "x"})
    for channel in ("DAPI", "CD20", "CD3"):
        df[channel] = ndimage.mean(image_dict[channel], labels=masks, index=df["label"])

    out_dir = tmp_path_factory.mktemp("tonsil1_features")
    csv_path = out_dir / "tonsil1_features.csv"
    # 'label' first, matching the column order extract_features() writes.
    df.to_csv(csv_path, index=False)
    return csv_path


def _run_compensation(seg_output, features_csv, output_dir):
    import spacec as sp

    output_path = pathlib.Path(output_dir)

    # This is the same object 03.1 works with: read via read_segdf (not a
    # raw pd.read_csv), so it carries the 'label' / 'unique_region' columns
    # compensate_cell_matrix joins on.
    df = sp.pp.read_segdf(
        segfile_list=[features_csv], seg_method="cellpose", region_list=["reg001"]
    )

    df_c = sp.pp.compensate_cell_matrix(
        df,
        image_dict=seg_output["image_dict"],
        masks=seg_output["masks"],
        overwrite=False,
        region="reg001",
    )
    assert "DAPI_compensated" in df_c.columns
    assert not df_c["DAPI_compensated"].isna().any(), (
        "every row's label came from this same mask, so every row should "
        "get a compensated value"
    )
    assert (df_c["DAPI_compensated"] >= 0).all()

    fig = sp.pl.compensation_qc(
        df_c,
        markers="DAPI",
        savefig=True,
        output_fname="tonsil1_compensation_qc",
        output_dir=str(output_path),
    )
    assert fig is not None

    fig_masks = sp.pl.compensation_qc(
        df_c,
        markers="DAPI",
        masks=seg_output["masks"],
        savefig=True,
        output_fname="tonsil1_compensation_qc_masks",
        output_dir=str(output_path),
    )
    assert fig_masks is not None

    with pytest.raises(ValueError):
        sp.pl.compensation_qc(
            df_c, markers="DAPI", masks=seg_output["masks"], label_col="not_a_column"
        )

    df_c.to_csv(output_path / "df_nn_demo_comp.csv")
    return df_c


def test_preprocessing_compensation(tonsil1_seg_output, tonsil1_features_csv):
    with TemporaryDirectory() as output_dir:
        _run_compensation(tonsil1_seg_output, tonsil1_features_csv, output_dir)


if __name__ == "__main__":
    with TemporaryDirectory() as output_dir:
        with open(
            TEST_DIR / "data/processed/tonsil/1/seg_output_tonsil1.pickle", "rb"
        ) as f:
            seg_output = pickle.load(f)
        masks = seg_output["masks"]
        image_dict = seg_output["image_dict"]
        props = regionprops_table(masks, properties=("label", "centroid", "area"))
        df = pd.DataFrame(props).rename(columns={"centroid-0": "y", "centroid-1": "x"})
        for channel in ("DAPI", "CD20", "CD3"):
            df[channel] = ndimage.mean(
                image_dict[channel], labels=masks, index=df["label"]
            )
        csv_path = pathlib.Path(output_dir) / "tonsil1_features.csv"
        df.to_csv(csv_path, index=False)
        _run_compensation(seg_output, csv_path, output_dir)
