"""
Cross-format consistency tests for `input_format` handling.

`sp.hf.hf_load_image_dict`, `sp.tl.cell_segmentation`, and `sp.pl.segmentation_ch`
all accept an `input_format` of "Multichannel", "CODEX", or "Channels" (see
`spacec._shared.segmentation.format_CODEX`). The example tonsil dataset used by
the rest of the test suite only ships in "Multichannel" form, so the "CODEX"
and "Channels" code paths are otherwise never exercised by real data.

These tests derive small "CODEX"- and "Channels"-shaped fixtures from the same
already-committed Multichannel crop used by `test_02_cell_segmentation.py`
(`tests/data/raw/tonsil/1/reg010_X01_Y01_Z01.tif`, 59 real channels cropped to
64x64), write them to pytest's `tmp_path`, and assert that all three formats
recover byte-identical per-channel image data and produce identical
segmentation output.
"""

import pathlib

import numpy as np
import pytest
import tifffile

TEST_DIR = pathlib.Path(__file__).parent
SOURCE_TIFF = TEST_DIR / "data" / "raw" / "tonsil" / "1" / "reg010_X01_Y01_Z01.tif"
SOURCE_CHANNEL_FILE = TEST_DIR / "data" / "raw" / "tonsil" / "channelnames.txt"

# Small, realistic subset of the real channel panel. 9 channels so it reshapes
# cleanly into 3 CODEX cycles x 3 images/cycle. Includes the nuclei channel
# ("DAPI") and the two membrane channels ("CD45", "betaCatenin") used by
# test_02_cell_segmentation.py, plus some extra markers.
SUBSET_CHANNELS = [
    "DAPI",
    "CD45",
    "betaCatenin",
    "CD3",
    "CD20",
    "CD68",
    "CD8",
    "FoxP3",
    "panCK",
]
NUCLEI_CHANNEL = "DAPI"
MEMBRANE_CHANNELS = ["CD45", "betaCatenin"]
NUMBER_CYCLES = 3
IMAGES_PER_CYCLE = 3


@pytest.fixture(scope="module")
def format_fixtures(tmp_path_factory):
    """
    Build Multichannel, CODEX, and Channels fixtures from the same source
    pixel data, written under pytest's tmp_path (never into the source
    dataset's own folder).

    Returns a dict keyed by format name, each value a dict with
    "file_name" and "channel_file" (channel_file is None for "Channels").
    """
    full_image = tifffile.imread(SOURCE_TIFF)
    with open(SOURCE_CHANNEL_FILE, "r") as f:
        full_channel_names = f.read().splitlines()

    indices = [full_channel_names.index(name) for name in SUBSET_CHANNELS]
    subset_image = full_image[indices]  # (9, H, W), same dtype as source

    base_dir = tmp_path_factory.mktemp("input_format_fixtures")

    # --- Multichannel: single (channel, y, x) TIFF ---
    mc_dir = base_dir / "multichannel"
    mc_dir.mkdir()
    mc_file = mc_dir / "image.tif"
    tifffile.imwrite(mc_file, subset_image)
    mc_channel_file = mc_dir / "channelnames.txt"
    mc_channel_file.write_text("\n".join(SUBSET_CHANNELS))

    # --- CODEX: single (cycle, y, x, channel-in-cycle) TIFF ---
    # format_CODEX recovers channel_names[i * images_per_cycle + n] from
    # image[i, :, :, n]; reshape+transpose so that mapping holds.
    reshaped = subset_image.reshape(
        NUMBER_CYCLES, IMAGES_PER_CYCLE, *subset_image.shape[1:]
    )
    codex_image = np.transpose(reshaped, (0, 2, 3, 1))
    codex_dir = base_dir / "codex"
    codex_dir.mkdir()
    codex_file = codex_dir / "image.tif"
    tifffile.imwrite(codex_file, codex_image)
    codex_channel_file = codex_dir / "channelnames.txt"
    codex_channel_file.write_text("\n".join(SUBSET_CHANNELS))

    # --- Channels: one single-channel TIFF per channel in a directory ---
    channels_dir = base_dir / "channels"
    channels_dir.mkdir()
    for name, idx in zip(SUBSET_CHANNELS, indices):
        tifffile.imwrite(
            channels_dir / f"{name}.tif", subset_image[SUBSET_CHANNELS.index(name)]
        )

    return {
        "subset_image": subset_image,
        "Multichannel": {"file_name": mc_file, "channel_file": mc_channel_file},
        "CODEX": {"file_name": codex_file, "channel_file": codex_channel_file},
        "Channels": {"file_name": channels_dir, "channel_file": None},
    }


def _expected_channel_arrays(fixtures):
    return {name: fixtures["subset_image"][i] for i, name in enumerate(SUBSET_CHANNELS)}


def test_multichannel_fixture_matches_source(format_fixtures):
    """
    Sanity check of the fixture-generation approach itself: loading the
    Multichannel fixture back (the format already covered by
    test_02_cell_segmentation.py) must reproduce the exact source pixel
    data before the CODEX/Channels variants derived from it can be trusted.
    """
    import spacec as sp

    expected = _expected_channel_arrays(format_fixtures)
    spec = format_fixtures["Multichannel"]
    image_dict = sp.hf.hf_load_image_dict(
        spec["file_name"],
        spec["channel_file"],
        input_format="Multichannel",
        nuclei_channel=NUCLEI_CHANNEL,
    )
    assert set(image_dict.keys()) == set(SUBSET_CHANNELS)
    for name, arr in expected.items():
        np.testing.assert_array_equal(image_dict[name], arr)


@pytest.mark.parametrize("input_format", ["CODEX", "Channels"])
def test_hf_load_image_dict_matches_multichannel(format_fixtures, input_format):
    """
    The key property under test: CODEX and Channels loading must recover
    the same underlying per-channel image data as the Multichannel control,
    not just load without crashing.
    """
    import spacec as sp

    expected = _expected_channel_arrays(format_fixtures)
    spec = format_fixtures[input_format]
    image_dict = sp.hf.hf_load_image_dict(
        spec["file_name"],
        spec["channel_file"],
        input_format=input_format,
        nuclei_channel=NUCLEI_CHANNEL,
    )
    assert (
        image_dict is not None
    ), f"hf_load_image_dict returned None for {input_format}"
    assert set(image_dict.keys()) == set(SUBSET_CHANNELS)
    for name, arr in expected.items():
        np.testing.assert_array_equal(
            image_dict[name],
            arr,
            err_msg=f"Channel '{name}' loaded via input_format={input_format!r} "
            "does not match the Multichannel control.",
        )


def test_cell_segmentation_consistent_across_formats(format_fixtures):
    """
    Run the real segmentation pipeline against all three formats and assert
    they produce identical masks, since the underlying pixel data is
    identical and Cellpose inference is deterministic for a fixed input.
    """
    import matplotlib

    matplotlib.use("Agg")
    import spacec as sp

    gpu = sp.hf.check_for_gpu()
    results = {}

    for input_format in ["Multichannel", "CODEX", "Channels"]:
        spec = format_fixtures[input_format]
        out_dir = (
            spec["file_name"].parent / "seg_out"
            if spec["file_name"].is_file()
            else spec["file_name"] / "seg_out"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        seg_output = sp.tl.cell_segmentation(
            file_name=spec["file_name"],
            channel_file=spec["channel_file"],
            output_dir=str(out_dir),
            output_fname=f"test_{input_format}",
            seg_method="cellpose",
            model="cyto3",
            diameter=10,
            nuclei_channel=NUCLEI_CHANNEL,
            membrane_channel_list=MEMBRANE_CHANNELS,
            input_format=input_format,
            resize_factor=1,
            size_cutoff=0,
            use_gpu=gpu,
        )
        assert seg_output is not None, f"cell_segmentation failed for {input_format}"
        assert (
            seg_output["masks"].max() > 0
        ), f"cell_segmentation produced an empty mask for input_format={input_format!r}"
        results[input_format] = seg_output

    np.testing.assert_array_equal(
        results["Multichannel"]["masks"],
        results["CODEX"]["masks"],
        err_msg="CODEX segmentation mask differs from the Multichannel control.",
    )
    np.testing.assert_array_equal(
        results["Multichannel"]["masks"],
        results["Channels"]["masks"],
        err_msg="Channels segmentation mask differs from the Multichannel control.",
    )


def test_segmentation_ch_runs_for_all_formats(format_fixtures, tmp_path):
    """
    Smoke test: sp.pl.segmentation_ch (the preview plot of the nuclei/membrane
    channels chosen for segmentation) should run without error for all three
    formats and save a figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import spacec as sp

    for input_format in ["Multichannel", "CODEX", "Channels"]:
        spec = format_fixtures[input_format]
        out_dir = tmp_path / f"segmentation_ch_{input_format}"
        out_dir.mkdir()

        sp.pl.segmentation_ch(
            file_name=spec["file_name"],
            channel_file=spec["channel_file"],
            output_dir=str(out_dir),
            savefig=True,
            output_fname=f"preview_{input_format}",
            membrane_channel_list=MEMBRANE_CHANNELS,
            nuclei_channel=NUCLEI_CHANNEL,
            input_format=input_format,
        )
        saved_files = list(out_dir.iterdir())
        assert len(saved_files) == 1, (
            f"Expected exactly one saved preview figure for input_format={input_format!r}, "
            f"found {saved_files}"
        )


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
