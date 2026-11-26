"""
Cross-format consistency tests for `input_format` handling in the tissue
extraction pipeline: `sp.hf.downscale_tissue` -> `sp.tl.label_tissue` ->
`sp.tl.save_labelled_tissue`.

This mirrors `test_input_formats.py` (which checks `input_format` handling
for `sp.tl.cell_segmentation`), but needs its own, larger fixture.
`downscale_tissue` downscales the full image by `downscale_factor` and then
runs Otsu thresholding on the *downscaled* result to separate tissue from
background. The small 64x64 crop used by `test_input_formats.py` (and the
only real tonsil image committed in this repo,
`tests/data/raw/tonsil/1/reg010_X01_Y01_Z01.tif`, which is itself already
cropped down to 64x64) collapses to roughly a 1x1 image under the default
`downscale_factor=64` -- far too small for Otsu/watershed to do anything
meaningful, and not a realistic exercise of this code path.

There is no larger real tonsil (or qptiff) image committed in this repo
small enough to keep in git but large enough to downscale meaningfully, so
this module builds its own synthetic 9-channel, 512x512 fixture: a bright
circular "tissue" blob on a darker, noisy "background" in the nuclear
(DAPI) channel, which gives Otsu real bimodal signal to find. To keep that
signal legible after the function's default Gaussian smoothing
(`sigma=5.0`) without needing a much larger (and slower) fixture image,
these tests use `downscale_factor=8` (passed explicitly and consistently to
both `downscale_tissue` and `save_labelled_tissue`, exactly as a real
pipeline run would use one consistent `downscale_factor` throughout) rather
than the functions' own default of 64. A 512x512 image downscaled by 64
leaves only an 8x8 array, over which `sigma=5.0` Gaussian smoothing
destroys almost all contrast; downscaled by 8 it leaves a 64x64 array, on
which the same smoothing still preserves a clear tissue/background
separation. This was confirmed empirically (otsu threshold found on a
64x64 downscaled array vs. a near-constant 8x8 array) before writing these
tests.

The other 8 channels are not used by this pipeline (only the nuclear
channel is), but are included so the Multichannel/CODEX/Channels fixtures
exercise the same multi-channel load/crop/write logic as the real
`downscale_tissue`/`save_labelled_tissue` code paths (and so that
`save_labelled_tissue`'s per-format output writing, which writes every
channel, is itself tested, not just the nuclear channel).
"""

import pathlib

import numpy as np
import pandas as pd
import pytest
import tifffile

TEST_DIR = pathlib.Path(__file__).parent

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
NUMBER_CYCLES = 3
IMAGES_PER_CYCLE = 3

IMAGE_SIZE = 512
DOWNSCALE_FACTOR = 8
BLOB_RADIUS = IMAGE_SIZE * 0.25
BACKGROUND_VALUE = 15
BACKGROUND_STD = 3
TISSUE_VALUE = 180
TISSUE_STD = 10
PADDING = 50
RNG_SEED = 12345


def _make_nuclei_channel(rng):
    """
    A bright circular "tissue" blob on a darker, noisy background -- gives
    Otsu thresholding (used by `downscale_tissue`) real bimodal signal to
    separate, unlike uniform noise.
    """
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    center = IMAGE_SIZE / 2
    dist = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    mask = dist < BLOB_RADIUS

    img = rng.normal(BACKGROUND_VALUE, BACKGROUND_STD, size=(IMAGE_SIZE, IMAGE_SIZE))
    img[mask] = rng.normal(TISSUE_VALUE, TISSUE_STD, size=int(mask.sum()))
    return np.clip(img, 0, 255).astype(np.uint8)


def _make_other_channel(rng):
    """A plain noisy channel; content doesn't matter, only that it round-trips."""
    return np.clip(rng.normal(20, 5, size=(IMAGE_SIZE, IMAGE_SIZE)), 0, 255).astype(
        np.uint8
    )


@pytest.fixture(scope="module")
def format_fixtures(tmp_path_factory):
    """
    Build synthetic Multichannel, CODEX, and Channels fixtures from the same
    underlying pixel data (one bright-blob-on-dark-background nuclear
    channel plus 8 plain noise channels), written under pytest's tmp_path.

    Returns a dict keyed by format name, each value a dict with
    "file_name" and "channel_file" (channel_file is None for "Channels").
    """
    rng = np.random.default_rng(RNG_SEED)

    channels = {NUCLEI_CHANNEL: _make_nuclei_channel(rng)}
    for name in SUBSET_CHANNELS:
        if name != NUCLEI_CHANNEL:
            channels[name] = _make_other_channel(rng)

    subset_image = np.stack([channels[name] for name in SUBSET_CHANNELS], axis=0)

    base_dir = tmp_path_factory.mktemp("tissue_format_fixtures")

    # --- Multichannel: single (channel, y, x) TIFF ---
    mc_dir = base_dir / "multichannel"
    mc_dir.mkdir()
    mc_file = mc_dir / "image.tif"
    tifffile.imwrite(mc_file, subset_image)
    mc_channel_file = mc_dir / "channelnames.txt"
    mc_channel_file.write_text("\n".join(SUBSET_CHANNELS))

    # --- CODEX: single (cycle, y, x, channel-in-cycle) TIFF ---
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
    for name in SUBSET_CHANNELS:
        tifffile.imwrite(channels_dir / f"{name}.tif", channels[name])

    return {
        "subset_image": subset_image,
        "Multichannel": {"file_name": mc_file, "channel_file": mc_channel_file},
        "CODEX": {"file_name": codex_file, "channel_file": codex_channel_file},
        "Channels": {"file_name": channels_dir, "channel_file": None},
    }


@pytest.fixture(scope="module")
def downscale_results(format_fixtures):
    """
    Run `downscale_tissue` for all three formats. Separated from the test
    functions so both the `downscale_tissue` consistency test and the
    `label_tissue`/`save_labelled_tissue` tests can reuse it without
    recomputing it per test.
    """
    import matplotlib

    matplotlib.use("Agg")
    import spacec as sp

    results = {}
    for input_format in ["Multichannel", "CODEX", "Channels"]:
        spec = format_fixtures[input_format]
        resized_im, cutoffs = sp.hf.downscale_tissue(
            file_path=spec["file_name"],
            channel_file=spec["channel_file"],
            input_format=input_format,
            nuclei_channel=NUCLEI_CHANNEL,
            downscale_factor=DOWNSCALE_FACTOR,
            showfig=False,
            savefig=False,
            verbose=False,
            return_cutoffs=True,
        )
        results[input_format] = {"resized_im": resized_im, "cutoffs": cutoffs}
    return results


def test_downscale_tissue_finds_real_signal(downscale_results):
    """
    Sanity check that the synthetic fixture actually gives Otsu real
    bimodal signal to work with (not a degenerate near-constant array, which
    is exactly the failure mode that makes the small 64x64 fixture used
    elsewhere in this suite unsuitable for this pipeline).
    """
    result = downscale_results["Multichannel"]
    cutoffs = result["cutoffs"]

    assert cutoffs["otsu_status"] == "success"
    assert result["resized_im"].shape == (
        IMAGE_SIZE // DOWNSCALE_FACTOR,
        IMAGE_SIZE // DOWNSCALE_FACTOR,
    )
    # A real bimodal split, not a near-constant array collapsed by smoothing.
    assert result["resized_im"].max() - result["resized_im"].min() > 0.1
    assert 0 < cutoffs["recommended_lower_cutoff"] < cutoffs["recommended_upper_cutoff"]


@pytest.mark.parametrize("input_format", ["CODEX", "Channels"])
def test_downscale_tissue_consistent_across_formats(downscale_results, input_format):
    """
    The key property under test: CODEX and Channels loading must recover the
    same downscaled image and the same cutoff recommendations as the
    Multichannel control, since the underlying nuclear-channel pixel data is
    identical.
    """
    expected = downscale_results["Multichannel"]
    actual = downscale_results[input_format]

    np.testing.assert_array_equal(
        actual["resized_im"],
        expected["resized_im"],
        err_msg=f"downscale_tissue resized image for input_format={input_format!r} "
        "does not match the Multichannel control.",
    )
    assert actual["cutoffs"] == expected["cutoffs"], (
        f"downscale_tissue cutoff recommendations for input_format={input_format!r} "
        "do not match the Multichannel control."
    )


@pytest.fixture(scope="module")
def label_results(downscale_results):
    """Run `label_tissue` for all three formats, using each format's own
    `downscale_tissue` output (not just reusing the Multichannel result), so
    a bug that only affects one format's upstream data would still surface
    here."""
    import matplotlib

    matplotlib.use("Agg")
    import spacec as sp

    results = {}
    for input_format, downscale_result in downscale_results.items():
        tissueframe = sp.tl.label_tissue(
            downscale_result["resized_im"],
            lower_cutoff=downscale_result["cutoffs"]["recommended_lower_cutoff"],
            upper_cutoff=downscale_result["cutoffs"]["recommended_upper_cutoff"],
            showfig=False,
        )
        results[input_format] = tissueframe
    return results


def test_label_tissue_finds_one_tissue_piece(label_results):
    """
    Sanity check: the single circular blob should be identified as exactly
    one tissue piece, with a non-trivial number of downscaled pixels
    assigned to it (not the 0-or-1-pixel degenerate result a too-small
    fixture would produce).
    """
    tissueframe = label_results["Multichannel"]
    assert tissueframe["tissue"].nunique() == 1
    assert len(tissueframe) > 100


@pytest.mark.parametrize("input_format", ["CODEX", "Channels"])
def test_label_tissue_consistent_across_formats(label_results, input_format):
    """label_tissue output (pixel-to-tissue assignment) must be identical
    across formats, since it's derived from identical downscaled images."""
    expected = label_results["Multichannel"].reset_index(drop=True)
    actual = label_results[input_format].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        obj=f"label_tissue output for input_format={input_format!r}",
    )


def _reload_crop(save_dir, input_format, channel_file):
    """Reload a saved tissue crop back into a {channel_name: array} dict,
    regardless of which input_format it was written in."""
    from spacec._shared.segmentation import format_CODEX

    saved = list(save_dir.iterdir())
    assert len(saved) == 1, f"Expected exactly one saved tissue crop, found {saved}"
    crop_path = saved[0]

    if input_format == "Multichannel":
        crop = tifffile.imread(crop_path)
        return {name: crop[i] for i, name in enumerate(SUBSET_CHANNELS)}
    elif input_format == "CODEX":
        crop = tifffile.imread(crop_path)
        image_dict, _ = format_CODEX(
            image=crop, channel_names=SUBSET_CHANNELS, input_format="CODEX"
        )
        return image_dict
    else:  # Channels
        image_dict, _ = format_CODEX(image=str(crop_path), input_format="Channels")
        return image_dict


def test_save_labelled_tissue_matches_source(format_fixtures, label_results, tmp_path):
    """
    Sanity check of the Multichannel crop itself (the format already
    covered by `test_01_qptiff_converter.py`): the saved tissue crop must
    reproduce the exact source pixel data at the expected bounding box
    before the CODEX/Channels variants can be trusted against it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import spacec as sp

    subset_image = format_fixtures["subset_image"]
    tissueframe = label_results["Multichannel"]

    spec = format_fixtures["Multichannel"]
    out_dir = tmp_path / "Multichannel"
    out_dir.mkdir()
    sp.tl.save_labelled_tissue(
        filepath=spec["file_name"],
        tissueframe=tissueframe,
        region="region1",
        padding=PADDING,
        downscale_factor=DOWNSCALE_FACTOR,
        output_dir=str(out_dir),
        output_fname="",
        scalebar=False,
        input_format="Multichannel",
        channel_file=spec["channel_file"],
    )

    # Mirror save_labelled_tissue's own bounding-box computation to get the
    # independently-expected crop straight from the known source pixels.
    bbox = tissueframe.groupby("region1").agg(["min", "max"])
    assert len(bbox) == 1, "Expected exactly one tissue piece in the fixture."
    row = bbox.iloc[0]
    ymin = max(row["y"]["min"] * DOWNSCALE_FACTOR - PADDING, 0)
    ymax = min(row["y"]["max"] * DOWNSCALE_FACTOR + PADDING, IMAGE_SIZE)
    xmin = max(row["x"]["min"] * DOWNSCALE_FACTOR - PADDING, 0)
    xmax = min(row["x"]["max"] * DOWNSCALE_FACTOR + PADDING, IMAGE_SIZE)

    expected = {
        name: subset_image[i, ymin:ymax, xmin:xmax]
        for i, name in enumerate(SUBSET_CHANNELS)
    }

    reloaded = _reload_crop(out_dir, "Multichannel", spec["channel_file"])
    assert set(reloaded.keys()) == set(SUBSET_CHANNELS)
    for name, arr in expected.items():
        np.testing.assert_array_equal(reloaded[name], arr)


@pytest.mark.parametrize("input_format", ["CODEX", "Channels"])
def test_save_labelled_tissue_consistent_across_formats(
    format_fixtures, label_results, tmp_path, input_format
):
    """
    The key property under test: save_labelled_tissue's CODEX and Channels
    output must contain the same cropped pixel data as the Multichannel
    control, for every channel, not just run without crashing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import spacec as sp

    # Use the shared Multichannel tissueframe (confirmed identical across
    # formats by test_label_tissue_consistent_across_formats) so all three
    # formats crop the same bounding box.
    tissueframe = label_results["Multichannel"]

    mc_spec = format_fixtures["Multichannel"]
    mc_out_dir = tmp_path / "Multichannel_control"
    mc_out_dir.mkdir()
    sp.tl.save_labelled_tissue(
        filepath=mc_spec["file_name"],
        tissueframe=tissueframe,
        region="region1",
        padding=PADDING,
        downscale_factor=DOWNSCALE_FACTOR,
        output_dir=str(mc_out_dir),
        output_fname="",
        scalebar=False,
        input_format="Multichannel",
        channel_file=mc_spec["channel_file"],
    )
    expected = _reload_crop(mc_out_dir, "Multichannel", None)

    spec = format_fixtures[input_format]
    out_dir = tmp_path / input_format
    out_dir.mkdir()
    sp.tl.save_labelled_tissue(
        filepath=spec["file_name"],
        tissueframe=tissueframe,
        region="region1",
        padding=PADDING,
        downscale_factor=DOWNSCALE_FACTOR,
        output_dir=str(out_dir),
        output_fname="",
        scalebar=False,
        input_format=input_format,
        channel_file=spec["channel_file"],
    )
    actual = _reload_crop(out_dir, input_format, spec["channel_file"])

    assert set(actual.keys()) == set(SUBSET_CHANNELS)
    for name in SUBSET_CHANNELS:
        np.testing.assert_array_equal(
            actual[name],
            expected[name],
            err_msg=f"save_labelled_tissue crop for channel '{name}', "
            f"input_format={input_format!r} does not match the Multichannel control.",
        )


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
