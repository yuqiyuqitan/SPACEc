"""Unit tests for spacec._shared.overlay_utils and spacec._shared.segmentation."""

import numpy as np
import pytest

from spacec._shared.overlay_utils import (
    _normalize_channel,
    create_rgb_image,
    make_outline_overlay,
)
from spacec._shared.segmentation import combine_channels, format_CODEX

# ---------------------------------------------------------------------------
# _normalize_channel
# ---------------------------------------------------------------------------


def test_normalize_channel_normal_range():
    channel = np.array([[0.0, 5.0], [10.0, 15.0]])
    result = _normalize_channel(channel)
    assert result.dtype == np.float32
    assert result.shape == (2, 2)
    assert result[0, 0] == pytest.approx(0.0)
    assert result[1, 1] == pytest.approx(1.0)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_normalize_channel_constant_array():
    channel = np.full((4, 4), 7.0)
    result = _normalize_channel(channel)
    assert result.dtype == np.float32
    assert np.all(result == 0.0), "constant array should return all zeros"


def test_normalize_channel_negative_values():
    channel = np.array([[-3.0, 0.0], [3.0, 6.0]])
    result = _normalize_channel(channel)
    assert result[0, 0] == pytest.approx(0.0)
    assert result[1, 1] == pytest.approx(1.0)
    assert result[0, 1] == pytest.approx(3.0 / 9.0)


def test_normalize_channel_integer_input():
    channel = np.array([[0, 100], [200, 50]], dtype=np.int32)
    result = _normalize_channel(channel)
    assert result.dtype == np.float32
    assert result[1, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# create_rgb_image
# ---------------------------------------------------------------------------


def test_create_rgb_image_basic_shape():
    image_batch = np.zeros((2, 8, 8, 3), dtype=np.float32)
    result = create_rgb_image(image_batch, channel_colors=["red", "green", "blue"])
    assert result.shape == (2, 8, 8, 3)
    assert result.dtype == np.float32
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_create_rgb_image_default_colors_all_blue():
    image_batch = np.zeros((1, 4, 4, 2), dtype=np.float32)
    # Use a gradient so _normalize_channel produces non-zero output
    image_batch[0, :, :, 0] = np.linspace(0, 1, 16).reshape(4, 4)
    result = create_rgb_image(image_batch, channel_colors=None)
    # default is "blue" for all channels
    assert np.all(result[0, :, :, 0] == 0.0), "red channel should be 0"
    assert np.all(result[0, :, :, 1] == 0.0), "green channel should be 0"
    assert np.any(result[0, :, :, 2] > 0.0), "blue channel should be non-zero"


def test_create_rgb_image_clipping():
    # Two channels both mapped to "red"; their contributions sum to >1.0 before clip
    image_batch = np.ones((1, 4, 4, 2), dtype=np.float32)
    result = create_rgb_image(image_batch, channel_colors=["red", "red"])
    assert np.max(result) <= 1.0, "output must be clipped to 1.0"
    assert np.min(result) >= 0.0


def test_create_rgb_image_wrong_ndim_raises():
    with pytest.raises(ValueError):
        create_rgb_image(np.zeros((8, 8, 3)))


def test_create_rgb_image_unknown_color_falls_back_to_white():
    # Use gradient so _normalize_channel gives non-zero output (constant arrays → zeros)
    image_batch = np.zeros((1, 4, 4, 1), dtype=np.float32)
    image_batch[0, :, :, 0] = np.linspace(0, 1, 16).reshape(4, 4)
    result = create_rgb_image(image_batch, channel_colors=["notacolor"])
    # unknown color → falls back to white [1, 1, 1]; brightest pixels should be white
    assert np.any(result[0, :, :, 0] > 0.0), "red (white) should be non-zero"
    assert np.any(result[0, :, :, 1] > 0.0), "green (white) should be non-zero"
    assert np.any(result[0, :, :, 2] > 0.0), "blue (white) should be non-zero"
    # R, G, B channels should be equal (white = equal proportions)
    np.testing.assert_array_equal(result[0, :, :, 0], result[0, :, :, 1])
    np.testing.assert_array_equal(result[0, :, :, 1], result[0, :, :, 2])


def test_create_rgb_image_color_padding_with_white():
    # Only 1 color provided for 3 channels; remaining 2 padded with "white"
    image_batch = np.zeros((1, 4, 4, 3), dtype=np.float32)
    result = create_rgb_image(image_batch, channel_colors=["red"])
    assert result.shape == (1, 4, 4, 3)


# ---------------------------------------------------------------------------
# make_outline_overlay
# ---------------------------------------------------------------------------


def test_make_outline_overlay_basic():
    rgb_data = np.zeros((1, 12, 12, 3), dtype=np.float32)
    predictions = np.zeros((1, 12, 12), dtype=np.int32)
    predictions[0, 4:8, 4:8] = 1  # filled 4×4 square not touching edge
    result = make_outline_overlay(rgb_data, predictions, outline_color=(1.0, 0.0, 0.0))
    assert result.shape == (1, 12, 12, 3)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)
    # interior pixel inside the filled square is foreground, not outer boundary
    assert result[0, 5, 5, 0] == 0.0, "interior pixel should be unchanged (red=0)"
    # at least some boundary pixels should be painted red
    assert np.any(result[0, :, :, 0] == 1.0), "red outline should appear"


def test_make_outline_overlay_wrong_rgb_ndim_raises():
    with pytest.raises(ValueError):
        make_outline_overlay(np.zeros((10, 10, 3)), np.zeros((1, 10, 10)))


def test_make_outline_overlay_wrong_rgb_last_dim_raises():
    with pytest.raises(ValueError):
        make_outline_overlay(np.zeros((1, 10, 10, 4)), np.zeros((1, 10, 10)))


def test_make_outline_overlay_predictions_2d_auto_batched():
    rgb_data = np.zeros((1, 8, 8, 3), dtype=np.float32)
    predictions = np.zeros((8, 8), dtype=np.int32)  # 2D → should be auto-batched
    result = make_outline_overlay(rgb_data, predictions)
    assert result.shape == (1, 8, 8, 3)


def test_make_outline_overlay_predictions_4d_with_trailing_1():
    rgb_data = np.zeros((1, 8, 8, 3), dtype=np.float32)
    predictions = np.zeros((1, 8, 8, 1), dtype=np.int32)  # (B,H,W,1) → squeezed
    result = make_outline_overlay(rgb_data, predictions)
    assert result.shape == (1, 8, 8, 3)


def test_make_outline_overlay_wrong_predictions_shape_raises():
    with pytest.raises(ValueError):
        make_outline_overlay(np.zeros((1, 8, 8, 3)), np.zeros((1, 8, 8, 2)))


def test_make_outline_overlay_adjacent_cells_separated():
    """Adjacent cells must each get their own inner boundary, not merge into one blob."""
    rgb_data = np.zeros((1, 10, 10, 3), dtype=np.float32)
    predictions = np.zeros((1, 10, 10), dtype=np.int32)
    predictions[0, 2:8, 2:5] = 1  # cell 1, left half
    predictions[0, 2:8, 5:8] = 2  # cell 2, right half, touching cell 1 at col 5
    result = make_outline_overlay(rgb_data, predictions)
    # The shared boundary columns (4 and 5) must be marked
    assert np.any(
        result[0, 2:8, 4:6, :] > 0
    ), "boundary between adjacent cells must be visible"


# ---------------------------------------------------------------------------
# combine_channels
# ---------------------------------------------------------------------------


def test_combine_channels_basic_max_projection():
    img_a = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    img_b = np.array([[0, 2], [3, 0]], dtype=np.uint8)
    image_dict = {"ch_a": img_a, "ch_b": img_b}
    result = combine_channels(image_dict, ["ch_a", "ch_b"], "combined")
    expected = np.array([[1, 2], [3, 1]], dtype=np.uint8)
    assert "combined" in result
    np.testing.assert_array_equal(result["combined"], expected)
    assert "ch_a" in result and "ch_b" in result


def test_combine_channels_preserves_dtype():
    img = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = combine_channels({"c": img}, ["c"], "out")
    assert result["out"].dtype == np.float32


def test_combine_channels_empty_channel_list():
    img = np.zeros((4, 4), dtype=np.uint8)
    image_dict = {"ch": img}
    result = combine_channels(image_dict, [], "combined")
    assert "combined" not in result
    assert len(result) == len(image_dict)


def test_combine_channels_missing_channels_skipped():
    img = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    image_dict = {"ch_a": img}
    result = combine_channels(image_dict, ["ch_a", "nonexistent"], "combined")
    assert "combined" in result
    np.testing.assert_array_equal(result["combined"], img)


def test_combine_channels_all_missing_channels():
    img = np.zeros((4, 4), dtype=np.uint8)
    image_dict = {"ch_a": img}
    result = combine_channels(image_dict, ["x", "y"], "combined")
    assert "combined" not in result


def test_combine_channels_shape_mismatch_skipped():
    img_a = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    img_b = np.zeros((4, 4), dtype=np.uint8)  # different shape
    image_dict = {"ch_a": img_a, "ch_b": img_b}
    result = combine_channels(image_dict, ["ch_a", "ch_b"], "combined")
    assert result["combined"].shape == (2, 2)
    np.testing.assert_array_equal(result["combined"], img_a)


# ---------------------------------------------------------------------------
# format_CODEX
# ---------------------------------------------------------------------------


def test_format_CODEX_multichannel_success():
    image = np.zeros((3, 8, 8), dtype=np.float32)
    channel_names = ["DAPI", "CD3", "CD20"]
    image_dict, names = format_CODEX(
        image, channel_names=channel_names, input_format="Multichannel"
    )
    assert image_dict is not None
    assert set(image_dict.keys()) == {"DAPI", "CD3", "CD20"}
    assert names == channel_names
    for v in image_dict.values():
        assert v.shape == (8, 8)


def test_format_CODEX_codex_success_auto_detect():
    image = np.zeros((2, 8, 8, 3), dtype=np.float32)
    channel_names = [f"ch{i}" for i in range(6)]
    image_dict, names = format_CODEX(
        image, channel_names=channel_names, input_format="CODEX"
    )
    assert image_dict is not None
    assert len(image_dict) == 6
    for v in image_dict.values():
        assert v.shape == (8, 8)


def test_format_CODEX_codex_explicit_params():
    image = np.zeros((2, 8, 8, 3), dtype=np.float32)
    channel_names = [f"ch{i}" for i in range(6)]
    image_dict, names = format_CODEX(
        image,
        channel_names=channel_names,
        number_cycles=2,
        images_per_cycle=3,
        input_format="CODEX",
    )
    assert image_dict is not None
    assert len(image_dict) == 6


def test_format_CODEX_multichannel_wrong_shape_returns_none():
    image = np.zeros((3, 8, 8), dtype=np.float32)
    channel_names = ["DAPI", "CD3"]  # 2 names for 3-channel image → mismatch
    image_dict, names = format_CODEX(
        image, channel_names=channel_names, input_format="Multichannel"
    )
    assert image_dict is None
    assert names is None


def test_format_CODEX_invalid_format_returns_none():
    image_dict, names = format_CODEX(
        np.zeros((3, 8, 8)), channel_names=["a", "b", "c"], input_format="BadFormat"
    )
    assert image_dict is None
    assert names is None


def test_format_CODEX_codex_no_channel_names_returns_none():
    image = np.zeros((2, 8, 8, 3), dtype=np.float32)
    image_dict, names = format_CODEX(image, channel_names=None, input_format="CODEX")
    assert image_dict is None
    assert names is None
