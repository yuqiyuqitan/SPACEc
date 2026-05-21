import numpy as np
from skimage.segmentation import find_boundaries


_COLOR_MAP = {
    "red": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "green": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "blue": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "yellow": np.array([1.0, 1.0, 0.0], dtype=np.float32),
    "magenta": np.array([1.0, 0.0, 1.0], dtype=np.float32),
    "cyan": np.array([0.0, 1.0, 1.0], dtype=np.float32),
    "white": np.array([1.0, 1.0, 1.0], dtype=np.float32),
}


def _normalize_channel(channel):
    channel = np.asarray(channel, dtype=np.float32)
    c_min = np.nanmin(channel)
    c_max = np.nanmax(channel)
    if not np.isfinite(c_min) or not np.isfinite(c_max) or c_max <= c_min:
        return np.zeros_like(channel, dtype=np.float32)
    return (channel - c_min) / (c_max - c_min)


def create_rgb_image(image_batch, channel_colors=None):
    """Create RGB visualization from a (B, H, W, C) array."""
    image_batch = np.asarray(image_batch)
    if image_batch.ndim != 4:
        raise ValueError(
            f"image_batch must have shape (batch, height, width, channels), got {image_batch.shape}"
        )

    batch, height, width, n_channels = image_batch.shape
    if channel_colors is None:
        channel_colors = ["blue"] * n_channels
    if len(channel_colors) < n_channels:
        channel_colors = list(channel_colors) + ["white"] * max(
            0, n_channels - len(channel_colors)
        )

    rgb = np.zeros((batch, height, width, 3), dtype=np.float32)
    for ch in range(n_channels):
        color = _COLOR_MAP.get(str(channel_colors[ch]).lower(), _COLOR_MAP["white"])
        normalized = _normalize_channel(image_batch[..., ch])
        rgb += normalized[..., None] * color

    return np.clip(rgb, 0.0, 1.0)


def make_outline_overlay(rgb_data, predictions, outline_color=(1.0, 0.0, 0.0)):
    """Overlay object boundaries from label masks on RGB images."""
    rgb_data = np.asarray(rgb_data, dtype=np.float32)
    if rgb_data.ndim != 4 or rgb_data.shape[-1] != 3:
        raise ValueError(f"rgb_data must have shape (batch, height, width, 3), got {rgb_data.shape}")

    predictions = np.asarray(predictions)
    if predictions.ndim == 4 and predictions.shape[-1] == 1:
        predictions = np.squeeze(predictions, axis=-1)
    elif predictions.ndim == 2:
        predictions = predictions[None, ...]
    if predictions.ndim != 3:
        raise ValueError(
            f"predictions must have shape (batch, height, width) or (batch, height, width, 1), got {predictions.shape}"
        )

    overlay = np.array(rgb_data, copy=True)
    outline_color = np.asarray(outline_color, dtype=np.float32)
    outline_color = np.clip(outline_color, 0.0, 1.0)

    batch_size = min(overlay.shape[0], predictions.shape[0])
    for i in range(batch_size):
        boundaries = find_boundaries(predictions[i] > 0, mode="outer")
        overlay[i][boundaries] = outline_color

    return np.clip(overlay, 0.0, 1.0)
