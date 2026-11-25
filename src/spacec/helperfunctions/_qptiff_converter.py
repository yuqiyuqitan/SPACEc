import warnings

import matplotlib.pyplot as plt
import tifffile
from scipy import ndimage as ndi
from skimage.transform import resize

from .._shared.segmentation import format_CODEX
from ..plotting._style import (
    SPACEcStyle,
    _add_scalebar,
    _apply_spines,
    _apply_style,
    introduce_line_breaks,
)


def downscale_tissue(
    file_path,
    DNAslice=None,
    downscale_factor=64,
    sigma=5.0,
    padding=50,
    savefig=False,
    showfig=True,
    output_dir="./",
    output_fname="",
    figsize=(12, 8),
    style: "Optional[Union[SPACEcStyle, str, pathlib.Path]]" = None,
    scalebar: bool = True,
    scalebar_length_px=None,
    um_per_px=None,
    scalebar_show_text: bool = True,
    return_cutoffs: bool = False,
    lower_cutoff_margin: float = 0.05,
    upper_cutoff_margin: float = 0.1,
    legend_loc=None,
    legend_bbox_to_anchor=None,
    verbose: bool = True,
    input_format="Multichannel",
    channel_file=None,
    nuclei_channel=None,
):
    """
    Load a qptiff file, extract a single nuclear channel, and return a
    downscaled, Gaussian-smoothed version suitable for tissue labelling.

    This function also generates a diagnostic figure and recommends cutoffs
    for the downstream ``label_tissue`` watershed segmentation function.

    Important watershed interpretation
    ----------------------------------
    In ``label_tissue``:

        markers[resized_im <= lower_cutoff] = 1
        markers[resized_im >= upper_cutoff] = 2

    Therefore:

    - ``lower_cutoff`` defines confident background markers.
    - ``upper_cutoff`` defines confident foreground/tissue markers.
    - Pixels between the two cutoffs are assigned by watershed.

    The recommended ``lower_cutoff`` is Otsu's threshold pulled down by
    ``lower_cutoff_margin`` (safely below background/tissue boundary even when
    tissue brightness varies), and ``upper_cutoff`` is simply
    ``lower_cutoff + upper_cutoff_margin``, a fixed band above it.

    Parameters
    ----------
    file_path : str
        Path to the input qptiff/tiff file (a directory of per-channel TIFFs when
        ``input_format="Channels"``).
    DNAslice : int, optional
        Deprecated alias for selecting the nuclear channel by position instead of
        by name. Only valid with ``input_format="Multichannel"``; raises a
        ``DeprecationWarning`` and overrides ``nuclei_channel`` if given. Will be
        removed in a future version -- use ``nuclei_channel`` (with
        ``channel_file``) instead. Default is ``None``.
    downscale_factor : int, optional
        Factor by which to reduce each spatial dimension. Default is 64.
    sigma : float, optional
        Standard deviation for the Gaussian smoothing kernel applied after
        downscaling. Default is 5.0.
    padding : int, optional
        Reserved parameter for future use. Default is 50.
    savefig : bool, optional
        If True, save the diagnostic figure to ``output_dir``. Default is False.
    showfig : bool, optional
        If True, display the diagnostic figure. Default is True.
    output_dir : str, optional
        Directory in which to save the figure when ``savefig=True``.
        Default is ``"./"``.
    output_fname : str, optional
        Base filename stem used when saving the figure. Default is ``""``.
    figsize : tuple of int, optional
        Width and height of the diagnostic figure in inches. Default is ``(12, 8)``.
    style : SPACEcStyle, str, pathlib.Path, optional
        Style configuration. Uses ``_apply_style(style)`` if available.
        Falls back to default matplotlib-like settings if unavailable.
    scalebar : bool, optional
        If True, add scalebars to the raw and processed image panels.
        Default is True.
    scalebar_length_px : int or float, optional
        Length of the scalebar in original image pixels. If None, a sensible
        automatic value is chosen. Default is None.
    um_per_px : float, optional
        Micrometers per original image pixel. If provided, scalebar text is
        shown in micrometers. If None, scalebar text is shown in pixels.
        Default is None.
    scalebar_show_text : bool, optional
        If True, show text label above the scalebar. Default is True.
    return_cutoffs : bool, optional
        If True, return ``(resized_im, cutoff_recommendations)``.
        If False, return only ``resized_im`` to preserve the original function
        behavior. Default is False.
    lower_cutoff_margin : float, optional
        Amount subtracted from the raw Otsu threshold to recommend
        ``lower_cutoff``. Default is 0.05.
    upper_cutoff_margin : float, optional
        Fixed amount added to ``lower_cutoff`` to recommend ``upper_cutoff``.
        Default is 0.1.
    legend_loc : str, optional
        ``loc`` passed directly to the histogram panel's
        ``Axes.legend(...)`` call (e.g. ``"upper center"``). Default is
        ``None``, i.e. matplotlib's own default (``"best"``, inside the
        axes). Pass together with ``legend_bbox_to_anchor`` to place the
        legend outside the axes, e.g. for styles with larger fonts.
    legend_bbox_to_anchor : tuple, optional
        ``bbox_to_anchor`` passed directly to the same ``legend(...)`` call,
        e.g. ``(0.5, -0.15)`` to anchor it centered, below the axes. Default
        is ``None`` (matplotlib's default anchor).
    verbose : bool, optional
        If True, print detailed explanations of each processing step.
        Default is True.
    input_format : str, optional
        The input_format used (either "CODEX", "Multichannel" or "Channels"), by
        default "Multichannel". "Multichannel": `file_path` is one stacked TIFF
        `(channels, H, W)`. "CODEX": `file_path` is one 4D TIFF
        `(cycles, H, W, images_per_cycle)`. "Channels": `file_path` is a directory
        of per-channel TIFFs.
    channel_file : str, optional
        Path to a text file listing channel names, one per line, in stack order.
        Required to resolve `nuclei_channel` by name for "Multichannel"/"CODEX".
        Ignored for "Channels" (channel names are taken from the filenames).
        Default is None.
    nuclei_channel : str, optional
        Name of the nuclear/DAPI channel to extract, matched against
        `channel_file` ("Multichannel"/"CODEX") or filenames ("Channels").
        Required unless `DNAslice` is given with `input_format="Multichannel"`.
        Default is None.

    Returns
    -------
    resized_im : numpy.ndarray
        2-D float array containing the downscaled and smoothed nuclear image.

    cutoff_recommendations : dict, optional
        Only returned when ``return_cutoffs=True``.
    """

    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.ndimage as ndi
    import tifffile
    from skimage.filters import threshold_otsu
    from skimage.transform import resize

    # ---------------------------------------------------------------------
    # Logging and style helpers
    # ---------------------------------------------------------------------
    def _log(message):
        if verbose:
            print(f"[downscale_tissue] {message}")

    def _get_style(style):
        """
        Apply SPACEcStyle if available. Falls back to sensible defaults if the
        project style helpers are not available.
        """
        try:
            return _apply_style(style)
        except Exception as exc:
            if verbose:
                print(
                    "[downscale_tissue] Warning: Could not apply SPACEcStyle. "
                    f"Using fallback matplotlib styling. Reason: {exc}"
                )

            return SPACEcStyle()

    def _style_attr(style_obj, attr, default):
        """
        Safely get an attribute from a SPACEcStyle-like object.
        """
        return getattr(style_obj, attr, default)

    def _style_axis(ax, style_obj):
        """
        Apply SPACEc-style axis formatting where possible.
        """
        try:
            _apply_spines(ax, style_obj)
        except Exception:
            pass

        try:
            ax.tick_params(labelsize=_style_attr(style_obj, "font_size_small", 8))
            ax.xaxis.label.set_size(_style_attr(style_obj, "font_size_small", 8))
            ax.yaxis.label.set_size(_style_attr(style_obj, "font_size_small", 8))
        except Exception:
            pass

    _style = _get_style(style)

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
    def _nice_length(value):
        """
        Return a visually pleasing rounded length using 1, 2, or 5 times
        a power of ten.
        """
        if value is None or value <= 0:
            return 1

        exponent = np.floor(np.log10(value))
        base = 10**exponent
        fraction = value / base

        if fraction >= 5:
            nice_fraction = 5
        elif fraction >= 2:
            nice_fraction = 2
        else:
            nice_fraction = 1

        return nice_fraction * base

    def _format_number(value):
        """
        Format scalebar, statistic, and cutoff labels cleanly.
        """
        if value is None:
            return "NA"

        try:
            if not np.isfinite(value):
                return "NA"
        except TypeError:
            return str(value)

        if abs(value) >= 100:
            return f"{value:.0f}"
        elif abs(value) >= 10:
            return f"{value:.2f}"
        elif abs(value) >= 1:
            return f"{value:.4f}"
        else:
            return f"{value:.4g}"

    def _make_scalebar_label(length_px_original):
        """
        Create scalebar label either in micrometers or original pixels.
        """
        if um_per_px is not None:
            length_um = length_px_original * um_per_px
            return f"{_format_number(length_um)} $\\mu$m"
        else:
            return f"{_format_number(length_px_original)} px"

    # ---------------------------------------------------------------------
    # Start workflow
    # ---------------------------------------------------------------------
    _log("Starting tissue downscaling workflow.")
    _log(
        "This function prepares a low-resolution nuclear image for tissue segmentation."
    )
    _log("")
    _log("Watershed cutoff interpretation:")
    _log("  lower_cutoff marks confident background pixels.")
    _log("  upper_cutoff marks confident tissue/foreground pixels.")
    _log("  pixels between lower_cutoff and upper_cutoff are assigned by watershed.")
    _log("")
    _log("Because upper_cutoff is a foreground marker threshold, it should not usually")
    _log(
        "be set to an extreme high percentile. Very high values can select only bright"
    )
    _log("tissue cores and miss dimmer tissue edges.")
    _log("")
    _log(f"Input file: {file_path}")
    _log(f"Input format: {input_format}")
    _log(f"Downscale factor: {downscale_factor}")
    _log(f"Gaussian smoothing sigma: {sigma}")
    _log(f"Lower cutoff margin: {lower_cutoff_margin}")
    _log(f"Upper cutoff margin: {upper_cutoff_margin}")
    _log(f"Padding parameter currently reserved/not applied: {padding}")

    # ---------------------------------------------------------------------
    # Read image and extract the nuclear channel
    # ---------------------------------------------------------------------
    if DNAslice is not None:
        if input_format != "Multichannel":
            raise ValueError(
                "'DNAslice' only applies to input_format='Multichannel' (it indexes "
                "the channel axis of a single stacked image). Use 'nuclei_channel' "
                "(with 'channel_file') for 'CODEX' or 'Channels' instead."
            )
        warnings.warn(
            "'DNAslice' is deprecated and will be removed in a future version. "
            "Instead, pass a 'channel_file' and specify channel name using "
            "'nuclei_channel' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _log(f"Requested DNA/nuclear slice index (deprecated): {DNAslice}")
        _log("Step 1/7: Reading qptiff file. This may take a while for large images.")
        currim = tifffile.imread(file_path)
        _log("Step 2/7: Extracting the selected nuclear/DNA channel by index.")
        nucim = currim[DNAslice]
        del currim
        nuclear_channel_label = f"index {DNAslice} (Multichannel, deprecated DNAslice)"
    elif nuclei_channel is not None:
        if input_format in ("Multichannel", "CODEX") and channel_file is None:
            raise ValueError(
                f"'channel_file' is required to resolve nuclei_channel={nuclei_channel!r} "
                f"by name when input_format={input_format!r}."
            )
        channel_names = None
        if channel_file is not None:
            with open(channel_file, "r") as f:
                channel_names = f.read().splitlines()
        _log(f"Requested nuclear channel: {nuclei_channel!r}")
        _log(f"Step 1/7: Reading image data (input_format={input_format}).")
        raw_image = (
            file_path if input_format == "Channels" else tifffile.imread(file_path)
        )
        image_dict, _ = format_CODEX(
            image=raw_image,
            channel_names=channel_names,
            input_format=input_format,
        )
        if image_dict is None or nuclei_channel not in image_dict:
            available = list(image_dict.keys()) if image_dict else []
            raise ValueError(
                f"nuclei_channel {nuclei_channel!r} not found. Available channels: {available}"
            )
        _log(f"Step 2/7: Extracting nuclear channel {nuclei_channel!r}.")
        nucim = image_dict[nuclei_channel]
        nuclear_channel_label = f"{nuclei_channel!r} (input_format={input_format})"
    else:
        if input_format != "Multichannel":
            raise ValueError(
                f"'nuclei_channel' is required when input_format={input_format!r}."
            )
        _log("Step 1/7: Reading qptiff file. This may take a while for large images.")
        currim = tifffile.imread(file_path)
        _log("Step 2/7: Extracting the default nuclear/DNA channel (index 0).")
        nucim = currim[0]
        del currim
        nuclear_channel_label = "index 0 (Multichannel, default)"

    _log(f"Loaded nuclear image with shape Y x X = {nucim.shape[0]} x {nucim.shape[1]}")
    _log(f"Input image dtype: {nucim.dtype}")

    # ---------------------------------------------------------------------
    # Downscale
    # ---------------------------------------------------------------------
    resized_shape = (
        nucim.shape[0] // downscale_factor,
        nucim.shape[1] // downscale_factor,
    )

    _log("Step 3/7: Downscaling image with anti-aliasing.")
    _log(
        "Downscaling from "
        f"{nucim.shape[0]} x {nucim.shape[1]} to "
        f"{resized_shape[0]} x {resized_shape[1]} pixels."
    )

    resized_im = resize(
        nucim,
        resized_shape,
        anti_aliasing=True,
    )

    # ---------------------------------------------------------------------
    # Smooth
    # ---------------------------------------------------------------------
    _log(
        "Step 4/7: Applying Gaussian smoothing to reduce small local intensity variation."
    )
    resized_im = ndi.gaussian_filter(resized_im, sigma=sigma)

    # ---------------------------------------------------------------------
    # Statistics and cutoff recommendations
    # ---------------------------------------------------------------------
    _log(
        "Step 5/7: Computing intensity statistics and watershed cutoff recommendations."
    )

    flat_vals = resized_im.ravel()
    flat_vals = flat_vals[np.isfinite(flat_vals)]

    min_val = np.min(flat_vals)
    max_val = np.max(flat_vals)
    mean_val = np.mean(flat_vals)
    median_val = np.median(flat_vals)

    p01_val = np.percentile(flat_vals, 1)
    p05_val = np.percentile(flat_vals, 5)
    p10_val = np.percentile(flat_vals, 10)
    p25_val = np.percentile(flat_vals, 25)
    p75_val = np.percentile(flat_vals, 75)
    p90_val = np.percentile(flat_vals, 90)
    p95_val = np.percentile(flat_vals, 95)
    p975_val = np.percentile(flat_vals, 97.5)
    p99_val = np.percentile(flat_vals, 99)
    p995_val = np.percentile(flat_vals, 99.5)

    _log("Trying Otsu thresholding for the lower/background cutoff.")
    try:
        otsu_val = threshold_otsu(flat_vals)
        otsu_status = "success"
        _log(f"Otsu threshold succeeded: {_format_number(otsu_val)}")
    except Exception as exc:
        otsu_val = p25_val
        otsu_status = f"failed: {exc}; used 25th percentile instead"
        _log(f"Otsu threshold failed: {exc}")

    # Recommended lower cutoff: Otsu's threshold pulled down by a fixed
    # margin, so it stays safely below the background/tissue boundary even
    # when tissue brightness varies a lot.
    # Used in label_tissue as:
    # markers[resized_im <= lower_cutoff] = 1
    recommended_lower_cutoff = max(otsu_val - lower_cutoff_margin, min_val)
    lower_cutoff_method = (
        f"Otsu threshold - {lower_cutoff_margin}; confident background marker cutoff"
    )

    _log("")
    _log("Lower cutoff recommendation:")
    _log(f"  Recommended lower_cutoff = {_format_number(recommended_lower_cutoff)}")
    _log("  Meaning in label_tissue:")
    _log("    markers[resized_im <= lower_cutoff] = 1")
    _log("  These pixels are treated as confident background markers.")
    _log("")

    # Recommended upper cutoff: a fixed band above lower_cutoff.
    # Used in label_tissue as:
    # markers[resized_im >= upper_cutoff] = 2
    recommended_upper_cutoff = min(
        recommended_lower_cutoff + upper_cutoff_margin, max_val
    )
    upper_cutoff_method = f"lower_cutoff + {upper_cutoff_margin}; confident foreground/tissue marker cutoff"

    _log("Upper cutoff recommendation:")
    _log(f"  Recommended upper_cutoff = {_format_number(recommended_upper_cutoff)}")
    _log("  Meaning in label_tissue:")
    _log("    markers[resized_im >= upper_cutoff] = 2")
    _log("  These pixels are treated as confident tissue/foreground markers.")
    _log("")
    _log("Important:")
    _log(
        "  The upper_cutoff is intentionally not recommended as the 99.5th percentile."
    )
    _log("  A very high value can mark only very bright tissue interiors and may cause")
    _log("  poor watershed behavior near tissue edges.")
    _log("")

    cutoff_recommendations = {
        "recommended_lower_cutoff": float(recommended_lower_cutoff),
        "recommended_upper_cutoff": float(recommended_upper_cutoff),
        "recommended_background_cutoff": float(recommended_lower_cutoff),
        "recommended_foreground_cutoff": float(recommended_upper_cutoff),
        "lower_cutoff_method": lower_cutoff_method,
        "upper_cutoff_method": upper_cutoff_method,
        "lower_cutoff_margin": float(lower_cutoff_margin),
        "upper_cutoff_margin": float(upper_cutoff_margin),
        "otsu_threshold": float(otsu_val),
        "otsu_status": otsu_status,
        "min": float(min_val),
        "p01": float(p01_val),
        "p05": float(p05_val),
        "p10": float(p10_val),
        "p25": float(p25_val),
        "median": float(median_val),
        "mean": float(mean_val),
        "p75": float(p75_val),
        "p90": float(p90_val),
        "p95": float(p95_val),
        "p975": float(p975_val),
        "p99": float(p99_val),
        "p995": float(p995_val),
        "max": float(max_val),
    }

    _log("Final cutoff recommendation summary for label_tissue:")
    _log(f"  lower_cutoff = {_format_number(recommended_lower_cutoff)}")
    _log(f"  upper_cutoff = {_format_number(recommended_upper_cutoff)}")
    _log("")
    _log("Suggested usage:")
    _log("  tissueframe = label_tissue(")
    _log("      resized_im,")
    _log(f"      lower_cutoff={recommended_lower_cutoff:.8g},")
    _log(f"      upper_cutoff={recommended_upper_cutoff:.8g},")
    _log("  )")
    _log("")

    # ---------------------------------------------------------------------
    # Scalebar
    # ---------------------------------------------------------------------
    if scalebar:
        _log("Step 6/7: Preparing scalebar information.")

        if scalebar_length_px is None:
            if um_per_px is not None:
                target_um = nucim.shape[1] * um_per_px * 0.20
                nice_um = _nice_length(target_um)
                scalebar_length_px_final = nice_um / um_per_px
            else:
                target_px = nucim.shape[1] * 0.20
                scalebar_length_px_final = _nice_length(target_px)
        else:
            scalebar_length_px_final = float(scalebar_length_px)

        scalebar_label = _make_scalebar_label(scalebar_length_px_final)

        _log(
            f"Scalebar enabled: {scalebar_label} "
            f"({scalebar_length_px_final:.4g} original pixels)."
        )
    else:
        _log("Step 6/7: Scalebar disabled.")

    # ---------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------
    if showfig or savefig:
        _log("Step 7/7: Generating diagnostic figure.")

        fig, axs = plt.subplots(
            2,
            2,
            figsize=figsize,
            constrained_layout=True,
        )

        fig.suptitle(
            "Tissue downscaling diagnostic with watershed marker recommendations",
            fontsize=_style_attr(_style, "font_size_large", 15),
            fontweight="bold",
        )

        raw_vmin, raw_vmax = np.percentile(nucim, [1, 99.5])
        resized_vmin, resized_vmax = np.percentile(resized_im, [1, 99.5])

        # -----------------------------------------------------------------
        # Panel 1: raw nuclear image
        # -----------------------------------------------------------------
        im0 = axs[0, 0].imshow(
            nucim,
            cmap="gray",
            vmin=raw_vmin,
            vmax=raw_vmax,
        )
        axs[0, 0].set_title(
            "Raw nuclear image",
            fontsize=_style_attr(_style, "font_size_medium", 12),
        )
        axs[0, 0].set_xlabel("X pixel")
        axs[0, 0].set_ylabel("Y pixel")
        cbar0 = fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
        cbar0.set_label("Intensity")

        if scalebar:
            _add_scalebar(
                axs[0, 0],
                nucim.shape[1],
                _style,
                scalebar_length_px=scalebar_length_px_final,
                um_per_px=um_per_px,
                show_text=scalebar_show_text,
            )

        # -----------------------------------------------------------------
        # Panel 2: processed image
        # -----------------------------------------------------------------
        im1 = axs[0, 1].imshow(
            resized_im,
            cmap="gray",
            vmin=resized_vmin,
            vmax=resized_vmax,
        )
        _panel2_title = (
            f"Downscaled + Gaussian-smoothed\n"
            f"factor = {downscale_factor}, sigma = {sigma}"
        )
        axs[0, 1].set_title(
            _panel2_title,
            fontsize=_style_attr(_style, "font_size_medium", 12),
        )
        axs[0, 1].set_xlabel("Downscaled X pixel")
        axs[0, 1].set_ylabel("Downscaled Y pixel")
        cbar1 = fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
        cbar1.set_label("Intensity")

        if scalebar:
            _add_scalebar(
                axs[0, 1],
                resized_im.shape[1],
                _style,
                scalebar_length_px=scalebar_length_px_final / downscale_factor,
                um_per_px=None if um_per_px is None else um_per_px * downscale_factor,
                show_text=scalebar_show_text,
            )

        # -----------------------------------------------------------------
        # Panel 3: histogram with watershed marker regions
        # -----------------------------------------------------------------
        axs[1, 0].hist(
            flat_vals,
            bins=100,
            color="steelblue",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
        )

        # Shaded regions showing how label_tissue will interpret intensities.
        axs[1, 0].axvspan(
            min_val,
            recommended_lower_cutoff,
            color="lightgray",
            alpha=0.45,
            label="Background markers",
        )

        axs[1, 0].axvspan(
            recommended_lower_cutoff,
            recommended_upper_cutoff,
            color="gold",
            alpha=0.18,
            label="Watershed-assigned region",
        )

        axs[1, 0].axvspan(
            recommended_upper_cutoff,
            max_val,
            color="mediumseagreen",
            alpha=0.20,
            label="Foreground markers",
        )

        axs[1, 0].axvline(
            recommended_lower_cutoff,
            color="black",
            linestyle="-",
            linewidth=2.5,
            label=f"lower_cutoff = {_format_number(recommended_lower_cutoff)}",
        )

        axs[1, 0].axvline(
            recommended_upper_cutoff,
            color="darkgreen",
            linestyle="-",
            linewidth=2.5,
            label=f"upper_cutoff = {_format_number(recommended_upper_cutoff)}",
        )

        axs[1, 0].axvline(
            median_val,
            color="crimson",
            linestyle="--",
            linewidth=1.6,
            label=f"Median = {_format_number(median_val)}",
        )

        axs[1, 0].axvline(
            mean_val,
            color="darkorange",
            linestyle="--",
            linewidth=1.6,
            label=f"Mean = {_format_number(mean_val)}",
        )

        _panel3_title = "Intensity histogram and watershed marker guidance"
        axs[1, 0].set_title(
            _panel3_title,
            fontsize=_style_attr(_style, "font_size_medium", 12),
        )
        axs[1, 0].set_xlabel("Processed image intensity")
        axs[1, 0].set_ylabel("Pixel count")
        _legend_kwargs = dict(
            frameon=False,
            fontsize=_style_attr(_style, "font_size_small", 8),
        )
        if legend_loc is not None:
            _legend_kwargs["loc"] = legend_loc
        if legend_bbox_to_anchor is not None:
            _legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        axs[1, 0].legend(**_legend_kwargs)
        axs[1, 0].grid(alpha=0.25)

        # -----------------------------------------------------------------
        # Panel 4: metadata and recommendations
        # -----------------------------------------------------------------
        axs[1, 1].axis("off")

        file_name = os.path.basename(file_path)

        if scalebar:
            if um_per_px is not None:
                scalebar_info = (
                    f"Scalebar: {scalebar_label}\n"
                    f"um per px: {um_per_px}\n"
                    f"Scalebar length: {scalebar_length_px_final:.4g} original px"
                )
            else:
                scalebar_info = (
                    f"Scalebar: {scalebar_label}\n"
                    "um per px: not provided\n"
                    f"Scalebar length: {scalebar_length_px_final:.4g} original px"
                )
        else:
            scalebar_info = "Scalebar: disabled"

        info_text = (
            "Processing summary\n"
            "------------------\n"
            f"File: {file_name}\n"
            f"Nuclear channel: {nuclear_channel_label}\n"
            f"Original shape: {nucim.shape[0]} x {nucim.shape[1]} px\n"
            f"Downscaled shape: {resized_im.shape[0]} x {resized_im.shape[1]} px\n"
            f"Downscale factor: {downscale_factor}\n"
            f"Gaussian sigma: {sigma}\n"
            f"Input dtype: {nucim.dtype}\n"
            f"{scalebar_info}\n"
            "\n"
            "Recommended label_tissue inputs\n"
            "-------------------------------\n"
            f"lower_cutoff: {_format_number(recommended_lower_cutoff)}\n"
            "  role: background marker\n"
            f"upper_cutoff: {_format_number(recommended_upper_cutoff)}\n"
            "  role: foreground marker\n"
            f"lower_cutoff_margin: {lower_cutoff_margin}\n"
            f"upper_cutoff_margin: {upper_cutoff_margin}\n"
            "\n"
            "Intensity statistics\n"
            "--------------------\n"
            f"Min: {_format_number(min_val)}\n"
            f"Median: {_format_number(median_val)}\n"
            f"Mean: {_format_number(mean_val)}\n"
            f"95th pct: {_format_number(p95_val)}\n"
            f"99th pct: {_format_number(p99_val)}\n"
            f"99.5th pct: {_format_number(p995_val)}\n"
            f"Max: {_format_number(max_val)}"
        )

        axs[1, 1].text(
            0.02,
            0.98,
            info_text,
            transform=axs[1, 1].transAxes,
            va="top",
            ha="left",
            fontsize=_style_attr(_style, "font_size_small", 8),
            family="monospace",
            # Fixed dark text: this box always has a fixed light
            # (whitesmoke) background regardless of style, so it must not
            # inherit the active style's text_color (white for tissue()),
            # which would make it unreadable against its own background.
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="whitesmoke",
                edgecolor="lightgray",
                alpha=0.95,
            ),
        )

        # -----------------------------------------------------------------
        # Apply SPACEc-style spines/ticks
        # -----------------------------------------------------------------
        for ax in axs.ravel():
            _style_axis(ax, _style)

        for cbar in [cbar0, cbar1]:
            try:
                cbar.ax.tick_params(labelsize=_style_attr(_style, "font_size_small", 8))
                cbar.ax.yaxis.label.set_size(_style_attr(_style, "font_size_small", 8))
            except Exception:
                pass

        # -----------------------------------------------------------------
        # Wrap panel 2/3 titles if they overflow their panel
        # -----------------------------------------------------------------
        # Rendered width depends on both content (downscale_factor, sigma)
        # and the active style's font size, so whether wrapping is needed is
        # conditional, not fixed -- only wrap when a title's *measured*
        # width would exceed its own (already laid out) panel's width.
        fig.canvas.draw()
        _renderer = fig.canvas.get_renderer()
        for _ax, _title_text in [
            (axs[0, 1], _panel2_title),
            (axs[1, 0], _panel3_title),
        ]:
            _max_title_px = _ax.get_window_extent(_renderer).width * 0.92
            _wrapped_title = introduce_line_breaks(
                _title_text,
                _max_title_px,
                _renderer,
                fontsize=_style_attr(_style, "font_size_medium", 12),
            )
            if _wrapped_title != _ax.get_title():
                _ax.set_title(
                    _wrapped_title, fontsize=_style_attr(_style, "font_size_medium", 12)
                )

        # -----------------------------------------------------------------
        # Save/show
        # -----------------------------------------------------------------
        if savefig:
            _log("Saving diagnostic figure.")

            os.makedirs(output_dir, exist_ok=True)

            if output_fname == "":
                output_fname = "tissue_downscaling_diagnostic"

            save_path = os.path.join(
                output_dir,
                output_fname + "_raw_tissue_plot.pdf",
            )

            fig.savefig(save_path, bbox_inches="tight")
            _log(f"Saved diagnostic figure to: {save_path}")

        if showfig:
            _log("Displaying diagnostic figure.")
            plt.show()
        else:
            _log("Closing diagnostic figure without displaying it.")
            plt.close(fig)

    _log("Finished tissue downscaling workflow.")
    _log("Returning scaled down image.")

    if return_cutoffs:
        return resized_im, cutoff_recommendations

    return resized_im
