import pathlib
from typing import Optional, Union

import matplotlib.pyplot as plt

from ._style import SPACEcStyle, _apply_spines, _apply_style


def tissue_lables(
    tissueframe,
    region="region1",
    style: "Optional[Union[SPACEcStyle, str, pathlib.Path]]" = None,
    resized_im=None,
    cutoffs=None,
    savefig=False,
    showfig=True,
    output_dir="./",
    output_fname="",
    figsize=(12, 6),
    point_size=45,
    annotate=True,
    mask_alpha=0.35,
    background_cmap="gray",
    label_cmap="tab20",
    verbose=True,
):
    """
    Plot tissue labels and region labels from a tissue segmentation DataFrame.

    This function is designed to integrate with the previous workflow::

        resized_im, cutoffs = downscale_tissue(..., return_cutoffs=True)
        tissueframe = label_tissue(
            resized_im,
            lower_cutoff=cutoffs["recommended_lower_cutoff"],
            upper_cutoff=cutoffs["recommended_upper_cutoff"],
        )
        tissue_lables(
            tissueframe,
            resized_im=resized_im,
            cutoffs=cutoffs,
        )

    Parameters
    ----------
    tissueframe : pandas.DataFrame
        DataFrame returned by ``label_tissue``. Expected columns are:
        ``"tissue"``, ``"y"``, ``"x"``, and the selected ``region`` column.
    region : str, optional
        Column to use for region labels. Default is ``"region1"``.
    style : SPACEcStyle or path, optional
        Style configuration. Defaults to the active style or publication defaults.
    resized_im : numpy.ndarray, optional
        The downscaled image returned by ``downscale_tissue``. If provided, it is
        shown as a grayscale background behind the tissue/region labels.
    cutoffs : dict, optional
        Cutoff recommendation dictionary returned by
        ``downscale_tissue(..., return_cutoffs=True)``. If provided, the selected
        lower and upper watershed marker cutoffs are printed and added to the figure.
    savefig : bool, optional
        If True, save the figure. Default is False.
    showfig : bool, optional
        If True, display the figure. Default is True.
    output_dir : str, optional
        Directory in which to save the figure. Default is ``"./"``.
    output_fname : str, optional
        Base filename stem used when saving the figure. Default is ``""``.
    figsize : tuple, optional
        Figure size in inches. Default is ``(12, 6)``.
    point_size : int or float, optional
        Size of centroid points. Default is 45.
    annotate : bool, optional
        If True, add text labels at tissue centroids. Default is True.
    mask_alpha : float, optional
        Transparency of the reconstructed label mask overlay. Default is 0.35.
    background_cmap : str, optional
        Colormap for the optional ``resized_im`` background. Default is ``"gray"``.
    label_cmap : str, optional
        Colormap for tissue and region label masks. Default is ``"tab20"``.
    verbose : bool, optional
        If True, print an explanation of what the function is doing. Default is True.

    Returns
    -------
    None
    """

    import os

    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import numpy as np

    def _log(message):
        if verbose:
            print(f"[tissue_lables] {message}")

    def _format_number(value):
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

    def _format_label(value):
        """
        Format centroid annotations nicely.
        """
        try:
            numeric_value = float(value)
            if numeric_value.is_integer():
                return str(int(numeric_value))
            return f"{numeric_value:.3g}"
        except Exception:
            return str(value)

    def _safe_style_attr(style_obj, attr, default):
        return getattr(style_obj, attr, default)

    def _add_text_with_outline(ax, x, y, text, fontsize):
        txt = ax.annotate(
            text,
            (x, y),
            fontsize=fontsize,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=30,
        )
        txt.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ]
        )

    def _prepare_axis(ax, title, width, height, style_obj):
        ax.set_title(
            title, fontsize=_safe_style_attr(style_obj, "font_size_medium", 12)
        )
        ax.set_xlabel("Downscaled X pixel")
        ax.set_ylabel("Downscaled Y pixel")

        # Match image coordinates: x increases rightward, y increases downward.
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        # set_box_aspect (not set_aspect(..., adjustable="box")): both keep
        # the data undistorted, but box_aspect is the mechanism
        # constrained_layout actually accounts for when sizing the axes, so
        # it shrinks the panel's allocated space to fit the box instead of
        # leaving it letterboxed inside whatever figsize was requested --
        # removing the need to pre-compute a matching figsize per panel
        # count/data shape (see panel_figsize below).
        ax.set_box_aspect(height / width)

        try:
            _apply_spines(ax, style_obj)
        except Exception:
            # Keep the function usable even if the external style helper is unavailable.
            pass

    def _show_background(ax, image):
        if image is None:
            return

        vals = image[np.isfinite(image)]
        if vals.size == 0:
            return

        vmin, vmax = np.percentile(vals, [1, 99.5])

        ax.imshow(
            image,
            cmap=background_cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            zorder=1,
        )

    def _show_mask(ax, mask):
        masked = np.ma.masked_where(mask == 0, mask)
        ax.imshow(
            masked,
            cmap=label_cmap,
            interpolation="nearest",
            alpha=mask_alpha,
            zorder=5,
        )

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------
    _log("Starting tissue label plotting.")
    _log("This function visualizes the segmentation output from label_tissue().")

    required_cols = {"tissue", "y", "x"}
    missing_cols = required_cols - set(tissueframe.columns)

    if len(missing_cols) > 0:
        raise ValueError(
            "tissueframe is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    if region not in tissueframe.columns:
        raise ValueError(
            f"Requested region column '{region}' was not found in tissueframe. "
            f"Available columns are: {list(tissueframe.columns)}"
        )

    if tissueframe.shape[0] == 0:
        raise ValueError("tissueframe is empty; there are no tissue labels to plot.")

    _log(f"Using region column: {region}")
    _log(f"Number of segmented pixels in tissueframe: {tissueframe.shape[0]}")

    # label_tissue initializes the region column as a copy of the tissue column, so
    # before regions are assigned manually both panels would show the same plot.
    region_equals_tissue = tissueframe[region].equals(tissueframe["tissue"])

    if region_equals_tissue:
        _log(
            f"Column '{region}' is still identical to 'tissue'. Showing a single panel. "
            f"Assign tissue pieces to regions (e.g. tissueframe['{region}'] = "
            "tissueframe['tissue'].map({1: 1, 2: 1, 3: 2})) to see the region panel."
        )

    _style = _apply_style(style)

    # ------------------------------------------------------------------
    # Determine image size
    # ------------------------------------------------------------------
    max_y_from_frame = int(np.nanmax(tissueframe["y"].values)) + 1
    max_x_from_frame = int(np.nanmax(tissueframe["x"].values)) + 1

    if resized_im is not None:
        height = max(max_y_from_frame, resized_im.shape[0])
        width = max(max_x_from_frame, resized_im.shape[1])
        _log(
            "Using resized_im as grayscale background with shape "
            f"{resized_im.shape[0]} x {resized_im.shape[1]} pixels."
        )
    else:
        height = max_y_from_frame
        width = max_x_from_frame
        _log(
            "No resized_im background was provided. "
            "The figure will show reconstructed label masks only."
        )

    _log(f"Plot canvas size: {height} x {width} pixels.")

    # ------------------------------------------------------------------
    # Reconstruct tissue label image from tissueframe
    # ------------------------------------------------------------------
    _log("Reconstructing a label image from tissueframe coordinates.")

    label_img = np.zeros((height, width), dtype=int)

    y_vals = tissueframe["y"].to_numpy(dtype=int)
    x_vals = tissueframe["x"].to_numpy(dtype=int)
    tissue_vals = tissueframe["tissue"].to_numpy(dtype=int)

    valid = (y_vals >= 0) & (y_vals < height) & (x_vals >= 0) & (x_vals < width)

    label_img[y_vals[valid], x_vals[valid]] = tissue_vals[valid]

    n_tissues = len(np.unique(tissue_vals))
    _log(f"Found {n_tissues} tissue piece(s).")

    # ------------------------------------------------------------------
    # Reconstruct region image
    # ------------------------------------------------------------------
    _log("Reconstructing a region-label image.")

    region_values = tissueframe[region].dropna().unique()

    try:
        region_values_sorted = sorted(region_values, key=lambda x: float(x))
    except (TypeError, ValueError):
        # Non-numeric region labels: fall back to string ordering.
        region_values_sorted = sorted(region_values, key=lambda x: str(x))

    region_to_code = {
        region_value: idx + 1 for idx, region_value in enumerate(region_values_sorted)
    }

    region_codes = tissueframe[region].map(region_to_code).to_numpy(dtype=int)

    region_img = np.zeros((height, width), dtype=int)
    region_img[y_vals[valid], x_vals[valid]] = region_codes[valid]

    n_regions = len(region_values_sorted)
    _log(f"Found {n_regions} unique region label(s) in column '{region}'.")

    # ------------------------------------------------------------------
    # Compute centroids
    # ------------------------------------------------------------------
    _log("Computing tissue centroids for label annotation.")

    centroids = tissueframe.groupby("tissue", as_index=True).agg(
        x=("x", "mean"),
        y=("y", "mean"),
        region_value=(region, "first"),
    )

    # ------------------------------------------------------------------
    # Cutoff information from downscale_tissue, if available
    # ------------------------------------------------------------------
    cutoff_text = ""

    if cutoffs is not None:
        lower_cutoff = cutoffs.get(
            "recommended_lower_cutoff",
            cutoffs.get("recommended_background_cutoff", None),
        )
        upper_cutoff = cutoffs.get(
            "recommended_upper_cutoff",
            cutoffs.get("recommended_foreground_cutoff", None),
        )

        cutoff_text = (
            f"lower_cutoff/background marker = {_format_number(lower_cutoff)}; "
            f"upper_cutoff/foreground marker = {_format_number(upper_cutoff)}"
        )

        _log("Received cutoff recommendations from downscale_tissue().")
        _log(f"  lower_cutoff/background marker: {_format_number(lower_cutoff)}")
        _log(f"  upper_cutoff/foreground marker: {_format_number(upper_cutoff)}")
    else:
        _log(
            "No cutoff dictionary was provided. "
            "If available, pass cutoffs from downscale_tissue(..., return_cutoffs=True)."
        )

    # ------------------------------------------------------------------
    # Create integrated figure
    # ------------------------------------------------------------------
    _log("Creating integrated tissue-label and region-label figure.")

    n_panels = 1 if region_equals_tissue else 2

    # Derive panel_figsize from the data's own aspect ratio (box_aspect
    # above keeps each panel undistorted regardless, but *also* starting
    # from a correctly-proportioned figsize avoids leaving the leftover
    # blank space as wasted canvas margin -- bbox_inches="tight" trims that
    # away on save, but showfig's inline display has no such trimming).
    # Title-stack budget computed from the active style's own font sizes
    # (points -> inches, x1.3 for line-height/leading) rather than a flat
    # guess, so it scales with the style instead of over/under-shooting.
    def _line_height_in(pt):
        return pt * 1.3 / 72

    # Both the suptitle and each panel's own title use font_size_medium
    # here (see fig.suptitle/_prepare_axis below): budget 2 stacked lines
    # plus a bit extra, since the gap between them needs more than pure
    # linear text-height scaling once fonts get bigger (confirmed by
    # measurement: 2x alone left publication clean but presentation's
    # suptitle touching the panel title).
    _title_stack_in = 2.6 * _line_height_in(
        _safe_style_attr(_style, "font_size_medium", 12)
    )
    _panel_width = figsize[0] / 2
    _panel_height = _panel_width * (height / width) + _title_stack_in
    if n_panels == 1:
        panel_figsize = (_panel_width, _panel_height)
    else:
        panel_figsize = (figsize[0], _panel_height)

    fig, axs = plt.subplots(
        1,
        n_panels,
        figsize=panel_figsize,
        constrained_layout=True,
        squeeze=False,
    )
    axs = axs[0]

    if n_panels == 1:
        title = "Tissue label overview"
    else:
        title = "Tissue and region label overview"

    if cutoff_text != "":
        fig.suptitle(
            f"{title}\n{cutoff_text}",
            fontsize=_safe_style_attr(_style, "font_size_medium", 12),
            fontweight="bold",
        )
    else:
        fig.suptitle(
            title,
            fontsize=_safe_style_attr(_style, "font_size_medium", 12),
            fontweight="bold",
        )

    # ------------------------------------------------------------------
    # Panel 1: tissue piece labels
    # ------------------------------------------------------------------
    _show_background(axs[0], resized_im)
    _show_mask(axs[0], label_img)

    axs[0].scatter(
        centroids["x"],
        centroids["y"],
        s=point_size,
        c="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=20,
    )

    if annotate:
        for tissue_id, row in centroids.iterrows():
            _add_text_with_outline(
                axs[0],
                row["x"],
                row["y"],
                _format_label(tissue_id),
                fontsize=_safe_style_attr(_style, "font_size_annotation", 9),
            )

    _prepare_axis(
        axs[0],
        "Tissue piece labels",
        width,
        height,
        _style,
    )

    # ------------------------------------------------------------------
    # Panel 2: region labels
    # ------------------------------------------------------------------
    if n_panels == 2:
        _show_background(axs[1], resized_im)
        _show_mask(axs[1], region_img)

        axs[1].scatter(
            centroids["x"],
            centroids["y"],
            s=point_size,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=20,
        )

        if annotate:
            for tissue_id, row in centroids.iterrows():
                _add_text_with_outline(
                    axs[1],
                    row["x"],
                    row["y"],
                    _format_label(row["region_value"]),
                    fontsize=_safe_style_attr(_style, "font_size_annotation", 9),
                )

        _prepare_axis(
            axs[1],
            f"Region labels: {region}",
            width,
            height,
            _style,
        )

    # ------------------------------------------------------------------
    # Save/show
    # ------------------------------------------------------------------
    if savefig:
        _log("Saving tissue label figure.")

        os.makedirs(output_dir, exist_ok=True)

        if output_fname == "":
            output_fname = "tissue_labels"

        save_path = os.path.join(
            output_dir,
            output_fname + "_tissue_region_labels.pdf",
        )

        fig.savefig(save_path, bbox_inches="tight")
        _log(f"Saved figure to: {save_path}")

    if showfig:
        _log("Displaying tissue label figure.")
        plt.show()
    else:
        _log("Closing tissue label figure without displaying it.")
        plt.close(fig)

    _log("Finished tissue label plotting.")
