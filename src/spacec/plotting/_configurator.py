from __future__ import annotations

import dataclasses

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from ._style import SPACEcStyle, preview_style


def style_configurator(initial_style: SPACEcStyle | None = None) -> None:
    """Launch the interactive SPACEc style configurator.

    Renders a Jupyter widget UI with live-updating preview plots.

    Parameters
    ----------
    initial_style:
        Starting style. Defaults to ``SPACEcStyle.publication()``.
    """
    DEFAULTS_STYLE = (
        initial_style if initial_style is not None else SPACEcStyle.publication()
    )
    S = dataclasses.asdict(DEFAULTS_STYLE)
    S["categorical_palette"] = list(S["categorical_palette"])

    _w: dict = {}
    FIG_OUT = widgets.Output(layout=widgets.Layout(width="100%", overflow="hidden"))
    yaml_out = widgets.Output()

    plt.ioff()

    # ── live-update engine ────────────────────────────────────────────────

    def refresh_plots(_=None):
        """
        Rebuild all preview plots from the current style state ``S``.

        Parameters
        ----------
        _ : object, optional
            Unused change event passed by widget observers. Default is None.
        """
        sty = SPACEcStyle.from_dict(S)
        with FIG_OUT:
            clear_output(wait=True)
            fig = preview_style(sty)
            plt.show()
            plt.close(fig)

    # ── widget factory helpers ────────────────────────────────────────────

    def fslider(key, label, mn, mx, step=0.05, fmt=".2f"):
        """
        Create a labelled float-slider widget bound to a style key.

        Parameters
        ----------
        key : str
            Key in the style state dict ``S`` to read from and write to.
        label : str
            Human-readable label displayed to the left of the slider.
        mn : float
            Minimum slider value.
        mx : float
            Maximum slider value.
        step : float, optional
            Slider step increment. Default is 0.05.
        fmt : str, optional
            Readout format string for the displayed value. Default is ``".2f"``.

        Returns
        -------
        widgets.HBox
            Horizontal box containing the label and the float slider.
        """
        w = widgets.FloatSlider(
            value=S[key],
            min=mn,
            max=mx,
            step=step,
            description="",
            continuous_update=False,
            layout=widgets.Layout(width="230px"),
            readout=True,
            readout_format=fmt,
            style={"description_width": "0px"},
        )
        _w[key] = w
        lbl = widgets.Label(label, layout=widgets.Layout(width="210px"))

        def _cb(change):
            S[key] = change["new"]
            refresh_plots()

        w.observe(_cb, names="value")
        return widgets.HBox(
            [lbl, w],
            layout=widgets.Layout(
                width="100%",
                flex_flow="row wrap",
                align_items="center",
            ),
        )

    def islider(key, label, mn, mx, step=1):
        """
        Create a labelled integer-slider widget bound to a style key.

        Parameters
        ----------
        key : str
            Key in the style state dict ``S`` to read from and write to.
        label : str
            Human-readable label displayed to the left of the slider.
        mn : int
            Minimum slider value.
        mx : int
            Maximum slider value.
        step : int, optional
            Slider step increment. Default is 1.

        Returns
        -------
        widgets.HBox
            Horizontal box containing the label and the integer slider.
        """
        w = widgets.IntSlider(
            value=int(S[key]),
            min=mn,
            max=mx,
            step=step,
            description="",
            continuous_update=False,
            layout=widgets.Layout(width="230px"),
            readout=True,
            style={"description_width": "0px"},
        )
        _w[key] = w
        lbl = widgets.Label(label, layout=widgets.Layout(width="210px"))

        def _cb(change):
            S[key] = change["new"]
            refresh_plots()

        w.observe(_cb, names="value")
        return widgets.HBox(
            [lbl, w],
            layout=widgets.Layout(
                width="100%",
                flex_flow="row wrap",
                align_items="center",
            ),
        )

    def dropdown(key, label, options):
        """
        Create a labelled dropdown widget bound to a style key.

        Parameters
        ----------
        key : str
            Key in the style state dict ``S`` to read from and write to.
        label : str
            Human-readable label displayed to the left of the dropdown.
        options : list
            Sequence of string options shown in the dropdown menu.

        Returns
        -------
        widgets.HBox
            Horizontal box containing the label and the dropdown.
        """
        w = widgets.Dropdown(
            value=S[key],
            options=options,
            description="",
            layout=widgets.Layout(width="230px"),
            style={"description_width": "0px"},
        )
        _w[key] = w
        lbl = widgets.Label(label, layout=widgets.Layout(width="210px"))

        def _cb(change):
            S[key] = change["new"]
            refresh_plots()

        w.observe(_cb, names="value")
        return widgets.HBox(
            [lbl, w],
            layout=widgets.Layout(
                width="100%",
                flex_flow="row wrap",
                align_items="center",
            ),
        )

    def toggle(key, label):
        """
        Create a checkbox widget bound to a boolean style key.

        Parameters
        ----------
        key : str
            Key in the style state dict ``S`` to read from and write to.
            The value must be a boolean.
        label : str
            Description text shown next to the checkbox.

        Returns
        -------
        widgets.Checkbox
            Checkbox widget that updates ``S[key]`` on change.
        """
        w = widgets.Checkbox(
            value=S[key],
            description=label,
            indent=False,
            layout=widgets.Layout(width="100%", max_width="440px"),
            style={"description_width": "initial"},
        )
        _w[key] = w

        def _cb(change):
            S[key] = change["new"]
            refresh_plots()

        w.observe(_cb, names="value")
        return w

    def colorpicker(key, label):
        """
        Create a labelled colour-picker widget bound to a style key.

        Parameters
        ----------
        key : str
            Key in the style state dict ``S`` to read from and write to.
            The value must be a CSS hex colour string (e.g. ``"#ffffff"``).
        label : str
            Human-readable label displayed to the left of the picker.

        Returns
        -------
        widgets.HBox
            Horizontal box containing the label and the colour picker.
        """
        w = widgets.ColorPicker(
            value=S[key],
            concise=False,
            description="",
            layout=widgets.Layout(width="230px"),
            style={"description_width": "0px"},
        )
        _w[key] = w
        lbl = widgets.Label(label, layout=widgets.Layout(width="210px"))

        def _cb(change):
            S[key] = change["new"]
            refresh_plots()

        w.observe(_cb, names="value")
        return widgets.HBox(
            [lbl, w],
            layout=widgets.Layout(
                width="100%",
                flex_flow="row wrap",
                align_items="center",
            ),
        )

    def palette_widget():
        """
        Build a row of 8 colour pickers for the categorical palette.

        Each picker is bound to the corresponding index of
        ``S["categorical_palette"]`` and triggers a live plot refresh on
        change.

        Returns
        -------
        widgets.VBox
            Vertical box with a descriptive label above a responsive grid
            of colour pickers.
        """
        pickers = []
        for i, hex_col in enumerate(S["categorical_palette"]):
            cp = widgets.ColorPicker(
                value=hex_col,
                concise=True,
                description=f"{i + 1}",
                layout=widgets.Layout(width="78px"),
            )
            _w[f"categorical_palette_{i}"] = cp

            def _cb(change, idx=i):
                S["categorical_palette"][idx] = change["new"]
                refresh_plots()

            cp.observe(_cb, names="value")
            pickers.append(cp)

        lbl = widgets.Label(
            "Categorical palette (8 colours)",
            layout=widgets.Layout(width="100%", max_width="440px"),
        )

        picker_grid = widgets.GridBox(
            pickers,
            layout=widgets.Layout(
                width="100%",
                grid_template_columns="repeat(auto-fit, minmax(78px, 78px))",
                grid_gap="4px 6px",
            ),
        )

        return widgets.VBox([lbl, picker_grid], layout=widgets.Layout(width="100%"))

    # ── event handlers ────────────────────────────────────────────────────

    path_w = widgets.Text(
        value="./spacec_style.yaml",
        description="Output path:",
        layout=widgets.Layout(width="440px"),
        style={"description_width": "100px"},
    )
    status_lbl = widgets.HTML(value="")

    def on_save(_):
        """
        Save the current style to the path entered in ``path_w``.

        Reads the output path from the ``path_w`` text widget, serialises
        the current style state ``S`` to YAML, and updates ``status_lbl``
        with a success or error message.

        Parameters
        ----------
        _ : object
            Unused click event passed by the button ``on_click`` callback.
        """
        path = path_w.value.strip()
        if not path:
            status_lbl.value = (
                '<span style="color:red">⚠ Please enter an output path.</span>'
            )
            return
        try:
            SPACEcStyle.from_dict(S).to_yaml(path)
            status_lbl.value = (
                f'<span style="color:green">✓ Style saved to <code>{path}</code></span>'
            )
        except Exception as e:
            status_lbl.value = f'<span style="color:red">✗ Error: {e}</span>'

    def on_reset(_):
        """
        Reset all widgets and the style state to the initial defaults.

        Restores ``S`` to ``DEFAULTS_STYLE``, updates every widget in
        ``_w`` to its default value, and triggers a live plot refresh.

        Parameters
        ----------
        _ : object
            Unused click event passed by the button ``on_click`` callback.
        """
        defaults = dataclasses.asdict(DEFAULTS_STYLE)
        S.update(defaults)
        S["categorical_palette"] = list(DEFAULTS_STYLE.categorical_palette)
        for key, w in _w.items():
            if key.startswith("categorical_palette_"):
                idx = int(key.rsplit("_", 1)[-1])
                w.value = S["categorical_palette"][idx]
            else:
                w.value = S[key]
        status_lbl.value = '<span style="color:#888">↺ Reset to defaults.</span>'
        refresh_plots()

    def on_preview(_):
        """
        Print the current style as a YAML string in the ``yaml_out`` output.

        Serialises the current style state ``S`` and prints the resulting
        YAML text into the ``yaml_out`` Output widget so the user can
        inspect or copy it.

        Parameters
        ----------
        _ : object
            Unused click event passed by the button ``on_click`` callback.
        """
        with yaml_out:
            clear_output(wait=True)
            print(SPACEcStyle.from_dict(S).to_yaml_string())

    # ── build panels ──────────────────────────────────────────────────────

    typo_panel = widgets.VBox(
        [
            dropdown(
                "font_family",
                "Font family",
                [
                    "sans-serif",
                    "serif",
                    "monospace",
                    "Arial",
                    "Helvetica",
                    "DejaVu Sans",
                ],
            ),
            fslider("font_size_tiny", "Tick label size (pt)", 4, 12, 0.5),
            fslider("font_size_small", "Axis label size (pt)", 4, 14, 0.5),
            fslider("font_size_medium", "Panel title size (pt)", 6, 16, 0.5),
            fslider("font_size_large", "Figure title size (pt)", 6, 18, 0.5),
            fslider("font_size_annotation", "Annotation size (pt)", 4, 12, 0.5),
        ],
        layout=widgets.Layout(width="100%"),
    )

    geo_panel = widgets.VBox(
        [
            fslider("figure_width_single", "Single-col width (in)", 2, 6, 0.05),
            fslider("figure_width_onehalf", "1.5-col width (in)", 3, 8, 0.05),
            fslider("figure_width_double", "Double-col width (in)", 4, 10, 0.05),
            fslider("figure_height_default", "Default height (in)", 2, 8, 0.05),
            islider("figure_dpi", "Raster DPI", 72, 600, 50),
        ],
        layout=widgets.Layout(width="100%"),
    )

    lines_panel = widgets.VBox(
        [
            fslider("axes_linewidth", "Spine linewidth", 0.25, 3.0, 0.05),
            fslider("tick_length", "Tick length (pt)", 1, 8, 0.5),
            fslider("tick_width", "Tick width (pt)", 0.1, 2, 0.1),
            dropdown("tick_direction", "Tick direction", ["in", "out", "inout"]),
            fslider("data_linewidth", "Data line width", 0.25, 3.0, 0.05),
            fslider("data_linewidth_box", "Box/bar edge width", 0.25, 3.0, 0.05),
        ],
        layout=widgets.Layout(width="100%"),
    )

    axes_panel = widgets.VBox(
        [
            toggle("remove_top_spine", "Remove top spine"),
            toggle("remove_right_spine", "Remove right spine"),
            colorpicker("background_color", "Background colour"),
            toggle("grid", "Show grid"),
            fslider("grid_alpha", "Grid alpha", 0.0, 1.0, 0.05),
            colorpicker("grid_color", "Grid colour"),
            dropdown(
                "seaborn_style",
                "Seaborn base style",
                ["ticks", "white", "whitegrid", "dark", "darkgrid"],
            ),
        ],
        layout=widgets.Layout(width="100%"),
    )

    marks_panel = widgets.VBox(
        [
            fslider("data_alpha_bar", "Bar fill alpha", 0, 1, 0.05),
            fslider("data_alpha_box", "Box fill alpha", 0, 1, 0.05),
            fslider("data_alpha_background", "Background point alpha", 0, 1, 0.05),
            fslider("data_alpha_spatial", "Spatial scatter alpha", 0, 1, 0.05),
            fslider("scatter_size_cell", "Cell dot size (pt²)", 0.5, 20, 0.5),
            colorpicker("edge_color_default", "Edge / outline colour"),
        ],
        layout=widgets.Layout(width="100%"),
    )

    colours_panel = widgets.VBox(
        [
            palette_widget(),
            dropdown(
                "large_categorical_palette",
                "Large palette (>8)",
                ["tab20", "tab20b", "tab20c", "Set1", "Set2", "Paired", "husl"],
            ),
            dropdown(
                "diverging_cmap",
                "Diverging colourmap",
                ["RdBu_r", "RdBu", "bwr", "seismic", "coolwarm", "PiYG"],
            ),
            dropdown(
                "sequential_cmap",
                "Sequential colourmap",
                [
                    "viridis",
                    "magma",
                    "plasma",
                    "inferno",
                    "cividis",
                    "YlOrRd",
                    "Blues",
                    "Greens",
                ],
            ),
            dropdown(
                "correlation_cmap",
                "Correlation colourmap",
                ["coolwarm", "RdBu_r", "bwr", "PiYG", "PRGn"],
            ),
            colorpicker("accent_color_positive", "Accent — positive/A"),
            colorpicker("accent_color_negative", "Accent — negative/B"),
            colorpicker("accent_color_neutral", "Accent — neutral"),
        ],
        layout=widgets.Layout(width="100%"),
    )

    heatmap_panel = widgets.VBox(
        [
            fslider("heatmap_vmin", "Colourscale min", -10, 0, 0.5),
            fslider("heatmap_vmax", "Colourscale max", 0, 10, 0.5),
            fslider("colorbar_fraction", "Colorbar width", 0.01, 0.15, 0.005),
            fslider("colorbar_pad", "Colorbar gap", 0.01, 0.20, 0.005),
        ],
        layout=widgets.Layout(width="100%"),
    )

    network_panel = widgets.VBox(
        [
            fslider("network_node_size", "Node size (pt²)", 50, 800, 25),
            colorpicker("network_node_default_color", "Default node colour"),
        ],
        layout=widgets.Layout(width="100%"),
    )

    layout_panel = widgets.VBox(
        [
            fslider("subplot_wspace", "Horizontal panel gap", 0, 1, 0.05),
            fslider("subplot_hspace", "Vertical panel gap", 0, 1, 0.05),
        ],
        layout=widgets.Layout(width="100%"),
    )

    legend_panel = widgets.VBox(
        [
            toggle("legend_frameon", "Legend frame on"),
            fslider("legend_markerscale", "Legend marker scale", 0.3, 3, 0.1),
        ],
        layout=widgets.Layout(width="100%"),
    )

    save_panel = widgets.VBox(
        [
            dropdown("save_format", "Save format", ["pdf", "png", "svg"]),
            toggle("save_transparent", "Transparent background"),
        ],
        layout=widgets.Layout(width="100%"),
    )

    # ── accordion ─────────────────────────────────────────────────────────

    SECTIONS = [
        ("Typography", typo_panel),
        ("Figure geometry", geo_panel),
        ("Lines & ticks", lines_panel),
        ("Axes appearance", axes_panel),
        ("Alpha & marks", marks_panel),
        ("Colours & palettes", colours_panel),
        ("Heatmap", heatmap_panel),
        ("Network graph", network_panel),
        ("Multi-panel layout", layout_panel),
        ("Legend", legend_panel),
        ("Save options", save_panel),
    ]

    accordion = widgets.Accordion(
        children=[p for _, p in SECTIONS],
        layout=widgets.Layout(width="100%"),
    )
    for i, (title, _) in enumerate(SECTIONS):
        accordion.set_title(i, title)
    accordion.selected_index = None

    # ── button row ────────────────────────────────────────────────────────

    save_btn = widgets.Button(
        description="Save style file",
        button_style="success",
        layout=widgets.Layout(width="160px", height="36px"),
    )
    reset_btn = widgets.Button(
        description="Reset defaults",
        button_style="warning",
        layout=widgets.Layout(width="140px", height="36px"),
    )
    preview_btn = widgets.Button(
        description="Preview YAML",
        button_style="",
        layout=widgets.Layout(width="130px", height="36px"),
    )

    save_btn.on_click(on_save)
    reset_btn.on_click(on_reset)
    preview_btn.on_click(on_preview)

    # ── final layout ──────────────────────────────────────────────────────

    responsive_css = widgets.HTML(
        """
        <style>
        .spacec-style-configurator {
            box-sizing: border-box;
            max-width: 100%;
        }

        .spacec-style-configurator * {
            box-sizing: border-box;
        }

        .spacec-style-configurator img,
        .spacec-style-configurator svg {
            max-width: 100% !important;
            height: auto !important;
        }

        .spacec-style-configurator .jupyter-widgets-output-area,
        .spacec-style-configurator .output,
        .spacec-style-configurator .output_wrapper,
        .spacec-style-configurator .output_area {
            max-width: 100% !important;
            overflow-x: hidden !important;
        }
        </style>
        """
    )

    header = widgets.HTML(
        "<h2 style='margin-bottom:4px'>SPACEc Style Configurator</h2>"
        "<p style='color:#555;margin-top:0'>"
        "Adjust settings in the accordion panels on the left. "
        "All four example plots update live. "
        "When satisfied, enter a path and click <b>Save style file</b>."
        "</p>"
    )

    path_w.layout = widgets.Layout(
        flex="1 1 320px",
        min_width="260px",
        max_width="440px",
    )

    save_row = widgets.HBox(
        [path_w, save_btn, preview_btn, reset_btn],
        layout=widgets.Layout(
            width="100%",
            align_items="center",
            flex_flow="row wrap",
            gap="6px",
        ),
    )

    left_col = widgets.VBox(
        [accordion],
        layout=widgets.Layout(
            flex="0 1 490px",
            width="100%",
            max_width="490px",
            overflow_y="auto",
            overflow_x="hidden",
            max_height="700px",
        ),
    )

    right_col = widgets.VBox(
        [FIG_OUT],
        layout=widgets.Layout(
            flex="1 1 520px",
            width="100%",
            min_width="0",
            overflow="hidden",
        ),
    )

    main_row = widgets.HBox(
        [left_col, right_col],
        layout=widgets.Layout(
            width="100%",
            align_items="flex-start",
            flex_flow="row wrap",
            gap="16px",
            overflow="hidden",
        ),
    )

    ui = widgets.VBox(
        [responsive_css, header, main_row, save_row, status_lbl, yaml_out],
        layout=widgets.Layout(
            width="100%",
            padding="12px",
            overflow="hidden",
        ),
    )

    ui.add_class("spacec-style-configurator")

    refresh_plots()
    display(ui)
