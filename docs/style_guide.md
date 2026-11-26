# SPACEc Style System

One object (`SPACEcStyle`, ~60 parameters) controls how all SPACEc/matplotlib plots look. Set it once, save it, reuse it.

## Quick Start

```python
import spacec as sp
from spacec.plotting import SPACEcStyle

sp.pl.set_style(SPACEcStyle.publication())          # global, stays until changed

with sp.pl.style_context(SPACEcStyle.presentation()):  # temporary, only inside block
    ...

s = SPACEcStyle.publication()                       # customize
s.font_size_small = 9
s.categorical_palette[0] = "#0072B2"
sp.pl.set_style(s)                                  # must pass the modified object
```

## Presets

| Preset           | Use              | Font     | Width (single / double) | DPI | Background     |
| ---------------- | ---------------- | -------- | ----------------------- | --- | -------------- |
| `publication()`  | Journals         | 6–10 pt  | 3.5" / 7.2"             | 300 | White          |
| `presentation()` | Slides, posters  | 12–18 pt | 6" / 12"                | 150 | White          |
| `tissue()`       | Imaging overlays | 6–10 pt  | 3.5" / 7.2"             | 300 | Dark `#111111` |
| `default()`      | Legacy SPACEc    | 14–18 pt | 3.5" / 7.2"             | 300 | White          |

All presets remove top/right spines and have the grid off. The first three are also usable as matplotlib styles: `plt.style.use("spacec_publication")` (`spacec_presentation`, `spacec_tissue`).

## Which Style Is Active?

1. `with sp.pl.style_context(...)` block (thread- and async-safe)
2. last `sp.pl.set_style(...)`
3. `publication()` fallback

Some plot functions also take `style=` directly: `sp.pl.umap(adata, style=SPACEcStyle.presentation())`.

## Parameters

| Group     | Parameters                                                                                                                         |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Fonts     | `font_family`, `font_size_tiny/small/medium/large/annotation`, `text_color`                                                        |
| Figure    | `figure_width_single/onehalf/double`, `figure_height_default`, `figure_dpi`, `figure_dpi_lineart`, `background_color`             |
| Lines     | `data_linewidth`, `data_linewidth_box`, `axes_linewidth`, `edge_color_default`, `tick_width`, `tick_length`, `tick_direction`     |
| Axes      | `remove_top_spine`, `remove_right_spine`, `grid`, `grid_alpha`, `grid_color`, `seaborn_style`                                      |
| Colors    | `categorical_palette` (8 colors, Wong colorblind-safe), `large_categorical_palette`, `diverging_cmap`, `sequential_cmap`, `correlation_cmap`, `accent_color_positive/negative/neutral`, `heatmap_vmin/vmax` |
| Legend    | `legend_frameon`, `legend_fontsize`, `legend_markerscale`                                                                          |
| Alpha     | `data_alpha_bar/box/background/spatial`                                                                                            |
| Layout    | `scatter_size_cell`, `network_node_size`, `network_node_default_color`, `colorbar_fraction`, `colorbar_pad`, `subplot_hspace`, `subplot_wspace` |
| Scale bar | `scalebar_color`, `scalebar_fraction`, `scalebar_location`, `scalebar_label_top`                                                   |
| Saving    | `save_format`, `save_transparent`, `save_bbox_inches`                                                                              |

## Save / Load

```python
s.to_yaml("my_style.yaml")                      # or .to_json("my_style.json")
s = SPACEcStyle.from_yaml("my_style.yaml")      # or .from_json()
sp.pl.set_style("my_style.yaml")                # load + apply directly
```

Keep the YAML in git next to your analysis for reproducible figures.

## Interactive Editor (Jupyter only)

```python
sp.pl.style_configurator()                   # sliders + live 2×2 preview, "Save YAML" button
sp.pl.preview_style(SPACEcStyle.tissue())    # static 2×2 preview
```

## API

| Call                                         | Does                                |
| -------------------------------------------- | ----------------------------------- |
| `sp.pl.set_style(style_or_path)`             | Apply globally                      |
| `sp.pl.style_context(s)`                     | Context manager, temporary          |
| `sp.pl.get_active_style()`                   | Return current style                |
| `s.to_rcparams()`                            | Convert to matplotlib rcParams dict |
| `s.to_yaml(p)` / `s.to_json(p)`              | Save to file                        |
| `SPACEcStyle.from_yaml(p)` / `.from_json(p)` | Load                                |
| `sp.pl.style_configurator()` / `sp.pl.preview_style(s)` | Visual tools             |

## Common Mistakes

* **Style not applied**: plotting happened outside the `with sp.pl.style_context(...)` block.
* **Custom colors ignored**: you called `set_style(SPACEcStyle.publication())` (new object) instead of `set_style(s)`.
* **Widget missing**: only works in Jupyter.
* **YAML not found**: use an absolute path.
* **Element unstyled**: custom plot functions may ignore some parameters; pass `style=` if supported.
