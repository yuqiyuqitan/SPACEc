"""Unit tests for spacec.plotting._style (SPACEcStyle and helpers).

Tests that modify module-level state (set_style) save and restore _default_style
in try/finally blocks to prevent pollution between tests.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import pytest

import spacec.plotting._style as _style_module
from spacec.plotting._style import (
    SPACEcStyle,
    _active_style,
    _apply_style,
    _get_palette,
    get_active_style,
    set_style,
    style,
)

# ---------------------------------------------------------------------------
# Factory presets
# ---------------------------------------------------------------------------


def test_publication_preset_is_spacecstyle():
    assert isinstance(SPACEcStyle.publication(), SPACEcStyle)


def test_publication_preset_default_values():
    s = SPACEcStyle.publication()
    assert s.font_size_small == 7.0
    assert s.figure_dpi == 300
    assert s.background_color == "white"
    assert s.grid is False


def test_presentation_preset_has_larger_fonts():
    s = SPACEcStyle.presentation()
    assert s.font_size_small == 14.0
    assert s.font_size_large == 18.0
    assert s.figure_dpi == 150
    assert s.legend_frameon is True


def test_default_preset_legacy_font_sizes():
    s = SPACEcStyle.default()
    assert s.font_size_small == 14.0
    assert s.font_size_medium == 16.0
    assert s.font_size_large == 18.0
    assert s.seaborn_style == "ticks"


def test_tissue_preset_dark_background():
    s = SPACEcStyle.tissue()
    assert s.background_color == "#111111"
    assert s.scatter_size_cell == 6.0
    assert s.seaborn_style == "dark"
    assert s.grid is False


def test_presets_are_independent_instances():
    s1 = SPACEcStyle.publication()
    s2 = SPACEcStyle.publication()
    assert s1 is not s2
    # modifying one should not affect the other
    s1.categorical_palette = ["#ffffff"]
    assert s2.categorical_palette != ["#ffffff"]


# ---------------------------------------------------------------------------
# to_rcparams()
# ---------------------------------------------------------------------------


def test_to_rcparams_returns_dict():
    assert isinstance(SPACEcStyle.publication().to_rcparams(), dict)


def test_to_rcparams_expected_keys():
    rcp = SPACEcStyle.publication().to_rcparams()
    for key in (
        "font.family",
        "font.size",
        "axes.titlesize",
        "figure.dpi",
        "legend.frameon",
        "image.cmap",
        "axes.grid",
    ):
        assert key in rcp, f"expected key '{key}' in rcparams"


def test_to_rcparams_values_match_style():
    s = SPACEcStyle.publication()
    rcp = s.to_rcparams()
    assert rcp["font.size"] == s.font_size_small
    assert rcp["figure.dpi"] == s.figure_dpi
    assert rcp["axes.grid"] == s.grid


def test_to_rcparams_legend_fontsize_none_uses_small():
    s = SPACEcStyle()  # legend_fontsize=None by default
    rcp = s.to_rcparams()
    assert rcp["legend.fontsize"] == s.font_size_small


def test_to_rcparams_legend_fontsize_explicit():
    s = SPACEcStyle(legend_fontsize=20.0)
    rcp = s.to_rcparams()
    assert rcp["legend.fontsize"] == 20.0


# ---------------------------------------------------------------------------
# from_dict()
# ---------------------------------------------------------------------------


def test_from_dict_basic():
    s = SPACEcStyle.from_dict({"font_size_small": 12.0, "figure_dpi": 150})
    assert isinstance(s, SPACEcStyle)
    assert s.font_size_small == 12.0
    assert s.figure_dpi == 150


def test_from_dict_ignores_unknown_keys():
    s = SPACEcStyle.from_dict(
        {"font_size_small": 9.0, "totally_unknown_key": "garbage"}
    )
    assert s.font_size_small == 9.0
    assert not hasattr(s, "totally_unknown_key")


def test_from_dict_empty_uses_defaults():
    s = SPACEcStyle.from_dict({})
    assert s.font_size_small == SPACEcStyle.publication().font_size_small


# ---------------------------------------------------------------------------
# get_active_style()
# ---------------------------------------------------------------------------


def test_get_active_style_returns_publication_when_no_default():
    original = _style_module._default_style
    try:
        _style_module._default_style = None
        result = get_active_style()
        assert isinstance(result, SPACEcStyle)
        assert result.font_size_small == 7.0
    finally:
        _style_module._default_style = original


# ---------------------------------------------------------------------------
# set_style() + persistence
# ---------------------------------------------------------------------------


def test_set_style_persists_module_default():
    original = _style_module._default_style
    try:
        set_style(SPACEcStyle.presentation())
        result = get_active_style()
        assert result.font_size_small == 14.0
        assert result.figure_dpi == 150
    finally:
        _style_module._default_style = original
        mpl.rcParams.update(mpl.rcParamsDefault)


def test_set_style_updates_rcparams():
    original = _style_module._default_style
    try:
        s = SPACEcStyle(figure_dpi=72)
        set_style(s)
        assert mpl.rcParams["figure.dpi"] == 72
    finally:
        _style_module._default_style = original
        mpl.rcParams.update(mpl.rcParamsDefault)


def test_set_style_invalid_file_extension_raises():
    with pytest.raises(ValueError):
        set_style("nonexistent_style.txt")


# ---------------------------------------------------------------------------
# style() context manager
# ---------------------------------------------------------------------------


def test_style_context_manager_overrides_and_restores():
    original = _style_module._default_style
    try:
        _style_module._default_style = None
        with style(SPACEcStyle.presentation()) as s:
            assert s.font_size_small == 14.0
            # ContextVar should be set inside the block
            assert _active_style.get() is not None
            assert _active_style.get().font_size_small == 14.0
        # After exiting, ContextVar should be reset
        assert _active_style.get() is None
    finally:
        _style_module._default_style = original
        mpl.rcParams.update(mpl.rcParamsDefault)


def test_style_context_manager_restores_on_exception():
    try:
        with style(SPACEcStyle.presentation()):
            raise RuntimeError("intentional test error")
    except RuntimeError:
        pass
    # ContextVar must be reset even after exception
    assert _active_style.get() is None


# ---------------------------------------------------------------------------
# _apply_style()
# ---------------------------------------------------------------------------


def test_apply_style_none_returns_active():
    original = _style_module._default_style
    try:
        _style_module._default_style = None
        result = _apply_style(None)
        assert isinstance(result, SPACEcStyle)
        assert result.font_size_small == 7.0  # publication defaults
    finally:
        _style_module._default_style = original


def test_apply_style_passthrough():
    s = SPACEcStyle.tissue()
    result = _apply_style(s)
    assert (
        result is s
    ), "_apply_style should return the identical object when given a SPACEcStyle"


# ---------------------------------------------------------------------------
# _get_palette()
# ---------------------------------------------------------------------------


def test_get_palette_small_n_uses_categorical():
    s = SPACEcStyle.publication()
    palette = _get_palette(s, 4)
    assert len(palette) == 4
    assert palette[0] == s.categorical_palette[0]


def test_get_palette_large_n_uses_large_palette():
    s = SPACEcStyle.publication()
    n = len(s.categorical_palette) + 5  # more than 8
    palette = _get_palette(s, n)
    assert len(palette) == n
    assert all(isinstance(c, (str, tuple)) for c in palette)


def test_get_palette_with_adata_uns_colors():
    import anndata
    import numpy as np

    s = SPACEcStyle.publication()
    adata = anndata.AnnData(X=np.zeros((3, 2)))
    adata.uns["cell_type_colors"] = ["#aaaaaa", "#bbbbbb", "#cccccc"]
    palette = _get_palette(s, 2, adata=adata, col="cell_type")
    assert len(palette) == 2
    assert palette[0] == "#aaaaaa"
    assert palette[1] == "#bbbbbb"
