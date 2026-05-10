import importlib.util
from pathlib import Path
import sys
import types

import matplotlib
import numpy as np
import pandas as pd
import tifffile

matplotlib.use("Agg")


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "spacec"

spacec_pkg = types.ModuleType("spacec")
spacec_pkg.__path__ = [str(SRC_ROOT)]
sys.modules.setdefault("spacec", spacec_pkg)

for subpkg in ("helperfunctions", "tools", "plotting"):
    module_name = f"spacec.{subpkg}"
    pkg = types.ModuleType(module_name)
    pkg.__path__ = [str(SRC_ROOT / subpkg)]
    sys.modules.setdefault(module_name, pkg)

HF_QPTIFF = _load_module(
    "spacec.helperfunctions._qptiff_converter",
    SRC_ROOT / "helperfunctions" / "_qptiff_converter.py",
)
TL_QPTIFF = _load_module(
    "spacec.tools._qptiff_converter",
    SRC_ROOT / "tools" / "_qptiff_converter.py",
)
PL_QPTIFF = _load_module(
    "spacec.plotting._qptiff_converter",
    SRC_ROOT / "plotting" / "_qptiff_converter.py",
)


def test_downscale_tissue_showfig_savefig(tmp_path, monkeypatch):
    image_path = tmp_path / "input.qptiff"
    image = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
    tifffile.imwrite(image_path, image)
    monkeypatch.setattr(HF_QPTIFF.plt, "show", lambda: None)

    out = HF_QPTIFF.downscale_tissue(
        file_path=image_path,
        DNAslice=0,
        downscale_factor=2,
        sigma=0.0,
        showfig=True,
        savefig=True,
        output_dir=f"{tmp_path}/",
        output_fname="preview",
        figsize=(4, 2),
    )

    assert out.shape == (4, 4)
    assert (tmp_path / "preview_raw_tissue_plot.pdf").exists()


def test_downscale_tissue_show_without_save(tmp_path, monkeypatch):
    image_path = tmp_path / "input_show.qptiff"
    image = np.ones((1, 6, 6), dtype=np.float32)
    tifffile.imwrite(image_path, image)

    show_called = {"value": False}

    def _show():
        show_called["value"] = True

    monkeypatch.setattr(HF_QPTIFF.plt, "show", _show)
    out = HF_QPTIFF.downscale_tissue(
        file_path=image_path,
        DNAslice=0,
        downscale_factor=3,
        sigma=0.0,
        showfig=True,
        savefig=False,
    )

    assert out.shape == (2, 2)
    assert show_called["value"] is True


def test_downscale_tissue_without_figure(tmp_path, monkeypatch):
    image_path = tmp_path / "input_nofig.qptiff"
    image = np.ones((1, 4, 4), dtype=np.float32)
    tifffile.imwrite(image_path, image)

    show_called = {"value": False}

    def _show():
        show_called["value"] = True

    monkeypatch.setattr(HF_QPTIFF.plt, "show", _show)
    out = HF_QPTIFF.downscale_tissue(
        file_path=image_path,
        downscale_factor=2,
        sigma=0.0,
        showfig=False,
        savefig=True,
    )

    assert out.shape == (2, 2)
    assert show_called["value"] is False


def test_label_tissue_and_save_labelled_tissue(tmp_path, monkeypatch):
    monkeypatch.setattr(TL_QPTIFF.plt, "show", lambda: None)
    resized = np.array(
        [
            [0.0, 0.0, 0.2, 0.2],
            [0.0, 0.0, 0.2, 0.2],
            [0.2, 0.2, 0.0, 0.0],
            [0.2, 0.2, 0.0, 0.0],
        ],
        dtype=float,
    )

    tissueframe = TL_QPTIFF.label_tissue(
        resized,
        lower_cutoff=0.05,
        upper_cutoff=0.15,
        showfig=False,
    )
    assert set(tissueframe.columns) == {"tissue", "y", "x", "region1"}
    assert not tissueframe.empty

    image_path = tmp_path / "source.tif"
    source = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)
    tifffile.imwrite(image_path, source)

    output_subdir = tmp_path / "tiles"
    output_subdir.mkdir()
    TL_QPTIFF.save_labelled_tissue(
        filepath=image_path,
        tissueframe=tissueframe,
        region="region1",
        padding=0,
        downscale_factor=1,
        output_dir=str(output_subdir),
        output_fname="",
    )

    files = list(output_subdir.glob("reg00*_X01_Y01_Z01.tif"))
    assert files


def test_label_tissue_showfig_without_save(monkeypatch):
    show_called = {"value": False}

    def _show():
        show_called["value"] = True

    monkeypatch.setattr(TL_QPTIFF.plt, "show", _show)
    resized = np.array([[0.0, 0.2], [0.2, 0.0]], dtype=float)
    TL_QPTIFF.label_tissue(
        resized,
        lower_cutoff=0.05,
        upper_cutoff=0.15,
        showfig=True,
        savefig=False,
    )
    assert show_called["value"] is True


def test_label_tissue_showfig_and_savefig(tmp_path, monkeypatch):
    monkeypatch.setattr(TL_QPTIFF.plt, "show", lambda: None)
    resized = np.array([[0.0, 0.2], [0.2, 0.0]], dtype=float)
    TL_QPTIFF.label_tissue(
        resized,
        lower_cutoff=0.05,
        upper_cutoff=0.15,
        showfig=True,
        savefig=True,
        output_dir=f"{tmp_path}/",
        output_fname="labels",
    )
    assert (tmp_path / "labels_labeled_seg_tissue_plot.pdf").exists()


def test_tissue_labels_plot(monkeypatch):
    monkeypatch.setattr(PL_QPTIFF.plt, "show", lambda: None)
    tissueframe = pd.DataFrame(
        {
            "tissue": [1, 1, 2, 2],
            "x": [0, 1, 4, 5],
            "y": [0, 1, 4, 5],
            "region1": [11, 11, 22, 22],
            "region": [1, 1, 2, 2],
        }
    )

    PL_QPTIFF.tissue_lables(tissueframe=tissueframe, region="region")
