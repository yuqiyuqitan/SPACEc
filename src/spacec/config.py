from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlottingConfig:
    default_region_label: str = "region1"
    cca_save_name: str = "CCA_vis.png"
    cca_p_threshold: float = 0.1
    cca_palette: str = "bright"
    cca_palette_size: int = 50


@dataclass
class PreprocessingConfig:
    arcsin_cofactor: int = 150


@dataclass
class RuntimeConfig:
    cuda_version: str = "12"


@dataclass
class SpacecConfig:
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


SPACEC_CONFIG = SpacecConfig()
