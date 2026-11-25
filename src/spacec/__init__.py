from importlib.metadata import PackageNotFoundError, version

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "SPACEc"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# Silences the harmless "OMP: Info #276: omp_set_nested routine deprecated"
# message libomp prints the first time numba (used by pynndescent inside
# sc.pp.neighbors) spins up its OpenMP threading layer. setdefault() so a
# user who explicitly set KMP_WARNINGS themselves isn't overridden, and it
# must happen before any scanpy/numba import triggers that first parallel
# JIT run, hence right up top.
import os

os.environ.setdefault("KMP_WARNINGS", "0")
del os

# Compatibility shim for newer IPython versions where `display` is no longer
# exported from `IPython.core.display`, but older imports still rely on it.
try:
    import IPython.core.display as _ipython_core_display
    from IPython.display import display as _ipython_display

    if not hasattr(_ipython_core_display, "display"):
        _ipython_core_display.display = _ipython_display
except Exception:
    pass

from . import helperfunctions as hf
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl

__all__ = [
    "__version__",
    "pp",
    "tl",
    "hf",
    "pl",
]
