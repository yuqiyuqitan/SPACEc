import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "SPACEc"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# TODO: hacky!
# NOTE: Be aware of potential side-effects: but that may cause crashes or silently produce incorrect results.
# sources:
# - https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
# - https://stackoverflow.com/questions/55714135/how-can-i-fix-an-omp-error-15-initializing-libiomp5-dylib-but-found-libomp
import os

from . import helperfunctions as hf
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


__all__ = [
    "__version__",
    "pp",
    "tl",
    "hf",
    "pl",
]
