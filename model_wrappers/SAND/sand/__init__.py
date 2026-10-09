import os
import warnings

# numexpr (imported transitively via pandas) caps its thread count at
# NUMEXPR_MAX_THREADS (default 64). On machines with more cores it emits a
# warning ("nthreads cannot be larger than environment variable
# NUMEXPR_MAX_THREADS"). Raise the cap to the available core count before
# numexpr/pandas is imported to silence it.
if "NUMEXPR_MAX_THREADS" not in os.environ:
    os.environ["NUMEXPR_MAX_THREADS"] = str(os.cpu_count() or 64)

try:
    from sand._version import __version__
except ImportError:
    __version__ = "not-installed"
    warnings.warn(
        "You are running a non-installed version of sand."
        "If you are running this from a git repo, please run `pip install -e .` to install the package."
    )

from sand.index.index import SANDIndex
from sand.model import SANDModel

# Convenient aliases
Index = SANDIndex
Model = SANDModel

__all__ = ["__version__", "SANDIndex", "SANDModel", "Index", "Model"]
