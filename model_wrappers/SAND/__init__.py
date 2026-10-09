import warnings

try:
    from sand._version import __version__
except ImportError:
    __version__ = "not-installed"
    warnings.warn(
        "You are running a non-installed version of sand."
        "If you are running this from a git repo, please run `pip install -e .` to install the package."
    )

__all__ = ["__version__"]
