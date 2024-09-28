"""
Utility method which prints system info to help with debugging,
and filing issues on GitHub.
Adapted from :func:`sklearn.show_versions`,
which was adapted from :func:`pandas.show_versions`
"""
from .. import __version__

def _get_deps_info():
    """Overview of the installed version of main dependencies
    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    pass

def show_versions(github=False):
    """Print debugging information.

    .. versionadded:: 0.5

    Parameters
    ----------
    github : bool,
        If true, wrap system info with GitHub markup.
    """
    pass