"""Test utilities."""
import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.neighbors import KDTree
from sklearn.utils._testing import ignore_warnings

def all_estimators(type_filter=None):
    """Get a list of all estimators from imblearn.

    This function crawls the module and gets all classes that inherit
    from BaseEstimator. Classes that are defined in test-modules are not
    included.
    By default meta_estimators are also not included.
    This function is adapted from sklearn.

    Parameters
    ----------
    type_filter : str, list of str, or None, default=None
        Which kind of estimators should be returned. If None, no
        filter is applied and all estimators are returned.  Possible
        values are 'sampler' to get estimators only of these specific
        types, or a list of these to get the estimators that fit at
        least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """
    pass

class _CustomNearestNeighbors(BaseEstimator):
    """Basic implementation of nearest neighbors not relying on scikit-learn.

    `kneighbors_graph` is ignored and `metric` does not have any impact.
    """

    def __init__(self, n_neighbors=1, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def kneighbors_graph(X=None, n_neighbors=None, mode='connectivity'):
        """This method is not used within imblearn but it is required for
        duck-typing."""
        pass

class _CustomClusterer(BaseEstimator):
    """Class that mimics a cluster that does not expose `cluster_centers_`."""

    def __init__(self, n_clusters=1, expose_cluster_centers=True):
        self.n_clusters = n_clusters
        self.expose_cluster_centers = expose_cluster_centers