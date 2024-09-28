"""Test for the testing module"""
import numpy as np
import pytest
from sklearn.neighbors._base import KNeighborsMixin
from imblearn.base import SamplerMixin
from imblearn.utils.testing import _CustomNearestNeighbors, all_estimators

def test_custom_nearest_neighbors():
    """Check that our custom nearest neighbors can be used for our internal
    duck-typing."""
    pass