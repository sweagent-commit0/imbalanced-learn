"""Test the module neighbourhood cleaning rule."""
from collections import Counter
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_array_equal
from imblearn.under_sampling import EditedNearestNeighbours, NeighbourhoodCleaningRule

def test_ncr_threshold_cleaning(data):
    """Test the effect of the `threshold_cleaning` parameter."""
    pass

def test_ncr_n_neighbors(data):
    """Check the effect of the NN on the cleaning of the second phase."""
    pass