"""Test the module one-sided selection."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils._testing import assert_array_equal
from imblearn.under_sampling import OneSidedSelection
RND_SEED = 0
X = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189], [-0.77740357, 0.74097941], [0.91542919, -0.65453327], [-0.03852113, 0.40910479], [-0.43877303, 1.07366684], [-0.85795321, 0.82980738], [-0.18430329, 0.52328473], [-0.30126957, -0.66268378], [-0.65571327, 0.42412021], [-0.28305528, 0.30284991], [0.20246714, -0.34727125], [1.06446472, -1.09279772], [0.30543283, -0.02589502], [-0.00717161, 0.00318087]])
Y = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0])

def test_one_sided_selection_multiclass():
    """Check the validity of the fitted attributes `estimators_`."""
    pass

def test_one_sided_selection_deprecation():
    """Check that we raise a FutureWarning when accessing the parameter `estimator_`."""
    pass