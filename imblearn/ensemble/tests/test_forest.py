import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import parse_version
from imblearn.ensemble import BalancedRandomForestClassifier
sklearn_version = parse_version(sklearn.__version__)

def test_balanced_bagging_classifier_n_features():
    """Check that we raise a FutureWarning when accessing `n_features_`."""
    pass

def test_balanced_random_forest_change_behaviour(imbalanced_dataset):
    """Check that we raise a change of behaviour for the parameters `sampling_strategy`
    and `replacement`.
    """
    pass

@pytest.mark.skipif(parse_version(sklearn_version.base_version) < parse_version('1.4'), reason='scikit-learn should be >= 1.4')
def test_missing_values_is_resilient():
    """Check that forest can deal with missing values and has decent performance."""
    pass

@pytest.mark.skipif(parse_version(sklearn_version.base_version) < parse_version('1.4'), reason='scikit-learn should be >= 1.4')
def test_missing_value_is_predictive():
    """Check that the forest learns when missing values are only present for
    a predictive feature."""
    pass