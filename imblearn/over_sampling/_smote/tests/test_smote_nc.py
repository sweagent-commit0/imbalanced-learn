"""Test the module SMOTENC."""
from collections import Counter
import numpy as np
import pytest
import sklearn
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import parse_version
from imblearn.over_sampling import SMOTENC
from imblearn.utils.estimator_checks import _set_checking_parameters, check_param_validation
sklearn_version = parse_version(sklearn.__version__)

def test_smotenc_categorical_encoder():
    """Check that we can pass our own categorical encoder."""
    pass

def test_smotenc_deprecation_ohe_():
    """Check that we raise a deprecation warning when using `ohe_`."""
    pass

def test_smotenc_param_validation():
    """Check that we validate the parameters correctly since this estimator requires
    a specific parameter.
    """
    pass

def test_smotenc_bool_categorical():
    """Check that we don't try to early convert the full input data to numeric when
    handling a pandas dataframe.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/974
    """
    pass

def test_smotenc_categorical_features_str():
    """Check that we support array-like of strings for `categorical_features` using
    pandas dataframe.
    """
    pass

def test_smotenc_categorical_features_auto():
    """Check that we can automatically detect categorical features based on pandas
    dataframe.
    """
    pass

def test_smote_nc_categorical_features_auto_error():
    """Check that we raise a proper error when we cannot use the `'auto'` mode."""
    pass