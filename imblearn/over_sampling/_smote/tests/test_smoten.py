import numpy as np
import pytest
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils._testing import _convert_container
from imblearn.over_sampling import SMOTEN

@pytest.mark.parametrize('sparse_format', ['sparse_csr', 'sparse_csc'])
def test_smoten_sparse_input(data, sparse_format):
    """Check that we handle sparse input in SMOTEN even if it is not efficient.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/971
    """
    pass

def test_smoten_categorical_encoder(data):
    """Check that `categorical_encoder` is used when provided."""
    pass