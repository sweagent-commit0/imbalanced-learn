"""Test the module under sampler."""
from collections import Counter
from datetime import datetime
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import _convert_container, assert_allclose, assert_array_equal
from imblearn.over_sampling import RandomOverSampler
RND_SEED = 0

@pytest.mark.parametrize('sampling_strategy', ['auto', 'minority', 'not minority', 'not majority', 'all'])
def test_random_over_sampler_strings(sampling_strategy):
    """Check that we support all supposed strings as `sampling_strategy` in
    a sampler inheriting from `BaseOverSampler`."""
    pass

def test_random_over_sampling_datetime():
    """Check that we don't convert input data and only sample from it."""
    pass

def test_random_over_sampler_full_nat():
    """Check that we can return timedelta columns full of NaT.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/1055
    """
    pass