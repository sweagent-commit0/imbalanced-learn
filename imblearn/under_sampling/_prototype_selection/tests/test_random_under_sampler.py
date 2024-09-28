"""Test the module random under sampler."""
from collections import Counter
from datetime import datetime
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_array_equal
from imblearn.under_sampling import RandomUnderSampler
RND_SEED = 0
X = np.array([[0.04352327, -0.20515826], [0.92923648, 0.76103773], [0.20792588, 1.49407907], [0.47104475, 0.44386323], [0.22950086, 0.33367433], [0.15490546, 0.3130677], [0.09125309, -0.85409574], [0.12372842, 0.6536186], [0.13347175, 0.12167502], [0.094035, -2.55298982]])
Y = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])

@pytest.mark.parametrize('sampling_strategy', ['auto', 'majority', 'not minority', 'not majority', 'all'])
def test_random_under_sampler_strings(sampling_strategy):
    """Check that we support all supposed strings as `sampling_strategy` in
    a sampler inheriting from `BaseUnderSampler`."""
    pass

def test_random_under_sampling_datetime():
    """Check that we don't convert input data and only sample from it."""
    pass

def test_random_under_sampler_full_nat():
    """Check that we can return timedelta columns full of NaT.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/1055
    """
    pass