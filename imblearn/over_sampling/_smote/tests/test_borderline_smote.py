from collections import Counter
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import assert_allclose, assert_array_equal
from imblearn.over_sampling import BorderlineSMOTE

@pytest.mark.parametrize('kind', ['borderline-1', 'borderline-2'])
def test_borderline_smote_no_in_danger_samples(kind):
    """Check that the algorithm behave properly even on a dataset without any sample
    in danger.
    """
    pass

def test_borderline_smote_kind():
    """Check the behaviour of the `kind` parameter.

    In short, "borderline-2" generates sample closer to the boundary decision than
    "borderline-1". We generate an example where a logistic regression will perform
    worse on "borderline-2" than on "borderline-1".
    """
    pass