"""Test the module edited nearest neighbour."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_equal
from imblearn.under_sampling import EditedNearestNeighbours
X = np.array([[2.59928271, 0.93323465], [0.25738379, 0.95564169], [1.42772181, 0.526027], [1.92365863, 0.82718767], [-0.10903849, -0.12085181], [-0.284881, -0.62730973], [0.57062627, 1.19528323], [0.03394306, 0.03986753], [0.78318102, 2.59153329], [0.35831463, 1.33483198], [-0.14313184, -1.0412815], [0.01936241, 0.17799828], [-1.25020462, -0.40402054], [-0.09816301, -0.74662486], [-0.01252787, 0.34102657], [0.52726792, -0.38735648], [0.2821046, -0.07862747], [0.05230552, 0.09043907], [0.15198585, 0.12512646], [0.70524765, 0.39816382]])
Y = np.array([1, 2, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 1, 2, 1])

def test_enn_check_kind_selection():
    """Check that `check_sel="all"` is more conservative than
    `check_sel="mode"`."""
    pass