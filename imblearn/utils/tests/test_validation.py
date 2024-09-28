"""Test for the validation helper"""
from collections import Counter, OrderedDict
import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors._base import KNeighborsMixin
from sklearn.utils._testing import assert_array_equal
from imblearn.utils import check_neighbors_object, check_sampling_strategy, check_target_type
from imblearn.utils._validation import ArraysTransformer, _deprecate_positional_args, _is_neighbors_object
from imblearn.utils.testing import _CustomNearestNeighbors
multiclass_target = np.array([1] * 50 + [2] * 100 + [3] * 25)
binary_target = np.array([1] * 25 + [0] * 100)