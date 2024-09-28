import numpy as np
import pytest
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_allclose, assert_array_equal
from imblearn.over_sampling import SMOTE, KMeansSMOTE