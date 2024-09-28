"""Test the module nearmiss."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_equal
from imblearn.under_sampling import NearMiss
X = np.array([[1.17737838, -0.2002118], [0.4960075, 0.86130762], [-0.05903827, 0.10947647], [0.91464286, 1.61369212], [-0.54619583, 1.73009918], [-0.60413357, 0.24628718], [0.45713638, 1.31069295], [-0.04032409, 3.01186964], [0.03142011, 0.12323596], [0.50701028, -0.17636928], [-0.80809175, -1.09917302], [-0.20497017, -0.26630228], [0.99272351, -0.11631728], [-1.95581933, 0.69609604], [1.15157493, -1.2981518]])
Y = np.array([1, 2, 1, 0, 2, 1, 2, 2, 1, 2, 0, 0, 2, 1, 2])
VERSION_NEARMISS = (1, 2, 3)