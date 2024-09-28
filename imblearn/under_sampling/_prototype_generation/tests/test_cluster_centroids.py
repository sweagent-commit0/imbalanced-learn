"""Test the module cluster centroids."""
from collections import Counter
import numpy as np
import pytest
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import ClusterCentroids
from imblearn.utils.testing import _CustomClusterer
RND_SEED = 0
X = np.array([[0.04352327, -0.20515826], [0.92923648, 0.76103773], [0.20792588, 1.49407907], [0.47104475, 0.44386323], [0.22950086, 0.33367433], [0.15490546, 0.3130677], [0.09125309, -0.85409574], [0.12372842, 0.6536186], [0.13347175, 0.12167502], [0.094035, -2.55298982]])
Y = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])
R_TOL = 0.0001