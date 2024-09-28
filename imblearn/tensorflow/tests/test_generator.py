import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.utils.fixes import parse_version
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator
from imblearn.under_sampling import NearMiss
tf = pytest.importorskip('tensorflow')