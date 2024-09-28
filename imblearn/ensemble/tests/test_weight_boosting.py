import numpy as np
import pytest
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import parse_version
from imblearn.ensemble import RUSBoostClassifier
sklearn_version = parse_version(sklearn.__version__)