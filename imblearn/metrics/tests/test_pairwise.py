"""Test for the metrics that perform pairwise distance computation."""
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils._testing import _convert_container
from imblearn.metrics.pairwise import ValueDifferenceMetric