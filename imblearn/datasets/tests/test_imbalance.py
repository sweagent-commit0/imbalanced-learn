"""Test the module easy ensemble."""
from collections import Counter
import numpy as np
import pytest
from sklearn.datasets import load_iris
from imblearn.datasets import make_imbalance