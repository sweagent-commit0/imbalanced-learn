"""Test for score"""
import pytest
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.metrics import geometric_mean_score, make_index_balanced_accuracy, sensitivity_score, specificity_score
R_TOL = 0.01