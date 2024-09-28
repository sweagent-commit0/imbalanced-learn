"""Testing the metric for classification with imbalanced dataset"""
from functools import partial
import numpy as np
import pytest
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, cohen_kappa_score, jaccard_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils._testing import assert_allclose, assert_array_equal, assert_no_warnings
from sklearn.utils.validation import check_random_state
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score, macro_averaged_mean_absolute_error, make_index_balanced_accuracy, sensitivity_score, sensitivity_specificity_support, specificity_score
RND_SEED = 42
R_TOL = 0.01

def make_prediction(dataset=None, binary=False):
    """Make some classification predictions on a toy dataset using a SVC
    If binary is True restrict to a binary classification problem instead of a
    multiclass classification problem
    """
    pass