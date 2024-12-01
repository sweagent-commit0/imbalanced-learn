"""Test the module SMOTE."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_allclose, assert_array_equal
from imblearn.over_sampling import SMOTE
RND_SEED = 0
X = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141], [1.25192108, -0.22367336], [0.53366841, -0.30312976], [1.52091956, -0.49283504], [-0.28162401, -2.10400981], [0.83680821, 1.72827342], [0.3084254, 0.33299982], [0.70472253, -0.73309052], [0.28893132, -0.38761769], [1.15514042, 0.0129463], [0.88407872, 0.35454207], [1.31301027, -0.92648734], [-1.11515198, -0.93689695], [-0.18410027, -0.45194484], [0.9281014, 0.53085498], [-0.14374509, 0.27370049], [-0.41635887, -0.38299653], [0.08711622, 0.93259929], [1.70580611, -0.11219234]])
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 0.0001

def test_generate_samples():
    smote = SMOTE(random_state=RND_SEED)
    smote._validate_estimator()
    X_new, y_new = smote._generate_samples(X, X, np.arange(len(X)), np.array([0, 1]), np.array([0.5, 0.5]), y_type=0)
    
    assert X_new.shape[0] == 2
    assert y_new.shape[0] == 2
    assert np.all(y_new == 0)

def test_in_danger_noise():
    smote = SMOTE(random_state=RND_SEED)
    nn_estimator = NearestNeighbors(n_neighbors=6)
    nn_estimator.fit(X)
    
    # Test 'danger' classification
    danger_samples = smote._in_danger_noise(nn_estimator, X[Y==0], 0, Y, kind='danger')
    assert isinstance(danger_samples, np.ndarray)
    assert danger_samples.dtype == bool
    
    # Test 'noise' classification
    noise_samples = smote._in_danger_noise(nn_estimator, X[Y==0], 0, Y, kind='noise')
    assert isinstance(noise_samples, np.ndarray)
    assert noise_samples.dtype == bool
    
    # Test invalid 'kind' parameter
    try:
        smote._in_danger_noise(nn_estimator, X[Y==0], 0, Y, kind='invalid')
    except ValueError:
        pass
    else:
        assert False, "ValueError not raised for invalid 'kind' parameter"

def test_smote_fit_resample():
    smote = SMOTE(random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    
    assert X_resampled.shape[0] > X.shape[0]
    assert y_resampled.shape[0] > Y.shape[0]
    assert np.sum(y_resampled == 0) == np.sum(y_resampled == 1)
