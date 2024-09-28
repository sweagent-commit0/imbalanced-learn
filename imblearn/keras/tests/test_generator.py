import numpy as np
import pytest
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
keras = pytest.importorskip('keras')
from keras.layers import Dense
from keras.models import Sequential
from imblearn.datasets import make_imbalance
from imblearn.keras import BalancedBatchGenerator, balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, NearMiss
3