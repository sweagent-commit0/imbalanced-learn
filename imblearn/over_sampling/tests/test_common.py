from collections import Counter
import numpy as np
import pytest
from sklearn.cluster import MiniBatchKMeans
from imblearn.over_sampling import ADASYN, SMOTE, SMOTEN, SMOTENC, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE
from imblearn.utils.testing import _CustomNearestNeighbors