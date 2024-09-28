"""Implement generators for ``tensorflow`` which will balance the data."""
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.utils import _safe_indexing, check_random_state
from ..under_sampling import RandomUnderSampler
from ..utils import Substitution
from ..utils._docstring import _random_state_docstring

@Substitution(random_state=_random_state_docstring)
def balanced_batch_generator(X, y, *, sample_weight=None, sampler=None, batch_size=32, keep_sparse=False, random_state=None):
    """Create a balanced batch generator to train tensorflow model.

    Returns a generator --- as well as the number of step per epoch --- to
    iterate to get the mini-batches. The sampler defines the sampling strategy
    used to balance the dataset ahead of creating the batch. The sampler should
    have an attribute ``sample_indices_``.

    .. versionadded:: 0.4

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Original imbalanced dataset.

    y : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Associated targets.

    sample_weight : ndarray of shape (n_samples,), default=None
        Sample weight.

    sampler : sampler object, default=None
        A sampler instance which has an attribute ``sample_indices_``.
        By default, the sampler used is a
        :class:`~imblearn.under_sampling.RandomUnderSampler`.

    batch_size : int, default=32
        Number of samples per gradient update.

    keep_sparse : bool, default=False
        Either or not to conserve or not the sparsity of the input ``X``. By
        default, the returned batches will be dense.

    {random_state}

    Returns
    -------
    generator : generator of tuple
        Generate batch of data. The tuple generated are either (X_batch,
        y_batch) or (X_batch, y_batch, sampler_weight_batch).

    steps_per_epoch : int
        The number of samples per epoch.
    """
    pass