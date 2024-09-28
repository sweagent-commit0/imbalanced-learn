import copy
import numbers
from copy import deepcopy
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _safe_indexing
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import has_fit_parameter
from ..base import _ParamsValidationMixin
from ..pipeline import make_pipeline
from ..under_sampling import RandomUnderSampler
from ..under_sampling.base import BaseUnderSampler
from ..utils import Substitution, check_target_type
from ..utils._docstring import _random_state_docstring
from ..utils._param_validation import Interval, StrOptions
from ..utils.fixes import _fit_context
from ._common import _adaboost_classifier_parameter_constraints
sklearn_version = parse_version(sklearn.__version__)

@Substitution(sampling_strategy=BaseUnderSampler._sampling_strategy_docstring, random_state=_random_state_docstring)
class RUSBoostClassifier(_ParamsValidationMixin, AdaBoostClassifier):
    """Random under-sampling integrated in the learning of AdaBoost.

    During learning, the problem of class balancing is alleviated by random
    under-sampling the sample at each iteration of the boosting algorithm.

    Read more in the :ref:`User Guide <boosting>`.

    .. versionadded:: 0.4

    Parameters
    ----------
    estimator : estimator object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is ``DecisionTreeClassifier(max_depth=1)``.

        .. versionadded:: 0.12

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, default=1.0
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    algorithm : {{'SAMME', 'SAMME.R'}}, default='SAMME.R'
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

        .. deprecated:: 0.12
            `"SAMME.R"` is deprecated and will be removed in version 0.14.
            '"SAMME"' will become the default.

    {sampling_strategy}

    replacement : bool, default=False
        Whether or not to sample randomly with replacement or not.

    {random_state}

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 0.10

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    base_sampler_ : :class:`~imblearn.under_sampling.RandomUnderSampler`
        The base sampler used to generate the subsequent samplers.

    samplers_ : list of :class:`~imblearn.under_sampling.RandomUnderSampler`
        The collection of fitted samplers.

    pipelines_ : list of Pipeline
        The collection of fitted pipelines (samplers + trees).

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : ndarray of shape (n_estimator,)
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of shape (n_estimator,)
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances if supported by the ``base_estimator``.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.9

    See Also
    --------
    BalancedBaggingClassifier : Bagging classifier for which each base
        estimator is trained on a balanced bootstrap.

    BalancedRandomForestClassifier : Random forest applying random-under
        sampling to balance the different bootstraps.

    EasyEnsembleClassifier : Ensemble of AdaBoost classifier trained on
        balanced bootstraps.

    References
    ----------
    .. [1] Seiffert, C., Khoshgoftaar, T. M., Van Hulse, J., & Napolitano, A.
       "RUSBoost: A hybrid approach to alleviating class imbalance." IEEE
       Transactions on Systems, Man, and Cybernetics-Part A: Systems and Humans
       40.1 (2010): 185-197.

    Examples
    --------
    >>> from imblearn.ensemble import RUSBoostClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, n_classes=3,
    ...                            n_informative=4, weights=[0.2, 0.3, 0.5],
    ...                            random_state=0)
    >>> clf = RUSBoostClassifier(random_state=0)
    >>> clf.fit(X, y)
    RUSBoostClassifier(...)
    >>> clf.predict(X)
    array([...])
    """
    if sklearn_version >= parse_version('1.4'):
        _parameter_constraints = copy.deepcopy(AdaBoostClassifier._parameter_constraints)
    else:
        _parameter_constraints = copy.deepcopy(_adaboost_classifier_parameter_constraints)
    _parameter_constraints.update({'sampling_strategy': [Interval(numbers.Real, 0, 1, closed='right'), StrOptions({'auto', 'majority', 'not minority', 'not majority', 'all'}), dict, callable], 'replacement': ['boolean']})
    if 'base_estimator' in _parameter_constraints:
        del _parameter_constraints['base_estimator']

    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', sampling_strategy='auto', replacement=False, random_state=None):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm, random_state=random_state)
        self.estimator = estimator
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        pass

    def _validate_estimator(self):
        """Check the estimator and the n_estimator attribute.

        Sets the `estimator_` attributes.
        """
        pass

    def _make_sampler_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        pass

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        pass

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        pass