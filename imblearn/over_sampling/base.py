"""
Base class for the over-sampling method.
"""
import numbers
from collections.abc import Mapping
from ..base import BaseSampler
from ..utils._param_validation import Interval, StrOptions

class BaseOverSampler(BaseSampler):
    """Base class for over-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    _sampling_type = 'over-sampling'
    _sampling_strategy_docstring = "sampling_strategy : float, str, dict or callable, default='auto'\n        Sampling information to resample the data set.\n\n        - When ``float``, it corresponds to the desired ratio of the number of\n          samples in the minority class over the number of samples in the\n          majority class after resampling. Therefore, the ratio is expressed as\n          :math:`\\alpha_{os} = N_{rm} / N_{M}` where :math:`N_{rm}` is the\n          number of samples in the minority class after resampling and\n          :math:`N_{M}` is the number of samples in the majority class.\n\n            .. warning::\n               ``float`` is only available for **binary** classification. An\n               error is raised for multi-class classification.\n\n        - When ``str``, specify the class targeted by the resampling. The\n          number of samples in the different classes will be equalized.\n          Possible choices are:\n\n            ``'minority'``: resample only the minority class;\n\n            ``'not minority'``: resample all classes but the minority class;\n\n            ``'not majority'``: resample all classes but the majority class;\n\n            ``'all'``: resample all classes;\n\n            ``'auto'``: equivalent to ``'not majority'``.\n\n        - When ``dict``, the keys correspond to the targeted classes. The\n          values correspond to the desired number of samples for each targeted\n          class.\n\n        - When callable, function taking ``y`` and returns a ``dict``. The keys\n          correspond to the targeted classes. The values correspond to the\n          desired number of samples for each class.\n        ".strip()
    _parameter_constraints: dict = {'sampling_strategy': [Interval(numbers.Real, 0, 1, closed='right'), StrOptions({'auto', 'minority', 'not minority', 'not majority', 'all'}), Mapping, callable], 'random_state': ['random_state']}