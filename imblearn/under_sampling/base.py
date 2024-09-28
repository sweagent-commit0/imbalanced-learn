"""
Base class for the under-sampling method.
"""
import numbers
from collections.abc import Mapping
from ..base import BaseSampler
from ..utils._param_validation import Interval, StrOptions

class BaseUnderSampler(BaseSampler):
    """Base class for under-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    _sampling_type = 'under-sampling'
    _sampling_strategy_docstring = "sampling_strategy : float, str, dict, callable, default='auto'\n        Sampling information to sample the data set.\n\n        - When ``float``, it corresponds to the desired ratio of the number of\n          samples in the minority class over the number of samples in the\n          majority class after resampling. Therefore, the ratio is expressed as\n          :math:`\\alpha_{us} = N_{m} / N_{rM}` where :math:`N_{m}` is the\n          number of samples in the minority class and\n          :math:`N_{rM}` is the number of samples in the majority class\n          after resampling.\n\n          .. warning::\n             ``float`` is only available for **binary** classification. An\n             error is raised for multi-class classification.\n\n        - When ``str``, specify the class targeted by the resampling. The\n          number of samples in the different classes will be equalized.\n          Possible choices are:\n\n            ``'majority'``: resample only the majority class;\n\n            ``'not minority'``: resample all classes but the minority class;\n\n            ``'not majority'``: resample all classes but the majority class;\n\n            ``'all'``: resample all classes;\n\n            ``'auto'``: equivalent to ``'not minority'``.\n\n        - When ``dict``, the keys correspond to the targeted classes. The\n          values correspond to the desired number of samples for each targeted\n          class.\n\n        - When callable, function taking ``y`` and returns a ``dict``. The keys\n          correspond to the targeted classes. The values correspond to the\n          desired number of samples for each class.\n        ".rstrip()
    _parameter_constraints: dict = {'sampling_strategy': [Interval(numbers.Real, 0, 1, closed='right'), StrOptions({'auto', 'majority', 'not minority', 'not majority', 'all'}), Mapping, callable]}

class BaseCleaningSampler(BaseSampler):
    """Base class for under-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    _sampling_type = 'clean-sampling'
    _sampling_strategy_docstring = "sampling_strategy : str, list or callable\n        Sampling information to sample the data set.\n\n        - When ``str``, specify the class targeted by the resampling. Note the\n          the number of samples will not be equal in each. Possible choices\n          are:\n\n            ``'majority'``: resample only the majority class;\n\n            ``'not minority'``: resample all classes but the minority class;\n\n            ``'not majority'``: resample all classes but the majority class;\n\n            ``'all'``: resample all classes;\n\n            ``'auto'``: equivalent to ``'not minority'``.\n\n        - When ``list``, the list contains the classes targeted by the\n          resampling.\n\n        - When callable, function taking ``y`` and returns a ``dict``. The keys\n          correspond to the targeted classes. The values correspond to the\n          desired number of samples for each class.\n        ".rstrip()
    _parameter_constraints: dict = {'sampling_strategy': [Interval(numbers.Real, 0, 1, closed='right'), StrOptions({'auto', 'majority', 'not minority', 'not majority', 'all'}), list, callable]}