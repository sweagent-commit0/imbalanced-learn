"""Utilities for docstring in imbalanced-learn."""

class Substitution:
    """Decorate a function's or a class' docstring to perform string
    substitution on it.

    This decorator should be robust even if obj.__doc__ is None
    (for example, if -OO was passed to the interpreter)
    """

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise AssertionError('Only positional or keyword args are allowed')
        self.params = args or kwargs

    def __call__(self, obj):
        if obj.__doc__:
            obj.__doc__ = obj.__doc__.format(**self.params)
        return obj
_random_state_docstring = 'random_state : int, RandomState instance, default=None\n        Control the randomization of the algorithm.\n\n        - If int, ``random_state`` is the seed used by the random number\n          generator;\n        - If ``RandomState`` instance, random_state is the random number\n          generator;\n        - If ``None``, the random number generator is the ``RandomState``\n          instance used by ``np.random``.\n    '.rstrip()
_n_jobs_docstring = 'n_jobs : int, default=None\n        Number of CPU cores used during the cross-validation loop.\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See\n        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_\n        for more details.\n    '.rstrip()