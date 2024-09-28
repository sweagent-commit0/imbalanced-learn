"""Test for the deprecation helper"""
import pytest
from imblearn.utils.deprecation import deprecate_parameter

class Sampler:

    def __init__(self):
        self.a = 'something'
        self.b = 'something'