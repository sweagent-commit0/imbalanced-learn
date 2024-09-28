"""Tests for the minimum dependencies in the README.rst file."""
import os
import platform
import re
from pathlib import Path
import pytest
from sklearn.utils.fixes import parse_version
import imblearn
from imblearn._min_dependencies import dependent_packages