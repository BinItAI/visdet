"""
Semantic code comparison tool for Python.

Compare Python code implementations ignoring whitespace and formatting differences.
Identify code that has been moved, renamed, or modified.
"""

__version__ = "0.1.0"
__author__ = "George Pearse"

from codediff.comparator import CodeComparator
from codediff.report import ComparisonReport

__all__ = ["CodeComparator", "ComparisonReport"]
