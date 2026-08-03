"""
A collection of aggregation functions to be used during prepared log derivation.
"""

# pylint: disable=redefined-builtin

from typing import Optional

import pandas as pd


def mean(x: pd.Series) -> Optional[pd.Series]:
    """
    Calculates the mean of a series, ignoring NA values.

    Parameters:
        x: The series for which the mean will be calculated.

    Returns:
        The mean of the series, or None if the series is all NA.
    """
    return x.mean(skipna=True) if x.isna().sum() < len(x) else None


def min(x: pd.Series) -> Optional[pd.Series]:
    """
    Calculates the minimum of a series, ignoring NA values.

    Parameters:
        x: The series for which the minimum will be calculated.

    Returns:
        The minimum of the series, or None if the series is all NA.
    """
    return x.min(skipna=True) if x.isna().sum() < len(x) else None


def max(x: pd.Series) -> Optional[pd.Series]:
    """
    Calculates the maximum of a series, ignoring NA values.

    Parameters:
        x: The series for which the maximum will be calculated.

    Returns:
        The maximum of the series, or None if the series is all NA.
    """
    return x.max(skipna=True) if x.isna().sum() < len(x) else None


def median(x: pd.Series) -> Optional[pd.Series]:
    """
    Calculates the median of a series, ignoring NA values.

    Parameters:
        x: The series for which the median will be calculated.

    Returns:
        The median of the series, or None if the series is all NA.
    """
    return x.median(skipna=True) if x.isna().sum() < len(x) else None


def mode(x: pd.Series) -> Optional[pd.Series]:
    """
    Calculates the mode of a series, ignoring NA values.

    Parameters:
        x: The series for which the mode will be calculated.

    Returns:
        The mode of the series, or None if the series is all NA.
    """
    return x.mode(dropna=True)[0] if x.isna().sum() < len(x) else None


def std(x: pd.Series) -> Optional[pd.Series]:
    """
    Calculates the standard deviation of a series, ignoring NA values.

    Parameters:
        x: The series for which the standard deviation will be calculated.

    Returns:
        The standard deviation of the series, or None if the series is all NA.
    """
    return x.std(skipna=True) if x.isna().sum() < len(x) else None


def last(x: pd.Series) -> Optional[pd.Series]:
    """
    Returns the last non-NA value in a series.

    Parameters:
        x: The series for which the last non-NA value will be returned.

    Returns:
        The last non-NA value of the series, or None if the series is all NA.
    """
    return x.dropna().tail(1) if x.isna().sum() < len(x) else None


def first(x: pd.Series) -> Optional[pd.Series]:
    """
    Returns the first non-NA value in a series.

    Parameters:
        x: The series for which the first non-NA value will be returned.

    Returns:
        The first non-NA value of the series, or None if the series is all NA.
    """
    return x.dropna().head(1) if x.isna().sum() < len(x) else None


def sum(x: pd.Series) -> Optional[pd.Series]:
    """
    Calculates the sum of a series, ignoring NA values.

    Parameters:
        x: The series for which the sum will be calculated.

    Returns:
        The sum of the series, or None if the series is all NA.
    """
    return x.sum(skipna=True) if x.isna().sum() < len(x) else None
