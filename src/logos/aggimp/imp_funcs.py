import pandas as pd

"""
A collection of imputation functions to be used during prepared log derivation.
"""


def ffill_imp(x: pd.Series) -> pd.Series:
    """
    Impute the NA values in a series by forward-filling and return the series.

    Parameters:
        x: The series for which the NA values will be imputed.

    Returns:
        The series, with NA values imputed.
    """
    return x.ffill()


def zero_imp(x: pd.Series) -> pd.Series:
    """
    Impute the NA values in a series with zeroes and return the series.

    Parameters:
        x: The series for which the NA values will be imputed.

    Returns:
        The series, with NA values imputed.
    """
    return x.fillna(0)


def no_imp(x: pd.Series) -> pd.Series:
    """
    No-op.

    Parameters:
        x: The series to be returned.

    Returns:
        The series passed as a parameter.
    """
    return x
