"""
Protocol that CausalDatasetPreparer and PreparedTableInput satisfy, letting
CausalExplorer accept either without an inheritance hierarchy.
"""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PreparedSource(Protocol):
    """
    Structural interface shared by CausalDatasetPreparer and PreparedTableInput.
    """

    @property
    def prepared_log(self) -> pd.DataFrame: ...

    @property
    def prepared_variables(self) -> pd.DataFrame: ...

    @property
    def parsed_variables(self) -> pd.DataFrame: ...

    @property
    def parsed_templates(self) -> pd.DataFrame: ...

    @property
    def workdir(self) -> str: ...
