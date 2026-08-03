from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PreparerLike(Protocol):

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
