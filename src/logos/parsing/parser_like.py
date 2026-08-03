from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ParserLike(Protocol):

    @property
    def parsed_log(self) -> pd.DataFrame: ...

    @property
    def parsed_variables(self) -> pd.DataFrame: ...

    @property
    def parsed_templates(self) -> pd.DataFrame: ...

    @property
    def workdir(self) -> str: ...

    @property
    def filename(self) -> str: ...

    @property
    def skip_writeout(self) -> bool: ...

    def get_tag_of_parsed(self, name: str) -> str: ...
