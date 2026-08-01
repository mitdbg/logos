"""
Entry Point 2: wrap a user-provided DataFrame as a ParsedSource so that
CausalDatasetPreparer can prepare it without running Drain.
"""

import os
from typing import Optional

import pandas as pd

from src.logos.tag_utils import TagOrigin, TagUtils


class ParsedTableInput:
    """
    A ParsedSource backed by a user-supplied DataFrame.

    The DataFrame is treated as if it were the output of LogParser.parse():
    one row per log event, one column per field.  No Drain run is performed.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        workdir: str,
        source_id: str = "parsed_input",
        variable_tags: Optional[dict[str, str]] = None,
        skip_writeout: bool = False,
    ) -> None:
        """
        Parameters:
            data: The user-provided table (one row per event).
            workdir: Directory used for prepare-stage cache files.
            source_id: Identifier used as the cache-path prefix (analogous
                to the log filename in LogParser).
            variable_tags: Optional mapping from column name to human-readable
                tag.  Columns absent from this mapping are tagged with their
                own name.
            skip_writeout: Whether to skip writing prepare-stage cache files.
        """
        self._source_id = source_id
        self._workdir = workdir
        self._skip_writeout = skip_writeout

        if not os.path.exists(self._workdir):
            os.makedirs(self._workdir, exist_ok=True)

        self._parsed_log: pd.DataFrame = data.copy(deep=True)
        self._parsed_variables: pd.DataFrame = self._synthesize_variables(
            data, variable_tags or {}
        )
        self._parsed_templates: pd.DataFrame = pd.DataFrame(
            columns=[
                "TemplateId",
                "TemplateText",
                "Occurrences",
                "VariableIndices",
                "RegexIndices",
            ]
        )

    # ------------------------------------------------------------------
    # ParsedSource interface
    # ------------------------------------------------------------------

    @property
    def parsed_log(self) -> pd.DataFrame:
        return self._parsed_log

    @property
    def parsed_variables(self) -> pd.DataFrame:
        return self._parsed_variables

    @property
    def parsed_templates(self) -> pd.DataFrame:
        return self._parsed_templates

    @property
    def filename(self) -> str:
        return self._source_id

    @property
    def workdir(self) -> str:
        return self._workdir

    @property
    def skip_writeout(self) -> bool:
        return self._skip_writeout

    def get_tag_of_parsed(self, name: str) -> str:
        return TagUtils.get_tag(self._parsed_variables, name, "parsed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_type(series: pd.Series) -> str:
        return "num" if pd.api.types.is_numeric_dtype(series) else "str"

    @staticmethod
    def _synthesize_variables(
        data: pd.DataFrame, variable_tags: dict[str, str]
    ) -> pd.DataFrame:
        rows = []
        for col in data.columns:
            col_type = ParsedTableInput._infer_type(data[col])
            examples = (
                data[col].dropna().unique()[:5].tolist()
            )
            rows.append(
                {
                    "Name": col,
                    "Tag": variable_tags.get(col, col),
                    "TagOrigin": int(TagOrigin.REGEX_VARIABLE),
                    "Type": col_type,
                    "IsUninteresting": False,
                    "Occurrences": int(data[col].notna().sum()),
                    "Preceding 3 tokens": [],
                    "Examples": examples,
                    "From regex": True,
                }
            )
        return pd.DataFrame(rows)
