"""
Entry Point 3: wrap a user-provided DataFrame as a prepared dataset,
bypassing both the parse and prepare stages entirely.
"""

import os
from typing import Optional

import pandas as pd


class PreparedDataFrameSource:
    """
    Holds a user-provided prepared DataFrame ready for CausalExplorer.

    The DataFrame is treated as if it were the output of
    CausalDatasetPreparer.prepare(): one row per causal unit, one column
    per feature.  No parsing or aggregation is performed.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        workdir: str,
        variable_tags: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Parameters:
            data: The user-provided table (one row per causal unit).
            workdir: Directory used for GPT log files during exploration.
            variable_tags: Optional mapping from column name to human-readable
                tag.  Columns absent from this mapping are tagged with their
                own name.
        """
        if not os.path.exists(workdir):
            os.makedirs(workdir, exist_ok=True)

        self._workdir = workdir
        self._prepared_log: pd.DataFrame = data.copy(deep=True)
        self._prepared_variables: pd.DataFrame = self._synthesize_variables(
            data, variable_tags or {}
        )
        # Empty parsed artefacts; inspect() skips lookups when From regex=True.
        self._parsed_variables: pd.DataFrame = pd.DataFrame(
            columns=["Name", "Tag", "Type", "IsUninteresting", "From regex"]
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

    @property
    def prepared_log(self) -> pd.DataFrame:
        return self._prepared_log

    @property
    def prepared_variables(self) -> pd.DataFrame:
        return self._prepared_variables

    @property
    def parsed_variables(self) -> pd.DataFrame:
        return self._parsed_variables

    @property
    def parsed_templates(self) -> pd.DataFrame:
        return self._parsed_templates

    @property
    def workdir(self) -> str:
        return self._workdir

    @staticmethod
    def _infer_type(series: pd.Series) -> str:
        return "num" if pd.api.types.is_numeric_dtype(series) else "str"

    @staticmethod
    def _synthesize_variables(
        data: pd.DataFrame, variable_tags: dict[str, str]
    ) -> pd.DataFrame:
        rows = []
        for col in data.columns:
            col_type = PreparedDataFrameSource._infer_type(data[col])
            examples = data[col].dropna().unique()[:5].tolist()
            rows.append(
                {
                    "Name": col,
                    "Tag": variable_tags.get(col, col),
                    "Base": col,
                    "Pre-agg Value": "",
                    "Agg": "",
                    "Post-agg Value": "",
                    "Type": col_type,
                    "Examples": examples,
                    "From regex": True,
                    "TemplateText": "",
                    "Base Variable Occurences": "",
                }
            )
        return pd.DataFrame(rows)
