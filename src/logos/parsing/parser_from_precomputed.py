"""
Entry Point 2: wrap a user-provided DataFrame as a ParserLike so that
CausalDatasetPreparer can prepare it without running Drain.
"""

import hashlib
import os
from typing import Optional

import pandas as pd

from logos.parsing.tag_utils import TagOrigin, get_tag


def _to_parsed_name(col: str, existing: set[str]) -> str:
    """Return a collision-free `{hex8}_{int}` key for a user column name."""
    hex_id = hashlib.md5(col.encode()).hexdigest()[:8]
    candidate = f"{hex_id}_0"
    suffix = 0
    while candidate in existing:
        suffix += 1
        candidate = f"{hex_id}_{suffix}"
    return candidate


def _to_template_id(val: str, existing: set[str]) -> str:
    """Return a collision-free 8-char hex ID for a template value."""
    base = hashlib.md5(val.encode()).hexdigest()
    for length in range(8, len(base) + 1):
        h = base[:length]
        if h not in existing:
            return h
    raise ValueError(f"Hash collision exhausted for {val!r}")


class ParserFromPrecomputed:
    """
    A ParserLike backed by a user-supplied DataFrame.

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
        template_col: Optional[str] = None,
        passthrough_cols: Optional[list[str]] = None,
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
            template_col: Column whose distinct values define templates.
                When provided, each value is hashed to a template ID; other
                columns are split into per-template variables (named
                `{template_hex8}_{col_idx}`) that are NaN outside their
                template's rows.  Mirrors Drain's output structure.
            passthrough_cols: Columns that should appear as single global
                variables rather than being split per template.  Only used
                when `template_col` is set.  Typical use: the causal-unit
                identifier (e.g. `query_id`) that must be the same for every
                row regardless of event type.
        """
        self._source_id = source_id
        self._workdir = workdir
        self._skip_writeout = skip_writeout

        if not os.path.exists(self._workdir):
            os.makedirs(self._workdir, exist_ok=True)

        if template_col is not None:
            (
                self._parsed_log,
                self._parsed_templates,
                self._parsed_variables,
            ) = ParserFromPrecomputed._build_from_template_col(
                data, template_col, variable_tags or {}, passthrough_cols or []
            )
        else:
            # Map every user column to a valid ParsedVariableName key so that
            # downstream code that calls .template_id() / .index() never crashes.
            col_rename: dict[str, str] = {}
            used: set[str] = set()
            for col in data.columns:
                internal = _to_parsed_name(col, used)
                col_rename[col] = internal
                used.add(internal)

            # Effective tags: internal key → display label (original col name as default)
            user_tags = variable_tags or {}
            effective_tags = {
                col_rename[col]: user_tags.get(col, col) for col in data.columns
            }

            self._parsed_log = data.rename(columns=col_rename).copy(deep=True)
            self._parsed_variables = self._synthesize_variables(
                self._parsed_log, effective_tags
            )
            self._parsed_templates = pd.DataFrame(
                columns=[
                    "TemplateId",
                    "TemplateText",
                    "Occurrences",
                    "VariableIndices",
                    "RegexIndices",
                ]
            )

    # ------------------------------------------------------------------
    # ParserLike interface
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
        return get_tag(self._parsed_variables, name, "parsed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_type(series: pd.Series) -> str:
        return "num" if pd.api.types.is_numeric_dtype(series) else "str"

    @staticmethod
    def _build_from_template_col(
        data: pd.DataFrame,
        template_col: str,
        variable_tags: dict[str, str],
        passthrough_cols: list[str],
    ) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]":
        """
        Split columns into per-template variables keyed by `{hex8}_{col_idx}`.

        Each unique value in `template_col` becomes a template whose ID is the
        first 8 hex chars of its MD5 hash.  A variable column for (template T,
        source column C) contains the values of C for rows where
        `template_col == T` and NaN everywhere else.

        Columns listed in `passthrough_cols` are kept as single global columns
        (not split per template) so they can serve as the causal-unit identifier
        across all event types.
        """
        pt_set = set(passthrough_cols)
        other_cols = [
            c for c in data.columns
            if c != template_col and c not in pt_set
        ]
        str_col = data[template_col].map(str)  # normalise dtype for comparisons

        used_ids: set[str] = set()
        template_id_map: dict[str, str] = {}  # str(template_value) -> hex8
        for t_val in str_col.unique():
            t_id = _to_template_id(t_val, used_ids)
            used_ids.add(t_id)
            template_id_map[t_val] = t_id

        new_log_cols: dict[str, pd.Series] = {
            "TemplateId": str_col.map(template_id_map)
        }
        template_rows = []
        var_rows = []

        for t_val, t_id in template_id_map.items():
            mask = str_col == t_val
            occurrences = int(mask.sum())

            active_cols = [
                col for col in other_cols
                if data.loc[mask, col].notna().any()
            ]

            for var_idx, col in enumerate(active_cols):
                var_name = f"{t_id}_{var_idx}"
                new_log_cols[var_name] = data[col].where(mask)
                var_rows.append({
                    "Name": var_name,
                    "Tag": variable_tags.get(col, col),
                    "TagOrigin": int(TagOrigin.REGEX_VARIABLE),
                    "Type": ParserFromPrecomputed._infer_type(
                        data.loc[mask, col]
                    ),
                    "IsUninteresting": False,
                    "Occurrences": occurrences,
                    "Preceding 3 tokens": [],
                    "Examples": (
                        data.loc[mask, col].dropna().unique()[:5].tolist()
                    ),
                    "From regex": False,
                })

            template_rows.append({
                "TemplateId": t_id,
                "TemplateText": t_val,
                "Occurrences": occurrences,
                "VariableIndices": list(range(len(active_cols))),
                "RegexIndices": [],
            })

        parsed_log = pd.DataFrame(new_log_cols, index=data.index)

        # Add passthrough columns as flat {hex8}_0 global variables.
        used_var_names: set[str] = set(new_log_cols.keys())
        pt_var_rows = []
        for col in passthrough_cols:
            pt_name = _to_parsed_name(col, used_var_names)
            used_var_names.add(pt_name)
            parsed_log[pt_name] = data[col].values
            pt_var_rows.append({
                "Name": pt_name,
                "Tag": variable_tags.get(col, col),
                "TagOrigin": int(TagOrigin.REGEX_VARIABLE),
                "Type": ParserFromPrecomputed._infer_type(data[col]),
                "IsUninteresting": False,
                "Occurrences": int(data[col].notna().sum()),
                "Preceding 3 tokens": [],
                "Examples": data[col].dropna().unique()[:5].tolist(),
                "From regex": True,
            })

        all_var_rows = var_rows + pt_var_rows
        parsed_templates = pd.DataFrame(
            template_rows,
            columns=[
                "TemplateId", "TemplateText", "Occurrences",
                "VariableIndices", "RegexIndices",
            ],
        )
        parsed_variables = pd.DataFrame(
            all_var_rows,
            columns=[
                "Name", "Tag", "TagOrigin", "Type", "IsUninteresting",
                "Occurrences", "Preceding 3 tokens", "Examples", "From regex",
            ],
        ) if all_var_rows else pd.DataFrame(
            columns=[
                "Name", "Tag", "TagOrigin", "Type", "IsUninteresting",
                "Occurrences", "Preceding 3 tokens", "Examples", "From regex",
            ]
        )
        return parsed_log, parsed_templates, parsed_variables

    @staticmethod
    def _synthesize_variables(
        data: pd.DataFrame, variable_tags: dict[str, str]
    ) -> pd.DataFrame:
        rows = []
        for col in data.columns:
            col_type = ParserFromPrecomputed._infer_type(data[col])
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
