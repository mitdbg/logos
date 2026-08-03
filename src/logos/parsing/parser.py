"""
Log-file parsing: Drain template extraction, type inference, and variable
tagging.
"""

import hashlib
import logging
import os
import warnings
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from tqdm.auto import tqdm

from logos.parsing.parsed_variable_name import ParsedVariableName
from logos.filesystem.cache import (
    artifact_exists,
    dump_dataframe,
    dump_metadata,
    load_dataframe,
    load_metadata,
)
from logos.parsing.drain import Drain
from logos.parsing.tag_utils import (
    deduplicate_tags,
    get_tag,
    preceding_tokens_tag,
    set_tag,
    waterfall_tag,
)

_logger = logging.getLogger(__name__)


class Parser:
    """Owns log-file parsing state and all parse-stage operations."""

    DEFAULT_REGEX_DICT = {
        "Timestamp": r"\d{4}\-\d{2}\-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
    }
    DEFAULT_MESSAGE_PREFIX = r".*"

    def __init__(
        self, filename: str, workdir: str, skip_writeout: bool = False
    ) -> None:
        self._filename = filename
        self._workdir = workdir
        self._skip_writeout = skip_writeout

        self._parsed_log: pd.DataFrame = pd.DataFrame()
        self._parsed_variables: pd.DataFrame = pd.DataFrame()
        self._parsed_templates: pd.DataFrame = pd.DataFrame()

        if not os.path.exists(self._workdir):
            os.makedirs(self._workdir, exist_ok=True)
        _logger.debug(f"Initialized LogParser with log file {filename}")

    # ------------------------------------------------------------------
    # Public properties
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
        return self._filename

    @property
    def workdir(self) -> str:
        return self._workdir

    @property
    def skip_writeout(self) -> bool:
        return self._skip_writeout

    # ------------------------------------------------------------------
    # Cache path helpers
    # ------------------------------------------------------------------

    def _get_parse_parquet_path(self, name: str) -> str:
        return os.path.join(
            self._workdir,
            f"{os.path.basename(self._filename)}_{name}.parquet",
        )

    def _get_parse_json_path(self, name: str) -> str:
        return os.path.join(
            self._workdir,
            f"{os.path.basename(self._filename)}_{name}.json",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_type(self, row: pd.Series) -> str:
        """
        Identify the type of a parsed variable.

        Parameters:
            row: A row of the parsed variables dataframe.

        Returns:
            The type of the parsed variable as a string. Options are "date",
                "time", "num" and "str".
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=UserWarning)

            try:
                y = pd.to_numeric(row["Examples"], errors="raise")
                return "num"
            except Exception as e:
                try:
                    y = pd.to_timedelta(row["Examples"], errors="raise")
                    return "time"
                except Exception as e:
                    try:
                        y = pd.to_datetime(row["Examples"], errors="raise")
                        return "date"
                    except Exception as e:
                        return "str"

    def _find_uninteresting(
        self, row: pd.Series, nunique_map: Optional[pd.Series] = None
    ) -> bool:
        """
        Identify whether a parsed variable is likely to be uninteresting.

        Parameters:
            row: A row of the parsed variables dataframe.
            nunique_map: Pre-computed unique-value counts for all parsed columns.
                If None, the count is computed on demand (single-variable path).

        Returns:
            True if the variable is likely to be uninteresting, False otherwise.
        """
        n = (
            self._parsed_log[row["Name"]].nunique()
            if nunique_map is None
            else nunique_map[row["Name"]]
        )
        return (row["Type"] != "num" and n >= 0.15 * row["Occurrences"]) or (
            n == 1
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def parse(
        self,
        regex_dict: dict[str, str] = DEFAULT_REGEX_DICT,
        sim_thresh: float = 0.65,
        depth: int = 5,
        force: bool = False,
        message_prefix: str = DEFAULT_MESSAGE_PREFIX,
        enable_gpt_tagging: bool = False,
    ) -> None:
        """
        Parse the log file into a dataframe.

        Parameters:
            regex_dict: (for Drain) A dictionary of regular expressions to be
                used for parsing.
            sim_thresh: (for Drain) The similarity threshold to be used for
                parsing.
            depth: (for Drain) The parse tree depth to be used for parsing.
            force: Whether to force re-parsing of the log file.
            message_prefix: A prefix used to identify the beginning of each log
                message. Can be used to collapse multiple lines into a single
                message. Each line that doesn't start with this prefix will be
                concatenated to the previous log message.
            enable_gpt_tagging: A boolean indicating whether GPT tagging should
                be enabled.
        """
        start_time = datetime.now()
        parser = Drain(
            indir=os.path.dirname(self._filename),
            depth=depth,
            st=sim_thresh,
            rex=regex_dict,
            skip_writeout=self._skip_writeout,
            message_prefix=message_prefix,
        )

        # Check if the parsed files already exist.
        files_exist = (
            not force
            and artifact_exists(self._get_parse_parquet_path("parsed_log"))
            and artifact_exists(self._get_parse_json_path("parsed_templates"))
            and artifact_exists(self._get_parse_json_path("parsed_variables"))
        )

        if files_exist:
            self._parsed_log = load_dataframe(
                self._get_parse_parquet_path("parsed_log")
            )
            self._parsed_templates = load_metadata(
                self._get_parse_json_path("parsed_templates")
            )
            self._parsed_variables = load_metadata(
                self._get_parse_json_path("parsed_variables")
            )
        else:
            (
                self._parsed_log,
                self._parsed_templates,
                self._parsed_variables,
            ) = parser.parse(self._filename.split("/")[-1])
            tqdm.pandas(desc="Determining variable types...")
            self._parsed_variables["Type"] = (
                self._parsed_variables.progress_apply(  # type: ignore[operator]
                    self._find_type, axis=1
                )
            )

            # Cast and convert date columns to Unix timestamps (float seconds)
            is_date = self._parsed_variables["Type"] == "date"
            date_cols = self._parsed_variables.loc[is_date, "Name"].tolist()
            for col in tqdm(date_cols, desc="Casting date variables..."):
                dt = pd.to_datetime(self._parsed_log[col], errors="coerce")
                # nanoseconds → seconds; NaT becomes NaN via .where
                self._parsed_log[col] = (
                    dt.astype("int64").where(dt.notna()) / 1e9
                )
            self._parsed_variables.loc[is_date, "Type"] = "num"

            # Cast and convert time columns to total seconds (float)
            is_time = self._parsed_variables["Type"] == "time"
            time_cols = self._parsed_variables.loc[is_time, "Name"].tolist()
            for col in tqdm(time_cols, desc="Casting time variables..."):
                self._parsed_log[col] = pd.to_timedelta(
                    self._parsed_log[col], errors="coerce"
                ).dt.total_seconds()
            self._parsed_variables.loc[is_time, "Type"] = "num"

            # Cast numeric columns
            is_num = self._parsed_variables["Type"] == "num"
            numeric_cols = self._parsed_variables.loc[is_num, "Name"].tolist()
            tqdm.pandas(desc="Casting numerical variables...")
            self._parsed_log[numeric_cols] = self._parsed_log[
                numeric_cols
            ].progress_apply(  # type: ignore[operator]
                pd.to_numeric, errors="coerce"
            )

            # Tag variables.
            tqdm.pandas(desc="Tagging variables...")
            if enable_gpt_tagging:
                tag, tag_origin = zip(
                    *self._parsed_variables.progress_apply(
                        lambda x: waterfall_tag(self.parsed_templates, x),
                        axis=1,  # type: ignore[operator]
                    )
                )
            else:
                tag, tag_origin = zip(
                    *self._parsed_variables.progress_apply(
                        lambda x: preceding_tokens_tag(x),
                        axis=1,  # type: ignore[operator]
                    )
                )
            self._parsed_variables["Tag"] = tag
            self._parsed_variables["TagOrigin"] = tag_origin
            deduplicate_tags(self._parsed_variables)

            # Detect identifiers.
            nunique_map = self._parsed_log.nunique()
            tqdm.pandas(desc="Detecting identifiers...")
            self._parsed_variables["IsUninteresting"] = (
                self._parsed_variables.progress_apply(
                    lambda row: self._find_uninteresting(row, nunique_map),
                    axis=1,  # type: ignore[operator]
                )
            )

            # Reorder columns.
            self._parsed_variables = self._parsed_variables[
                [
                    "Name",
                    "Tag",
                    "TagOrigin",
                    "Type",
                    "IsUninteresting",
                    "Occurrences",
                    "Preceding 3 tokens",
                    "Examples",
                    "From regex",
                ]
            ]

        # Write out files if appropriate.
        if not self._skip_writeout and not files_exist:
            dump_dataframe(
                self._parsed_log, self._get_parse_parquet_path("parsed_log")
            )
            dump_metadata(
                self._parsed_templates,
                self._get_parse_json_path("parsed_templates"),
            )
            dump_metadata(
                self._parsed_variables,
                self._get_parse_json_path("parsed_variables"),
            )

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        _logger.debug(f"Parsing complete in {elapsed:.6f} seconds!")

    def include_in_template(
        self,
        var: str,
        enable_gpt_tagging: bool = False,
        skip_writeout: Optional[bool] = None,
    ) -> None:
        """
        Treat a certain parsed variable as part of its template and regenerate
        parsed dataframes.

        Parameters:
            var: The name or tag of the variable to be included in its template.
            enable_gpt_tagging: A boolean indicating whether GPT-3.5 tagging
                should be enabled.
            skip_writeout: Whether to skip writing out the regenerated parsed
                dataframes. Defaults to the value of self._skip_writeout.
        """
        name = name_of(self._parsed_variables, var, "parsed")

        old_template_id = ParsedVariableName(name).template_id()
        idx = ParsedVariableName(name).index()
        value_counts = self._parsed_log[name].value_counts().to_dict()

        ### Modify _parsed_templates
        old_template_row = (
            self._parsed_templates.loc[
                self._parsed_templates["TemplateId"] == old_template_id
            ]
            .iloc[0]
            .copy()
        )
        toks = old_template_row["TemplateText"].split(" ")
        new_template_ids = {}
        new_variable_indices = old_template_row["VariableIndices"]
        new_variable_indices.remove(idx)

        for value, occurences in value_counts.items():
            new_template_row = old_template_row.copy()
            toks[idx] = value

            new_template_row["TemplateText"] = " ".join(toks)
            new_template_row["TemplateId"] = hashlib.md5(
                new_template_row["TemplateText"].encode("utf-8")
            ).hexdigest()[0:8]
            new_template_row["Occurrences"] = occurences
            new_template_row["VariableIndices"] = new_variable_indices
            new_template_row["RegexIndices"] = old_template_row["RegexIndices"]

            self._parsed_templates.loc[len(self._parsed_templates)] = (
                new_template_row
            )
            new_template_ids[value] = new_template_row["TemplateId"]

        self._parsed_templates = self._parsed_templates[
            self._parsed_templates["TemplateId"] != old_template_id
        ].reset_index(drop=True)

        ### Modify _parsed_log

        # Update the template ids of all rows that belonged to the old template
        mask = self._parsed_log["TemplateId"] == old_template_id
        self._parsed_log.loc[mask, "TemplateId"] = self._parsed_log.loc[
            mask, name
        ].map(new_template_ids)

        # Create new variables for each new template id and assign the value of
        # the old variables to them
        new_variables = []
        for new_template_id in new_template_ids.values():
            for other_idx in new_variable_indices:
                new_var_name = f"{new_template_id}_{str(other_idx)}"
                new_variables.append(new_var_name)
                old_col = f"{old_template_id}_{other_idx}"
                tmpl_mask = self._parsed_log["TemplateId"] == new_template_id
                self._parsed_log[new_var_name] = self._parsed_log[
                    old_col
                ].where(tmpl_mask)

        # Drop variable columns associated with old template id
        variables_to_drop = [
            v for v in self._parsed_log.columns if v.startswith(old_template_id)
        ]
        self._parsed_log.drop(columns=variables_to_drop, inplace=True)

        ### Modify _parsed_variables

        # Add variable rows for each new variable
        for value, occurrences in value_counts.items():
            for other_idx in new_variable_indices:
                new_template_id = new_template_ids[value]
                new_var_name = f"{new_template_id}_{str(other_idx)}"

                x: dict[str, Any] = {}
                x["Name"] = new_var_name
                x["Occurrences"] = occurrences
                x["Preceding 3 tokens"] = (
                    self._parsed_templates[
                        self._parsed_templates["TemplateId"] == new_template_id
                    ]["TemplateText"]
                    .values[0]
                    .split()[max(0, other_idx - 3) : other_idx]
                )
                x["Examples"] = (
                    self._parsed_log[new_var_name]
                    .loc[self._parsed_log[new_var_name].notna()]
                    .unique()[:5]
                    .tolist()
                )
                x["From regex"] = False
                if enable_gpt_tagging:
                    x["Tag"], x["TagOrigin"] = waterfall_tag(
                        self.parsed_templates, pd.Series(x)
                    )
                else:
                    x["Tag"], x["TagOrigin"] = preceding_tokens_tag(
                        pd.Series(x)
                    )
                x["Type"] = self._find_type(pd.Series(x))
                x["IsUninteresting"] = self._find_uninteresting(pd.Series(x))

                self._parsed_variables.loc[len(self._parsed_variables)] = x

        # Drop variable rows associated with old template id
        self._parsed_variables = self._parsed_variables[
            ~self._parsed_variables["Name"].isin(variables_to_drop)
        ].reset_index(drop=True)

        # Deduplicate tags again
        deduplicate_tags(self._parsed_variables)

        # Write out files if appropriate.
        if skip_writeout is None:
            skip_writeout = self._skip_writeout
        if not skip_writeout:
            dump_dataframe(
                self._parsed_log, self._get_parse_parquet_path("parsed_log")
            )
            dump_metadata(
                self._parsed_templates,
                self._get_parse_json_path("parsed_templates"),
            )
            dump_metadata(
                self._parsed_variables,
                self._get_parse_json_path("parsed_variables"),
            )

    def tag_parsed_variable(self, name: str, tag: str) -> None:
        """
        Tag a parsed variable.

        Parameters:
            name: The name of the variable to be tagged.
            tag: The tag to be assigned to the variable.
        """
        set_tag(self._parsed_variables, name, tag, "parsed")
        deduplicate_tags(self._parsed_variables)

    def get_tag_of_parsed(self, name: str) -> str:
        """
        Get the tag of a parsed variable.

        Parameters:
            name: The name of the variable.

        Returns:
            The tag of the variable.
        """
        return get_tag(self._parsed_variables, name, "parsed")
