"""
Data preparation: aggregation, imputation, one-hot encoding, and causal-unit
management.
"""

import importlib
import inspect
import logging
import os
from typing import Callable, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

from logos.aggregate_selector import AggregateSelector
from logos.cache import Cache
from logos.causal_unit_suggester import CausalUnitSuggester
from logos.parsed_source import ParsedSource
from logos.tag_utils import TagUtils
from logos.variable_name.prepared_variable_name import PreparedVariableName

_logger = logging.getLogger(__name__)


class CausalDatasetPreparer:
    """Owns data-preparation state and all prepare-stage operations."""

    def __init__(self, parser: ParsedSource) -> None:
        self._parser = parser

        self._causal_unit_var: Optional[str] = None
        self._num_causal_units: Optional[int] = None

        self._prepared_log: pd.DataFrame = pd.DataFrame()
        self._prepared_variables: pd.DataFrame = pd.DataFrame()

        agg_module = importlib.import_module("logos.aggimp.agg_funcs")
        self._agg_funcs: dict[str, Callable] = {
            n: f for n, f in inspect.getmembers(agg_module, inspect.isfunction)
        }
        imp_module = importlib.import_module("logos.aggimp.imp_funcs")
        self._imp_funcs: dict[str, Callable] = {
            n: f for n, f in inspect.getmembers(imp_module, inspect.isfunction)
        }

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def prepared_log(self) -> pd.DataFrame:
        return self._prepared_log

    @property
    def prepared_variables(self) -> pd.DataFrame:
        return self._prepared_variables

    @property
    def prepared_variable_names(self) -> list[str]:
        return self._prepared_variables["Name"].values.tolist()

    @property
    def prepared_variable_tags(self) -> list[str]:
        return self._prepared_variables["Tag"].values.tolist()

    @property
    def num_prepared_variables(self) -> int:
        return len(self._prepared_variables)

    @property
    def parsed_variables(self) -> pd.DataFrame:
        """
        Forward parsed_variables from the underlying parser 
        (satisfies PreparedSource).
        """
        return self._parser.parsed_variables

    @property
    def parsed_templates(self) -> pd.DataFrame:
        """
        Forward parsed_templates from the underlying parser 
        (satisfies PreparedSource).
        """
        return self._parser.parsed_templates

    @property
    def workdir(self) -> str:
        """
        Forward workdir from the underlying parser 
        (satisfies PreparedSource).
        """
        return self._parser.workdir

    # ------------------------------------------------------------------
    # Cache path helpers
    # ------------------------------------------------------------------

    def _get_prepare_parquet_path(self, name: str) -> str:
        return os.path.join(
            self._parser.workdir,
            f"{os.path.basename(self._parser.filename)}"
            f"_{name}_{self._causal_unit_var}_{self._num_causal_units}.parquet",
        )

    def _get_prepare_json_path(self, name: str) -> str:
        return os.path.join(
            self._parser.workdir,
            f"{os.path.basename(self._parser.filename)}"
            f"_{name}_{self._causal_unit_var}_{self._num_causal_units}.json",
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_causal_unit_info(self) -> Tuple[Optional[str], Optional[int]]:
        """
        Get the variable used to define causal units and the number of
        causal units.

        Returns:
            causal_unit_var: The variable used to define causal units, if
                specified.
            num_causal_units: The number of causal units, if specified.
        """
        return self._causal_unit_var, self._num_causal_units

    def suggest_causal_unit_defs(
        self,
        min_causal_units: int = 4,
        num_suggestions: int = 10,
    ) -> Optional[pd.DataFrame]:
        """
        Suggest at most `num_suggestions` causal unit definitions based on IUS
        maximization, while returning at least `min_causal_units` causal units.

        Parameters:
            min_causal_units: The minimum number of causal units that a
                suggested definition should create.
            num_suggestions: The maximum number of causal unit definitions to
                suggest.

        Returns:
            A DataFrame with one row for each suggested causal unit definition,
                or `None` if no suggestions were made.
        """
        return CausalUnitSuggester.suggest_causal_unit_defs(
            self._parser.parsed_log[
                self._parser.parsed_variables["Name"].values
            ],
            self._parser.parsed_variables,
            min_causal_units=min_causal_units,
            num_suggestions=num_suggestions,
        )

    def set_causal_unit(
        self,
        var: Optional[str] = None,
        num_units: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Set the variable used to define causal units.

        When called with no `var`, runs the IUS maximizer and returns a ranked
        DataFrame of suggestions. Call again with a chosen `var` to set the unit.

        Parameters:
            var: The name or tag of the variable to use as the causal unit.
                If None, suggestions are returned without setting anything.
            num_units: The number of causal units to create (required for
                numerical variables).

        Returns:
            A suggestion DataFrame when `var` is None; None otherwise.

        Raises:
            ValueError: If the variable is numerical and `num_units` is not
                specified.
        """
        if var is None:
            return self.suggest_causal_unit_defs()

        var_name = TagUtils.name_of(
            self._parser.parsed_variables, var, "parsed"
        )
        var_type = self._parser.parsed_variables.loc[
            self._parser.parsed_variables["Name"] == var_name, "Type"
        ].values[0]

        if var_type == "num" and num_units is None:
            raise ValueError(
                "The number of causal units must be specified if the causal "
                "unit is numerical."
            )

        self._causal_unit_var = var_name
        self._num_causal_units = num_units

        tag = self._parser.get_tag_of_parsed(var_name)
        _logger.debug(
            f"Causal unit set to {var_name} (tag: {tag})"
            + (
                ""
                if not self._num_causal_units
                else f" with {self._num_causal_units} causal units."
            )
        )
        return None

    def prepare(
        self,
        custom_agg: Optional[dict[str, list[str]]] = None,
        custom_imp: Optional[dict[str, str]] = None,
        count_occurrences: bool = False,
        ignore_uninteresting: bool = True,
        force: bool = False,
        drop_bad_aggs: bool = True,
    ) -> bool:
        """
        Prepare the parsed log for causal analysis.

        Returns:
            True if preparation succeeded; False if aborted because no causal
                unit is set.
        """
        if custom_agg is None:
            custom_agg = {}
        if custom_imp is None:
            custom_imp = {}

        # Ensure causal unit is set. TODO: make IUS maximizer the default
        if self._causal_unit_var is None:
            print("Causal unit not defined. Aborting.")
            return False

        # Check if the prepared files already exist.
        files_exist = (
            not force
            and Cache.artifact_exists(
                self._get_prepare_parquet_path("prepared_log")
            )
            and Cache.artifact_exists(
                self._get_prepare_json_path("prepared_variables")
            )
        )

        if files_exist:
            self._prepared_log = Cache.load_dataframe(
                self._get_prepare_parquet_path("prepared_log")
            )
            self._prepared_variables = Cache.load_metadata(
                self._get_prepare_json_path("prepared_variables")
            )
        else:
            self._prepare_anew(
                custom_agg,
                custom_imp,
                count_occurrences=count_occurrences,
                ignore_uninteresting=ignore_uninteresting,
                drop_bad_aggs=drop_bad_aggs,
            )

        return True

    def _prepare_anew(
        self,
        custom_agg: Optional[dict[str, list[str]]] = None,
        custom_imp: Optional[dict[str, str]] = None,
        count_occurrences: bool = False,
        ignore_uninteresting: bool = True,
        drop_bad_aggs: bool = True,
    ) -> None:
        """
        Prepare the log anew.

        Parameters:
            custom_agg: A dictionary of custom aggregation functions to be used
                for specific variables.
            custom_imp: A dictionary of the custom imputation function to be
                used for specific variables.
            count_occurrences: Whether to include extra variables counting the
                occurrence of each template.
            ignore_uninteresting: Whether to ignore uninteresting variables.
            drop_bad_aggs: Whether to drop prepared variables that do not add
                information compared to other variables based on the same base
                variable but using a different aggregation function.
        """

        if custom_agg is None:
            custom_agg = {}
        if custom_imp is None:
            custom_imp = {}

        _logger.debug(f"Determining the causal unit assignment...")
        causal_unit_assignment = CausalUnitSuggester.discretize(
            self._parser.parsed_log[self._causal_unit_var],
            self._parser.parsed_variables[
                self._parser.parsed_variables["Name"] == self._causal_unit_var
            ]["Type"].values[0],
            self._num_causal_units if self._num_causal_units else 0,
        )

        # Convert keys in custom_agg and custom_imp to the names of the
        # variables, if they are tags.
        custom_agg = {
            TagUtils.name_of(self._parser.parsed_variables, k, "parsed"): v
            for k, v in custom_agg.items()
        }
        custom_imp = {
            TagUtils.name_of(self._parser.parsed_variables, k, "parsed"): v
            for k, v in custom_imp.items()
        }

        # Start with the parsed log, optionally with extra variables counting
        # the occurence of each template.
        if count_occurrences and "TemplateId" in self._parser.parsed_log.columns:
            _logger.debug("Adding template occurrence count variables...")
            self._prepared_log = pd.concat(
                [
                    self._parser.parsed_log,
                    pd.get_dummies(
                        self._parser.parsed_log["TemplateId"],
                        prefix="TemplateId",
                        prefix_sep="=",
                        dtype=float,
                    ),
                ],
                axis=1,
            )
        else:
            if count_occurrences:
                _logger.debug(
                    "count_occurrences=True ignored: no TemplateId column."
                )
            self._prepared_log = self._parser.parsed_log.copy(deep=True)

        # Drop the TemplateId column if present.
        if "TemplateId" in self._prepared_log.columns:
            self._prepared_log.drop(columns="TemplateId", inplace=True)

        # Build dictionary of aggregation functions
        agg_dict: dict[str, list[str]] = {
            variable.Name: (
                custom_agg[variable.Name]
                if variable.Name in custom_agg
                else AggregateSelector.DEFAULT_AGGREGATES[variable.Type]
            )
            for variable in self._parser.parsed_variables.itertuples()
        }

        # Add aggregations for template counts
        for col in self._prepared_log.columns:
            if PreparedVariableName(col).base_var() == "TemplateId":
                agg_dict[col] = ["sum"]

        # Drop uninteresting columns if requested, except if they are the causal
        # unit.
        ui_cols = self._parser.parsed_variables.loc[
            self._parser.parsed_variables["IsUninteresting"], "Name"
        ].values
        ui_cols = [x for x in ui_cols if x != self._causal_unit_var]
        if ignore_uninteresting:
            self._prepared_log.drop(
                columns=ui_cols,
                inplace=True,
            )
            for col in ui_cols:
                agg_dict.pop(col, None)
            _logger.debug(
                f"Dropped {len(ui_cols)} uninteresting columns, out of an "
                f"original total of {len(self._parser.parsed_variables)}."
            )

        # Ensure the causal unit variable only has one aggregation function
        assert self._causal_unit_var is not None
        agg_dict[self._causal_unit_var] = agg_dict[self._causal_unit_var][:1]

        # Perform the aggregation
        _logger.debug("Calculating aggregates for each causal unit...")
        agg_func_dict: dict[str, list[Callable]] = {
            name: [self._agg_funcs[f] for f in funcs]
            for name, funcs in agg_dict.items()
        }
        self._prepared_log = self._prepared_log.groupby(
            causal_unit_assignment
        ).aggregate(agg_func_dict)
        self._prepared_log.columns = [
            "+".join(col) for col in self._prepared_log.columns.values
        ]
        self._parser.parsed_variables["Aggregates"] = (
            self._parser.parsed_variables["Name"].map(
                lambda x: agg_dict.get(x, [])
            )
        )
        cond = self._parser.parsed_variables["Name"] == self._causal_unit_var
        self._prepared_log.set_index(
            f"{self._causal_unit_var}+"
            f"{self._parser.parsed_variables[cond]['Aggregates'].values[0][0]}",
            inplace=True,
        )
        self._prepared_log.sort_index(inplace=True)
        self._prepared_log.index = self._prepared_log.index.astype(str)

        # Perform the imputation
        for col in tqdm(
            self._prepared_log.columns, desc="Imputing missing values..."
        ):
            if self._prepared_log[col].isnull().values.any():
                base_var = PreparedVariableName(col).base_var()
                func_name: str = (
                    custom_imp[base_var] if base_var in custom_imp else "no_imp"
                )
                self._prepared_log[col] = (self._imp_funcs[func_name])(
                    self._prepared_log[col]
                )
        self._prepared_log.dropna(inplace=True)

        # Drop variables that do not add information compared to other variables 
        # based on the same base variable but using a different aggregation 
        # function.
        if drop_bad_aggs:
            _logger.debug("Dropping aggregates that do not add information...")
            cols_to_drop = AggregateSelector.find_uninformative_aggregates(
                self._prepared_log,
                self._parser.parsed_variables,
                self._causal_unit_var,
            )
            self._prepared_log.drop(columns=cols_to_drop, inplace=True)

        # Identify the categorical variables and one-hot encode them
        categorical_vars = self._prepared_log.select_dtypes(
            include="object"
        ).columns.tolist()
        for col in tqdm(
            categorical_vars, desc="One-hot encoding categorical variables..."
        ):
            self._prepared_log = pd.concat(
                [
                    self._prepared_log,
                    pd.get_dummies(
                        self._prepared_log[col],
                        prefix=col,
                        prefix_sep="=",
                        dtype=float,
                    ),
                ],
                axis=1,
            )
            self._prepared_log.drop(col, axis=1, inplace=True)
        # Deal with https://github.com/pydot/pydot/issues/258
        self._prepared_log.columns = [
            x.replace(":", ";") for x in self._prepared_log.columns
        ]

        # Generate dataframe of prepared variables for later tagging etc.
        self._generate_prepared_variables_df()

        # Convert any date columns to Unix timestamps in milliseconds
        date_cols = self._prepared_variables.loc[
            self._prepared_variables["Type"] == "date", "Name"
        ].tolist()
        self._prepared_log[date_cols] = self._prepared_log[date_cols].map(
            lambda x: x.timestamp() * 1000.0
        )

        # Convert any time columns to milliseconds
        time_cols = self._prepared_variables.loc[
            self._prepared_variables["Type"] == "time", "Name"
        ].tolist()
        self._prepared_log[time_cols] = self._prepared_log[time_cols].map(
            lambda x: x.total_seconds() * 1000.0
        )

        # Write out prepared log and variables
        if not self._parser.skip_writeout:
            Cache.dump_dataframe(
                self._prepared_log,
                self._get_prepare_parquet_path("prepared_log"),
            )
            Cache.dump_metadata(
                self._prepared_variables,
                self._get_prepare_json_path("prepared_variables"),
            )

        cuv = self._causal_unit_var
        _logger.debug(
            f"Successfully prepared the log with causal unit {cuv} "
            f"(tag: {self._parser.get_tag_of_parsed(cuv)})"
            + (
                ""
                if not self._num_causal_units
                else f" with {self._num_causal_units} causal units."
            )
        )

    def _generate_prepared_variables_df(self) -> None:
        """Generate dataframe of prepared variables for later tagging etc."""
        self._prepared_variables = pd.DataFrame()
        self._prepared_variables["Name"] = self._prepared_log.columns

        # Bring in variable name components leveraging PreparedVariableName
        self._prepared_variables["Base"] = self._prepared_variables[
            "Name"
        ].apply(lambda x: PreparedVariableName(x).base_var())
        self._prepared_variables["Pre-agg Value"] = self._prepared_variables[
            "Name"
        ].apply(lambda x: PreparedVariableName(x).pre_agg_value())
        self._prepared_variables["Agg"] = self._prepared_variables[
            "Name"
        ].apply(lambda x: PreparedVariableName(x).aggregate())
        self._prepared_variables["Post-agg Value"] = self._prepared_variables[
            "Name"
        ].apply(lambda x: PreparedVariableName(x).post_agg_value())

        # Bring in other info from parsed_variables
        self._prepared_variables["Tag"] = self._prepared_variables.apply(
            lambda x: (
                self._parser.parsed_variables.loc[
                    self._parser.parsed_variables["Name"] == x["Base"],
                    "Tag",
                ].values[0]
                if x["Base"] != "TemplateId"
                else "TemplateId"
            )
            + (f" {x['Pre-agg Value']}" if x["Pre-agg Value"] != "" else "")
            + (f" {x['Agg']}" if x["Agg"] != "" else "")
            + (f" {x['Post-agg Value']}" if x["Post-agg Value"] != "" else ""),
            axis=1,
        )
        self._prepared_variables[
            "Base Variable Occurences"
        ] = self._prepared_variables["Base"].apply(
            lambda x: (
                self._parser.parsed_variables.loc[
                    self._parser.parsed_variables["Name"] == x, "Occurrences"
                ].values[0]
                if x != "TemplateId"
                else ""
            )
        )
        self._prepared_variables["Type"] = self._prepared_variables[
            "Base"
        ].apply(
            lambda x: (
                self._parser.parsed_variables.loc[
                    self._parser.parsed_variables["Name"] == x, "Type"
                ].values[0]
                if x != "TemplateId"
                else ""
            )
        )
        self._prepared_variables["Examples"] = self._prepared_variables[
            "Base"
        ].apply(
            lambda x: (
                self._parser.parsed_variables.loc[
                    self._parser.parsed_variables["Name"] == x, "Examples"
                ].values[0]
                if x != "TemplateId"
                else ""
            )
        )
        self._prepared_variables["From regex"] = self._prepared_variables[
            "Base"
        ].apply(
            lambda x: (
                self._parser.parsed_variables.loc[
                    self._parser.parsed_variables["Name"] == x, "From regex"
                ].values[0]
                if x != "TemplateId"
                else ""
            )
        )

        def _get_template_text(row: pd.Series) -> str:
            if row["From regex"]:
                return ""
            tpl_id = PreparedVariableName(row["Name"]).template_id()
            matches = self._parser.parsed_templates.loc[
                self._parser.parsed_templates["TemplateId"] == tpl_id,
                "TemplateText",
            ]
            return matches.values[0] if len(matches) > 0 else ""

        # Bring in template text, only for appropriate base variables.
        self._prepared_variables["TemplateText"] = (
            self._prepared_variables.apply(_get_template_text, axis=1)
        )

    def tag_prepared_variable(self, name: str, tag: str) -> None:
        """
        Tag a prepared variable.

        Parameters:
            name: The name of the variable to be tagged.
            tag: The tag to be assigned to the variable.
        """
        TagUtils.set_tag(self._prepared_variables, name, tag, "prepared")
        TagUtils.deduplicate_tags(self._prepared_variables)

    def get_tag_of_prepared(self, name: str) -> str:
        """
        Get the tag of a prepared variable.

        Parameters:
            name: The name of the variable.

        Returns:
            The tag of the variable.
        """
        return TagUtils.get_tag(self._prepared_variables, name, "prepared")
