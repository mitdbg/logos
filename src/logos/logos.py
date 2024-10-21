import hashlib
import importlib
import inspect
import logging
import multiprocessing
import os
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from datetime import datetime
from eccs.eccs import ECCS
from IPython.display import display
from tqdm.auto import tqdm
from typing import Optional, Any, Callable, Tuple, Union, List
from varname import nameof

from .aggregate_selector import AggregateSelector
from .ate_calculator import ATECalculator
from .candidate_cause_ranker import CandidateCauseRanker, CandidateCauseRankerMethod
from .causal_discoverer import CausalDiscoverer
from .causal_unit_suggester import CausalUnitSuggester
from .clustering_params import ClusteringParams
from .drain import Drain
from .edge_state_matrix import Edge, EdgeStateMatrix
from .graph_renderer import GraphRenderer
from .interactive_causal_graph_refiner import (
    InteractiveCausalGraphRefiner,
    InteractiveCausalGraphRefinerMethod,
)
from .pickler import Pickler
from .printer import Printer
from .pruner import Pruner
from .regression import Regression
from .tag_utils import TagUtils, TagOrigin
from .variable_name.parsed_variable_name import ParsedVariableName
from .variable_name.prepared_variable_name import PreparedVariableName


# Suppress logging below WARNING level
logging.getLogger().setLevel(logging.WARNING)


class LOGos:
    """
    LOGos provides a high-level interface for causal analysis of event logs.
    """

    def _set_vars_to_defaults(self) -> None:
        """
        Set some of the variables to their default values.
        """
        # The parsed log as a dataframe, and metadata about the parsed variables.
        self._parsed_log: pd.DataFrame = pd.DataFrame()
        self._parsed_variables: pd.DataFrame = pd.DataFrame()
        self._parsed_templates: pd.DataFrame = pd.DataFrame()

        # The variable used to define causal units and the number of causal units.
        self._causal_unit_var: Optional[str] = None
        self._num_causal_units: Optional[int] = None

        # The prepared log as a dataframe, and metadata about the prepared variables.
        self._prepared_log: pd.DataFrame = pd.DataFrame()
        self._prepared_variables: pd.DataFrame = pd.DataFrame()

        # The available aggregation and imputation functions.
        agg_module = importlib.import_module("src.logos.aggimp.agg_funcs")
        self._agg_funcs: dict[str, Callable] = {
            n: f for n, f in inspect.getmembers(agg_module, inspect.isfunction)
        }

        imp_module = importlib.import_module("src.logos.aggimp.imp_funcs")
        self._imp_funcs: dict[str, Callable] = {
            n: f for n, f in inspect.getmembers(imp_module, inspect.isfunction)
        }

        # The graph of causal relationships.
        self._graph: nx.DiGraph = nx.DiGraph()

        # The exploration progress matrix, indicating which edges have been explored.
        self._edge_states: Optional[EdgeStateMatrix] = None

        # The most recent next exploration suggestion.
        self._next_exploration: Optional[str] = None

        # An ECCS object for refinement.
        self._eccs: Optional[ECCS] = None

    @property
    def parsed_log(self) -> pd.DataFrame:
        """
        Get the parsed log as a dataframe.
        """
        return self._parsed_log

    @property
    def parsed_variables(self) -> pd.DataFrame:
        """
        Get the parsed variables as a dataframe.
        """
        return self._parsed_variables

    @property
    def parsed_templates(self) -> pd.DataFrame:
        """
        Get the parsed templates as a dataframe.
        """
        return self._parsed_templates

    @property
    def prepared_log(self) -> pd.DataFrame:
        """
        Get the prepared log as a dataframe.
        """
        return self._prepared_log

    @property
    def prepared_variables(self) -> pd.DataFrame:
        """
        Get the prepared variables as a dataframe.
        """
        return self._prepared_variables

    @property
    def prepared_variable_names(self) -> list[str]:
        """
        Get the names of the prepared variables.
        """
        return self._prepared_variables["Name"].values.tolist()

    @property
    def prepared_variable_tags(self) -> list[str]:
        """
        Get the tags of the prepared variables.
        """
        return self._prepared_variables["Tag"].values.tolist()

    def prepared_variable_names_with_base_x_and_no_pre_post_agg(
        self, x: Union[str, PreparedVariableName]
    ) -> list[str]:
        """
        Get all prepared variables with the given base variable and no pre-
        or post-aggregate values.

        Parameters:
            x: The base variable to check.

        Returns:
            A list of variables with the given base variable and no pre-
            or post-aggregate values.
        """
        return [
            var
            for var in self.prepared_variable_names
            if PreparedVariableName(var).has_base_var(x)
            and PreparedVariableName(var).no_pre_post_aggs()
        ]

    @property
    def num_prepared_variables(self) -> int:
        """
        Get the number of prepared variables.
        """
        return len(self.prepared_variables)

    def __init__(
        self, filename: str, workdir: str, skip_writeout: bool = False
    ) -> None:
        """
        Initialize a LOGos instance, giving it the full path to the log file that will be analyzed.

        Parameters:
            filename: The full path to the log file that will be analyzed.
            workdir: The directory where the parsed and prepared dataframes will be stored.
            skip_writeout: Whether to skip writing out the parsed and prepared dataframes.
        """

        self._set_vars_to_defaults()
        self._filename = filename
        Printer.printv(f"Initialized LOGos with log file {filename}")

        # Set and create working directory
        self._workdir = workdir
        if not os.path.exists(self._workdir):
            os.makedirs(self._workdir, exist_ok=True)
        Printer.printv(f"Work directory set to {self._workdir}")

        self._skip_writeout = skip_writeout

    def set_verbose_to(self, val: bool) -> None:
        """
        Set the verbosity of the printer.

        Parameters:
            val: The new verbosity value.
        """
        Printer.set_verbose(val)
        if self._eccs:
            self._eccs.set_verbose_to(val)

    def _get_filename(self, var_name: str) -> str:
        """
        Create the file name string for dumping/loading pkl files.

        Parameters:
            var_name: The name of the variable to be dumped/loaded.

        Returns:
            The file name string.
        """
        return os.path.join(
            self._workdir,
            os.path.basename(self._filename)
            + f"{var_name}_{self._causal_unit_var}_{self._num_causal_units}.pkl",
        )

    def _find_type(self, row: pd.Series) -> str:
        """
        Identify the type of a parsed variable.

        Parameters:
            row: A row of the parsed variables dataframe.

        Returns:
            The type of the parsed variable as a string. Options are "date", "time", "num" and "str".
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

    def _find_uninteresting(self, row: pd.Series) -> bool:
        """
        Identify whether a parsed variable is likely to be uninteresting.

        Parameters:
            row: A row of the parsed variables dataframe.

        Returns:
            True if the variable is likely to be uninteresting, False otherwise.
        """
        return (
            row["Type"] != "num"
            and (self._parsed_log[row["Name"]].nunique() >= 0.15 * row["Occurrences"])
        ) or (self._parsed_log[row["Name"]].nunique() == 1)

    """
    A default dictionary of regular expressions to be used for parsing the log.
    """
    DEFAULT_REGEX_DICT = {
        "Timestamp": r"\d{4}\-\d{2}\-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
    }

    def parse(
        self,
        regex_dict: dict[str, str] = DEFAULT_REGEX_DICT,
        sim_thresh: float = 0.65,
        depth: int = 5,
        force: bool = False,
        message_prefix: str = r".*",
        enable_gpt_tagging: bool = False,
    ) -> str:
        """
        Parse the log file into a dataframe.

        Parameters:
            regex_dict: (for Drain) A dictionary of regular expressions to be used for parsing.
            sim_thresh: (for Drain) The similarity threshold to be used for parsing.
            depth: (for Drain) The parse tree depth to be used for parsing.
            force: Whether to force re-parsing of the log file.
            message_prefix: A prefix used to identify the beginning of each log message.
                Can be used to collapse multiple lines into a single message. Each line that doesn't start with this
                prefix will be concatenated to the previous log message.
            enable_gpt_tagging: A boolean indicating whether GPT tagging should be enabled.

        Returns:
            The time elapsed for parsing, as a string.
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
        files_exist = not force
        parsed_df_names = [
            nameof(self._parsed_log),
            nameof(self._parsed_templates),
            nameof(self._parsed_variables),
        ]
        for var_name in parsed_df_names:
            if not os.path.isfile(self._get_filename(var_name)):
                files_exist = False
                break

        if files_exist:
            self._parsed_log = Pickler.load(self._get_filename(parsed_df_names[0]))
            self._parsed_templates = Pickler.load(
                self._get_filename(parsed_df_names[1])
            )
            self._parsed_variables = Pickler.load(
                self._get_filename(parsed_df_names[2])
            )
        else:
            (
                self._parsed_log,
                self._parsed_templates,
                self._parsed_variables,
            ) = parser.parse(self._filename.split("/")[-1])
            tqdm.pandas(desc="Determining variable types...")
            self._parsed_variables["Type"] = self._parsed_variables.progress_apply(
                self._find_type, axis=1
            )

            # Cast and convert date columns
            is_date = self._parsed_variables["Type"] == "date"
            date_cols = self._parsed_variables.loc[is_date, "Name"]
            tqdm.pandas(desc="Casting date variables...")
            self._parsed_log[date_cols] = self._parsed_log[date_cols].progress_apply(
                pd.to_datetime, errors="coerce"
            )
            tqdm.pandas(desc="Casting date variables round 2...")
            self._parsed_log[date_cols] = self._parsed_log[date_cols].progress_applymap(
                lambda x: x.timestamp() if not pd.isnull(x) else None
            )
            self._parsed_variables.loc[is_date, "Type"] = "num"

            # Cast and convert time columns
            is_time = self._parsed_variables["Type"] == "time"
            time_cols = self._parsed_variables.loc[is_time, "Name"]
            tqdm.pandas(desc="Casting time variables...")
            self._parsed_log[time_cols] = self._parsed_log[time_cols].progress_apply(
                pd.to_timedelta, errors="coerce"
            )
            tqdm.pandas(desc="Casting time variables round 2...")
            self._parsed_log[time_cols] = self._parsed_log[time_cols].progress_applymap(
                lambda x: x.total_seconds() if not pd.isnull(x) else None
            )
            self._parsed_variables.loc[is_time, "Type"] = "num"

            # Cast numeric columns
            is_num = self._parsed_variables["Type"] == "num"
            numeric_cols = self._parsed_variables.loc[is_num, "Name"]
            tqdm.pandas(desc="Casting numerical variables...")
            self._parsed_log[numeric_cols] = self._parsed_log[
                numeric_cols
            ].progress_apply(pd.to_numeric, errors="coerce")

            # Tag variables.
            tqdm.pandas(desc="Tagging variables...")
            if enable_gpt_tagging:
                tag, tag_origin = zip(
                    *self._parsed_variables.progress_apply(
                        lambda x: TagUtils.waterfall_tag(self.parsed_templates, x),
                        axis=1,
                    )
                )
            else:
                tag, tag_origin = zip(
                    *self._parsed_variables.progress_apply(
                        lambda x: TagUtils.preceding_tokens_tag(x),
                        axis=1,
                    )
                )
            self._parsed_variables["Tag"] = tag
            self._parsed_variables["TagOrigin"] = tag_origin
            TagUtils.deduplicate_tags(self._parsed_variables)

            # Detect identifiers.
            tqdm.pandas(desc="Detecting identifiers...")
            self._parsed_variables["IsUninteresting"] = (
                self._parsed_variables.progress_apply(self._find_uninteresting, axis=1)
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
            Pickler.dump(self._parsed_log, self._get_filename(parsed_df_names[0]))
            Pickler.dump(self._parsed_templates, self._get_filename(parsed_df_names[1]))
            Pickler.dump(self._parsed_variables, self._get_filename(parsed_df_names[2]))

        end_time = datetime.now()
        elapsed = "{:.6f}".format((end_time - start_time).total_seconds())
        Printer.printv(f"Parsing complete in {elapsed} seconds!")
        return elapsed

    def include_in_template(
        self,
        var: str,
        enable_gpt_tagging: bool = False,
        skip_writeout: Optional[bool] = None,
    ) -> None:
        """
        Treat a certain parsed variable as part of its template and regenerate parsed dataframes.

        Parameters:
            var: The name or tag of the variable to be included in its template.
            enable_gpt_tagging: A boolean indicating whether GPT-3.5 tagging should be enabled.
            skip_writeout: Whether to skip writing out the regenerated parsed dataframes. Defaults
                to the value of self._skip_writeout.
        """
        name = TagUtils.name_of(self._parsed_variables, var, "parsed")

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

            self._parsed_templates.loc[len(self._parsed_templates)] = new_template_row
            new_template_ids[value] = new_template_row["TemplateId"]

        self._parsed_templates = self._parsed_templates[
            self._parsed_templates["TemplateId"] != old_template_id
        ].reset_index(drop=True)

        ### Modify _parsed_log

        # Update the template ids of all rows that belonged to the old template
        self._parsed_log["TemplateId"] = self._parsed_log.apply(
            lambda x: (
                new_template_ids[x[name]]
                if (x["TemplateId"] == old_template_id)
                else x["TemplateId"]
            ),
            axis=1,
        )

        # Create new variables for each new template id and assign the value of the old variables to them
        new_variables = []
        for new_template_id in new_template_ids.values():
            for other_idx in new_variable_indices:
                new_var_name = f"{new_template_id}_{str(other_idx)}"
                new_variables.append(new_var_name)
                self._parsed_log[new_var_name] = self._parsed_log.apply(
                    lambda x: (
                        x[f"{old_template_id}_{other_idx}"]
                        if (x["TemplateId"] == new_template_id)
                        else None
                    ),
                    axis=1,
                )

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

                x = {}
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
                    x["Tag"], x["TagOrigin"] = TagUtils.waterfall_tag(
                        self.parsed_templates, pd.Series(x)
                    )
                else:
                    x["Tag"], x["TagOrigin"] = TagUtils.preceding_tokens_tag(
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
        TagUtils.deduplicate_tags(self._parsed_variables)

        # Write out files if appropriate.
        if skip_writeout is None:
            skip_writeout = self._skip_writeout
        if not skip_writeout:
            Pickler.dump(self._parsed_log, self._get_filename(nameof(self._parsed_log)))
            Pickler.dump(
                self._parsed_templates,
                self._get_filename(nameof(self._parsed_templates)),
            )
            Pickler.dump(
                self._parsed_variables,
                self._get_filename(nameof(self._parsed_variables)),
            )

    def tag_parsed_variable(self, name: str, tag: str) -> None:
        """
        Tag a parsed variable.

        Parameters:
            name: The name of the variable to be tagged.
            tag: The tag to be assigned to the variable.
        """
        TagUtils.set_tag(self._parsed_variables, name, tag, "parsed")
        TagUtils.deduplicate_tags(self._parsed_variables)

    def get_tag_of_parsed(self, name: str) -> str:
        """
        Get the tag of a parsed variable.

        Parameters:
            name: The name of the variable.

        Returns:
            The tag of the variable.
        """
        return TagUtils.get_tag(self._parsed_variables, name, "parsed")

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

    def get_causal_unit_info(self) -> Tuple[str, int]:
        """
        Get the variable used to define causal units and the number of
        causal units.

        Returns:
            The name of the variable used to define causal units
            and the number of causal units.
        """
        return self._causal_unit_var, self._num_causal_units

    def suggest_causal_unit_defs(
        self,
        min_causal_units: int = 4,
        num_suggestions: int = 10,
    ) -> Optional[pd.DataFrame]:
        """
        Suggest at most `num_suggestions` causal unit definitions based on IUS maximization,
        while returning at least `min_causal_units` causal units.

        Parameters:
            min_causal_units: The minimum number of causal units that a suggested
                definition should create.
            num_suggestions: The maximum number of causal unit definitions to suggest.

        Returns:
            A DataFrame with one row for each suggested causal unit definition, or `None`
                if no suggestions were made.
        """

        return CausalUnitSuggester.suggest_causal_unit_defs(
            self._parsed_log[self._parsed_variables["Name"].values],
            self._parsed_variables,
            min_causal_units=min_causal_units,
            num_suggestions=num_suggestions,
        )

    def set_causal_unit(
        self,
        var: str,
        num_units: Optional[int] = None,
    ) -> None:
        """
        Set the variable used to define causal units and optionally the number of
        causal units. The latter will be ignored if the variable is categorical, but it
        must be specified if the variable is numerical.

        Parameters:
            var: The name or tag of the variable to be used as the causal unit.
            num_units: The number of causal units to be created.

        Raises:
            ValueError: If the variable is numerical and `num_units` is not specified.
        """
        var_name = TagUtils.name_of(self._parsed_variables, var, "parsed")
        var_type = self._parsed_variables.loc[
            self._parsed_variables["Name"] == var_name, "Type"
        ].values[0]

        if var_type == "num" and num_units is None:
            raise ValueError(
                "The number of causal units must be specified if the causal unit is numerical."
            )

        self._causal_unit_var = var_name
        self._num_causal_units = num_units

        Printer.printv(
            f"Causal unit set to {var_name} (tag: {self.get_tag_of_parsed(var_name)}) "
            + (
                ""
                if not self._num_causal_units
                else f" with {self._num_causal_units} causal units."
            )
        )

    def prepare(
        self,
        custom_agg: dict[str, list[str]] = {},
        custom_imp: dict[str, list[str]] = {},
        count_occurences: bool = False,
        ignore_uninteresting: bool = True,
        force: bool = False,
        lasso_alpha: float = Pruner.LASSO_DEFAULT_ALPHA,
        lasso_max_iter: int = Pruner.LASSO_DEFAULT_MAX_ITER,
        drop_bad_aggs: bool = True,
        reject_prunable_edges: bool = False,
    ) -> str:
        """
        Prepare the log parsed from the table for causal analysis, using aggregation and imputation as needed.

        Parameters:
            custom_agg: A dictionary of custom aggregation functions to be used for specific variables.
            custom_imp: A dictionary of custom imputation functions to be used for specific variables.
            count_occurences: Whether to include extra variables counting the occurence of each template.
            ignore_uninteresting: Whether to ignore uninteresting variables.
            force: Whether to force re-preparation of the log.
            lasso_alpha: The alpha parameter to be used for LASSO regression.
            lasso_max_iter: The maximum number of iterations to be used for LASSO regression.
            drop_bad_aggs: Whether to drop prepared variables that do not add information compared to other
                variables based on the same base variable but using a different aggregation function.
            reject_prunable_edges: Whether to reject edges that are prunable based on LASSO applciation.

        Returns:
            The time elapsed for preparation, as a string.
        """

        start_time = datetime.now()
        # Ensure causal unit is set. TODO: make IUS maximizer the default
        if self._causal_unit_var is None:
            print("Causal unit not defined. Aborting.")
            return None

        # Check if the prepared files already exist.
        files_exist = not force
        prepared_df_names = [
            nameof(self._prepared_log),
            nameof(self._prepared_variables),
        ]
        for var_name in prepared_df_names:
            if not os.path.isfile(self._get_filename(var_name)):
                files_exist = False
                break

        if files_exist:
            self._prepared_log = Pickler.load(self._get_filename(prepared_df_names[0]))
            self._prepared_variables = Pickler.load(
                self._get_filename(prepared_df_names[1])
            )
        else:
            self._prepare_anew(
                custom_agg,
                custom_imp,
                count_occurences=count_occurences,
                ignore_uninteresting=ignore_uninteresting,
                drop_bad_aggs=drop_bad_aggs,
            )

        self._edge_states = EdgeStateMatrix(self.prepared_variable_names)
        if reject_prunable_edges:
            Printer.printv(f"Pruning edges...")
            self.reject_all_prunable_edges(
                lasso_alpha=lasso_alpha, lasso_max_iter=lasso_max_iter
            )

        self._eccs = ECCS(self._prepared_log, nx.DiGraph())
        self._eccs.set_verbose_to(Printer.LOGOS_VERBOSE)

        end_time = datetime.now()
        elapsed = "{:.6f}".format((end_time - start_time).total_seconds())
        Printer.printv(
            f"""Preparation complete in {elapsed} seconds! """
            f"""{np.count_nonzero(self._edge_states.m == -1)} of the {self.num_prepared_variables ** 2} possible edges were auto-rejected."""
        )

        return elapsed

    def _prepare_anew(
        self,
        custom_agg: dict[str, list[str]] = {},
        custom_imp: dict[str, list[str]] = {},
        count_occurences: bool = False,
        ignore_uninteresting: bool = True,
        drop_bad_aggs: bool = True,
    ) -> None:
        """
        Prepare the log anew.

        Parameters:
            custom_agg: A dictionary of custom aggregation functions to be used for specific variables.
            custom_imp: A dictionary of custom imputation functions to be used for specific variables.
            count_occurences: Whether to include extra variables counting the occurence of each template.
            ignore_uninteresting: Whether to ignore uninteresting variables.
            drop_bad_aggs: Whether to drop prepared variables that do not add information compared to other
                variables based on the same base variable but using a different aggregation function.
        """

        Printer.printv(f"Determining the causal unit assignment...")
        causal_unit_assignment = CausalUnitSuggester._discretize(
            self._parsed_log[self._causal_unit_var],
            self._parsed_variables[
                self._parsed_variables["Name"] == self._causal_unit_var
            ]["Type"].values[0],
            self._num_causal_units,
        )

        # Convert keys in custom_agg and custom_imp to the names of the variables, if they are tags.
        custom_agg = {
            TagUtils.name_of(self._parsed_variables, k, "parsed"): v
            for k, v in custom_agg.items()
        }
        custom_imp = {
            TagUtils.name_of(self._parsed_variables, k, "parsed"): v
            for k, v in custom_imp.items()
        }

        # Start with the parsed log, optionally with extra variables counting the occurence of each template.
        if count_occurences:
            Printer.printv(f"Adding template occurrence count variables...")
            self._prepared_log = pd.concat(
                [
                    self._parsed_log,
                    pd.get_dummies(
                        self._parsed_log["TemplateId"],
                        prefix="TemplateId",
                        prefix_sep="=",
                        dtype=float,
                    ),
                ],
                axis=1,
            )
        else:
            self._prepared_log = self._parsed_log.copy(deep=True)

        # No longer need the column storing the actual template IDs
        self._prepared_log.drop(columns="TemplateId", inplace=True)

        # Build dictionary of aggregation functions
        agg_dict: dict[str, str] = {
            variable.Name: (
                custom_agg[variable.Name]
                if variable.Name in custom_agg
                else AggregateSelector.DEFAULT_AGGREGATES[variable.Type]
            )
            for variable in self._parsed_variables.itertuples()
        }

        # Add aggregations for template counts
        for col in self._prepared_log.columns:
            if PreparedVariableName(col).base_var() == "TemplateId":
                agg_dict[col] = ["sum"]

        # Drop uninteresting columns if requested, except if they are the causal unit.
        ui_cols = self._parsed_variables.loc[
            self._parsed_variables["IsUninteresting"], "Name"
        ].values
        ui_cols = [x for x in ui_cols if x != self._causal_unit_var]
        if ignore_uninteresting:
            self._prepared_log.drop(
                columns=ui_cols,
                inplace=True,
            )
            for col in ui_cols:
                agg_dict.pop(col, None)
            Printer.printv(
                f"Dropped {len(ui_cols)} uninteresting columns, out of an original total of {len(self.parsed_variables)}."
            )

        # Ensure the causal unit variable only has one aggregation function
        agg_dict[self._causal_unit_var] = agg_dict[self._causal_unit_var][:1]

        # Perform the aggregation
        Printer.printv("Calculating aggregates for each causal unit...")
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
        self._parsed_variables["Aggregates"] = self._parsed_variables["Name"].map(
            lambda x: agg_dict.get(x, [])
        )
        self._prepared_log.set_index(
            f"{self._causal_unit_var}+{self._parsed_variables[self._parsed_variables['Name'] == self._causal_unit_var]['Aggregates'].values[0][0]}",
            inplace=True,
        )
        self._prepared_log.sort_index(inplace=True)
        self._prepared_log.index = self._prepared_log.index.astype(str)

        # Perform the imputation
        for col in tqdm(self._prepared_log.columns, desc="Imputing missing values..."):
            if self._prepared_log[col].isnull().values.any():
                base_var = PreparedVariableName(col).base_var()
                func_name: str = (
                    custom_imp[base_var] if base_var in custom_imp else "no_imp"
                )
                self._prepared_log[col] = (self._imp_funcs[func_name])(
                    self._prepared_log[col]
                )
        self._prepared_log.dropna(inplace=True)

        # Drop variables that do not add information compared to other variables based on the same base variable
        # but using a different aggregation function.
        if drop_bad_aggs:
            Printer.printv(f"Dropping aggregates that do not add information...")
            cols_to_drop = AggregateSelector.find_uninformative_aggregates(
                self._prepared_log, self._parsed_variables, self._causal_unit_var
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
                        self._prepared_log[col], prefix=col, prefix_sep="=", dtype=float
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
        ].values
        self._prepared_log[date_cols] = self._prepared_log[date_cols].map(
            lambda x: x.timestamp() * 1000.0
        )

        # Convert any time columns to milliseconds
        time_cols = self._prepared_variables.loc[
            self._prepared_variables["Type"] == "time", "Name"
        ].values
        self._prepared_log[time_cols] = self._prepared_log[time_cols].map(
            lambda x: x.total_seconds() * 1000.0
        )

        # Write out prepared log and variables
        if not self._skip_writeout:
            Pickler.dump(
                self._prepared_log, self._get_filename(nameof(self._prepared_log))
            )
            Pickler.dump(
                self._prepared_variables,
                self._get_filename(nameof(self._prepared_variables)),
            )

        Printer.printv(
            f"""Successfully prepared the log with causal unit {self._causal_unit_var} """
            f"""(tag: {self.get_tag_of_parsed(self._causal_unit_var)})"""
            + (
                ""
                if not self._num_causal_units
                else f" with {self._num_causal_units} causal units."
            )
        )

        return

    def _generate_prepared_variables_df(self) -> None:
        """
        Generate dataframe of prepared variables for later tagging etc.
        """

        self._prepared_variables = pd.DataFrame()
        self._prepared_variables["Name"] = self._prepared_log.columns

        # Bring in varable name components leveraging PreparedVariableName
        self._prepared_variables["Base"] = self._prepared_variables["Name"].apply(
            lambda x: PreparedVariableName(x).base_var()
        )
        self._prepared_variables["Pre-agg Value"] = self._prepared_variables[
            "Name"
        ].apply(lambda x: PreparedVariableName(x).pre_agg_value())
        self._prepared_variables["Agg"] = self._prepared_variables["Name"].apply(
            lambda x: PreparedVariableName(x).aggregate()
        )
        self._prepared_variables["Post-agg Value"] = self._prepared_variables[
            "Name"
        ].apply(lambda x: PreparedVariableName(x).post_agg_value())

        # Bring in other info from self._parsed_variables
        self._prepared_variables["Tag"] = self._prepared_variables.apply(
            lambda x: (
                self._parsed_variables.loc[
                    self._parsed_variables["Name"] == x["Base"],
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
        self._prepared_variables["Base Variable Occurences"] = self._prepared_variables[
            "Base"
        ].apply(
            lambda x: (
                self._parsed_variables.loc[
                    self._parsed_variables["Name"] == x, "Occurrences"
                ].values[0]
                if x != "TemplateId"
                else ""
            )
        )
        self._prepared_variables["Type"] = self._prepared_variables["Base"].apply(
            lambda x: (
                self._parsed_variables.loc[
                    self._parsed_variables["Name"] == x, "Type"
                ].values[0]
                if x != "TemplateId"
                else ""
            )
        )
        self._prepared_variables["Examples"] = self._prepared_variables["Base"].apply(
            lambda x: (
                self._parsed_variables.loc[
                    self._parsed_variables["Name"] == x, "Examples"
                ].values[0]
                if x != "TemplateId"
                else ""
            )
        )
        self._prepared_variables["From regex"] = self._prepared_variables["Base"].apply(
            lambda x: (
                self._parsed_variables.loc[
                    self._parsed_variables["Name"] == x, "From regex"
                ].values[0]
                if x != "TemplateId"
                else ""
            )
        )

        # Bring in template text, only for appropriate base variables.
        self._prepared_variables["TemplateText"] = self._prepared_variables.apply(
            lambda x: (
                self._parsed_templates.loc[
                    self._parsed_templates["TemplateId"]
                    == PreparedVariableName(x["Name"]).template_id(),
                    "TemplateText",
                ].values[0]
                if x["From regex"] == False
                else ""
            ),
            axis=1,
        )

    def inspect(
        self,
        var: str,
        ref_var: Optional[str] = None,
        row_limit: Optional[int] = 10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Print information about a specific prepared variable.

        Parameters:
            var: The name or tag of the variable.
            ref_var: The name or tag of a reference variable.
            row_limit: The number of rows of the prepared log to print out,
                to illustrate example values of this variable.

        Returns:
            A tuple containing:
                (1) Information about the base variable of `var`, if `var` is not related to the
                    occurrence count of a template.
                (2) Information about the template of `var`, if `var` was not created from a regex.
                (3) A sample of the prepared log, with `row_limit` rows.
        """

        # Retrieve the name of this variable, if a tag was passed in.
        name = TagUtils.name_of(self._prepared_variables, var, "prepared")

        print(f"Information about prepared variable {name}:\n")
        base_var = PreparedVariableName(name).base_var()
        from_regex = False

        base_var_info_df = pd.DataFrame()
        if base_var != "TemplateId":
            print(f"--> Variable Information about {base_var}:")
            base_var_info_df = self._parsed_variables[
                self._parsed_variables["Name"] == base_var
            ]
            from_regex = base_var_info_df["From regex"].values[0]
            display(base_var_info_df)

        template_info_df = pd.DataFrame()
        if not from_regex:
            template_id = PreparedVariableName(name).template_id()
            print(f"--> Template Information about {template_id}:")
            template_info_df = self._parsed_templates[
                self._parsed_templates["TemplateId"] == template_id
            ]
            display(template_info_df)

        print("--> Causal Unit Partial Information:")
        if row_limit == None:
            row_limit = len(self._prepared_log)
        col_list = [name]
        col_list.extend([ref_var] if ref_var is not None else [])
        prepared_log_info_df = self._prepared_log[col_list].head(row_limit)
        col_names = [f"{name} (candidate)"]
        col_names.extend([f"{ref_var} (outcome)"] if ref_var is not None else [])
        prepared_log_info_df.columns = col_names
        display(prepared_log_info_df)

        return base_var_info_df, template_info_df, prepared_log_info_df

    def clear_graph(self, clear_edge_states: bool = True) -> None:
        """
        Clear the graph and possibly edge states.

        Parameters:
            clear_edge_states: Whether to also clear the edge states.
        """
        self._graph = nx.DiGraph()
        if clear_edge_states:
            self._edge_states = EdgeStateMatrix(self.prepared_variable_names)
        if self._eccs:
            self._eccs.clear_graph(clear_edge_states)

    def display_graph(self) -> None:
        """
        Display the current graph.
        """
        GraphRenderer.display_graph(self._graph, self._prepared_variables)

    def save_graph(self, filename: str) -> None:
        """
        Save the current graph to a file.

        Parameters:
            filename: The name of the file to save to.
        """
        GraphRenderer.save_graph(self._graph, self._prepared_variables, filename)

    def accept(
        self,
        src: str,
        dst: str,
        also_fix: bool = False,
        interactive: bool = True,
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """
        Mark a causal graph edge as accepted.

        This will also reject the edge from `dst` to `src` and remove any other variables with the
        same base variable as either `src` or `dst` from consideration for the partial causal graph.

        Parameters:
            src: The name or tag of the source variable.
            dst: The name or tag of the destination variable.
            also_fix: Whether to also fix the edge, for ECCS.
            interactive: Whether to display the graph interactively after accepting the edge.

        Returns:
            A tuple containing:
                (1) the exploration score after the edge addition,
                (2) the max-impact variable to explore next, if any,
                (3) optionally a string representation of the graph, if `interactive` is False.
        """

        src_name = TagUtils.name_of(self._prepared_variables, src, "prepared")
        dst_name = TagUtils.name_of(self._prepared_variables, dst, "prepared")
        to_drop = self._edge_states.mark_edge(src_name, dst_name, "Accepted")
        for node in to_drop:
            if node in self._graph.nodes:
                self._graph.remove_node(node)

        self._graph.add_node(src_name)
        self._graph.add_node(dst_name)
        self._graph.add_edge(src_name, dst_name)
        if (dst_name, src_name) in self._graph.edges:
            self._graph.remove_edge(dst_name, src_name)
        if interactive:
            GraphRenderer.display_graph(self._graph, self._prepared_variables)
        if self._eccs:
            self._eccs.remove_edge(dst_name, src_name)
            self._eccs.add_edge(src_name, dst_name)
            if also_fix:
                self._eccs.fix_edge(src_name, dst_name)

        return (
            self.exploration_score,
            self.suggest_next_exploration(),
            (
                GraphRenderer.draw_graph(self._graph, self._prepared_variables)
                if not interactive
                else ""
            ),
        )

    def reject(
        self,
        src: str,
        dst: str,
        also_ban: bool,
        interactive: bool = True,
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """
        Mark a causal graph edge as rejected.

        Parameters:
            src: The name or tag of the source variable.
            dst: The name or tag of the destination variable.
            also_ban: Whether to also ban the edge, for ECCS.
            interactive: Whether to display the graph interactively after rejecting the edge.

        Returns:
            A tuple containing:
                (1) the exploration score after the edge rejection,
                (2) the max-impact variable to explore next, if any,
                (3) optionally a string representation of the graph, if `interactive` is False.
        """

        src_name = TagUtils.name_of(self._prepared_variables, src, "prepared")
        dst_name = TagUtils.name_of(self._prepared_variables, dst, "prepared")
        self._edge_states.mark_edge(src_name, dst_name, "Rejected")
        if self._eccs and also_ban:
            self._eccs.ban_edge(src_name, dst_name)

        if interactive:
            GraphRenderer.display_graph(self._graph, self._prepared_variables)

        return (
            self.exploration_score,
            self.suggest_next_exploration(),
            (
                GraphRenderer.draw_graph(self._graph, self._prepared_variables)
                if not interactive
                else ""
            ),
        )

    def reject_undecided_incoming(
        self, dst: str, also_ban: bool, interactive: bool = True
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """
        Mark all undecided incoming edges to a variable as rejected.

        Parameters:
            dst: The name or tag of the destination variable.
            also_ban: Whether to also ban the edges, for ECCS.
            interactive: Whether to display the graph interactively after rejecting the edges.

        Returns:
            A tuple containing:
                (1) the exploration score after the edge rejections,
                (2) the max-impact variable to explore next, if any,
                (3) optionally a string representation of the graph, if `interactive` is False.
        """
        dst_name = TagUtils.name_of(self._prepared_variables, dst, "prepared")
        for v in self.prepared_variable_names:
            if self._edge_states.get_edge_state(v, dst_name) == "Undecided":
                self._edge_states.mark_edge(v, dst_name, "Rejected")
                if self._eccs and also_ban:
                    self._eccs.ban_edge(v, dst_name)

        if interactive:
            GraphRenderer.display_graph(self._graph, self._prepared_variables)

        return (
            self.exploration_score,
            self.suggest_next_exploration(),
            (
                GraphRenderer.draw_graph(self._graph, self._prepared_variables)
                if not interactive
                else ""
            ),
        )

    def reject_undecided_outgoing(
        self, src: str, also_ban: bool, interactive: bool = True
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """
        Mark all undecided outgoing edges from a variable as rejected.

        Parameters:
            src: The name or tag of the source variable.
            also_ban: Whether to also ban the edges, for ECCS.
            interactive: Whether to display the graph interactively after rejecting the edges.

        Returns:
            A tuple containing:
                (1) the exploration score after the edge rejections,
                (2) the max-impact variable to explore next, if any,
                (3) optionally a string representation of the graph, if `interactive` is False.
        """
        src_name = TagUtils.name_of(self._prepared_variables, src, "prepared")
        for v in self.prepared_variable_names:
            if self._edge_states.get_edge_state(src_name, v) == "Undecided":
                self._edge_states.mark_edge(src_name, v, "Rejected")
                if self._eccs and also_ban:
                    self._eccs.ban_edge(src_name, v)

        if interactive:
            GraphRenderer.display_graph(self._graph, self._prepared_variables)

        return (
            self.exploration_score,
            self.suggest_next_exploration(),
            (
                GraphRenderer.draw_graph(self._graph, self._prepared_variables)
                if not interactive
                else ""
            ),
        )

    def reject_all_prunable_edges(
        self,
        also_ban: bool,
        lasso_alpha: float = Pruner.LASSO_DEFAULT_ALPHA,
        lasso_max_iter: int = Pruner.LASSO_DEFAULT_MAX_ITER,
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """
        For every prepared variable, reject all incoming edges that start at a variable
        that is pruned by our pruning approach. This may be time-consuming depending on the number of variables.

        Parameters:
            also_ban: Whether to also ban the edges, for ECCS.
            lasso_alpha: The alpha parameter to be used for Lasso regression.
            lasso_max_iter: The maximum number of iterations to be used for Lasso regression.

        Returns:
            A tuple containing:
                (1) the exploration score after the edge rejections,
                (2) the max-impact variable to explore next, if any,
                (3) optionally a string representation of the graph, if `interactive` is False.
        """
        num_processors = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processors) as pool:
            all_candidates = pool.starmap(
                Pruner.prune_with_lasso,
                tqdm(
                    [
                        (self._prepared_log, [target], lasso_alpha, lasso_max_iter)
                        for target in self.prepared_variable_names
                    ],
                    total=self.num_prepared_variables,
                    desc="Finding pruned variables...",
                ),
            )

        Printer.printv(all_candidates)

        for candidates, target in zip(all_candidates, self.prepared_variable_names):
            non_candidates = (
                set(self._prepared_log.columns) - set(candidates) - set([target])
            )
            for nc in non_candidates:
                self._edge_states.mark_edge(nc, target, "Rejected")
                if self._eccs and also_ban:
                    self._eccs.ban_edge(nc, target)

        return (
            self.exploration_score,
            self.suggest_next_exploration(),
            GraphRenderer.draw_graph(self._graph, self._prepared_variables),
        )

    @property
    def exploration_score(self) -> float:
        """
        Calculate the exploration score of the current partial causal graph,
        based on the edge state matrix.

        Returns:
            The exploration score of the current partial causal graph.
        """
        # Number of edges incident to a node in the current partial graph
        M = self._graph.number_of_nodes()
        N = self.num_prepared_variables
        incident = M * (2 * N - M - 1)
        if incident == 0:
            return 0

        # Number of edges among the incident that have been considered
        graph_var_indices = [self._edge_states.idx(x) for x in list(self._graph.nodes)]
        other_indices = list(np.setdiff1d(np.arange(N), graph_var_indices))
        considered = np.sum(
            self._edge_states.m[graph_var_indices][:, graph_var_indices] != 0
        )
        considered -= M  # subtract self-edges
        considered += np.sum(
            self._edge_states.m[graph_var_indices][:, other_indices] != 0
        )
        considered += np.sum(
            self._edge_states.m[other_indices][:, graph_var_indices] != 0
        )

        Printer.printv(f"Considered: {considered}")
        Printer.printv(f"Incident: {incident}")

        return considered / incident

    def rank_candidate_causes(
        self,
        target: Optional[str] = None,
        ignore: Optional[List[str]] = None,
        method: CandidateCauseRankerMethod = CandidateCauseRankerMethod.LOGOS,
        prune_candidates: bool = True,
        lasso_alpha: float = Pruner.LASSO_DEFAULT_ALPHA,
        lasso_max_iter: int = Pruner.LASSO_DEFAULT_MAX_ITER,
        model: str = "gpt-4o-mini-2024-07-18",
        gpt_log_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Present the user with ranked candidate causes for `target`. If no `target`
        is specified, the most recent suggestion of `suggest_next_exploration()` is used, if any.
        If `ignore` is specified, the variables in `ignore` are not considered as candidate causes.

        Parameters:
            target: The name or tag of the target variable.
            ignore: A list of variables to ignore.
            method: The method to use for ranking candidate causes.
            prune_candidates: Whether to prune the candidate causes using Lasso regression. Only
                applies if `method` is `CandidateCauseRankerMethod.LOGOS`.
            lasso_alpha: The alpha parameter to be used for Lasso regression. Only applies if
                `method` is `CandidateCauseRankerMethod.LOGOS` and `prune_candidates` is True.
            lasso_max_iter: The maximum number of iterations to be used for Lasso regression. Only
                applies if `method` is `CandidateCauseRankerMethod.LOGOS` and `prune_candidates` is True.
            model: The model to use for the langmodel method. Only applies if the method is
                `CandidateCauseRankerMethod.LANGMODEL`.
            gpt_log_path: The path to the log file to use for the langmodel method. Only applies if
                the method is `CandidateCauseRankerMethod.LANGMODEL`.
        Returns:
            A tuple containing:
            (1) A dataframe containing the candidate causes for `target` and
            (2) The time elapsed for exploration, as a string.
        """

        start_time = datetime.now()

        # Handle the case where the user has not specified a target.
        if target is None and self._next_exploration is None:
            Printer.printv("No target specified.")
            return pd.DataFrame(columns=CandidateCauseRanker.COLUMN_ORDER), ""
        elif target is None:
            target = self._next_exploration

        # If the user provided the target as a tag, retrieve its name
        target = TagUtils.name_of(self._prepared_variables, target, "prepared")

        # Use the specified method to rank candidate causes
        result_df, pruned = CandidateCauseRanker.rank(
            self.prepared_log,
            self.prepared_variables,
            target,
            ignore,
            method,
            prune_candidates,
            lasso_alpha,
            lasso_max_iter,
            model,
            (
                gpt_log_path
                if (gpt_log_path is not None)
                else os.path.join(
                    self._workdir,
                    f"ranker-gpt-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log",
                )
            ),
        )

        # Mark the edges rejected by the pruning step, if any.
        for var in pruned:
            self._edge_states.mark_edge(var, target, "Rejected")

        # Add fields to the returned dataframe
        result_df["Candidate->Target Edge Status"] = result_df["Candidate"].apply(
            lambda x: self._edge_states.get_edge_state(x, target)
        )
        result_df["Target->Candidate Edge Status"] = result_df["Candidate"].apply(
            lambda x: self._edge_states.get_edge_state(target, x)
        )

        ret_val = result_df[CandidateCauseRanker.COLUMN_ORDER]

        end_time = datetime.now()
        elapsed = "{:.6f}".format((end_time - start_time).total_seconds())
        Printer.printv(f"Candidate cause exploration complete in {elapsed} seconds!")

        return ret_val, elapsed

    def get_causal_graph_refinement_suggestion(
        self,
        method: InteractiveCausalGraphRefinerMethod = InteractiveCausalGraphRefinerMethod.LOGOS,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        model: str = "gpt-4o-mini-2024-07-18",
        gpt_log_path: Optional[str] = None,
    ) -> Tuple[Edge, str]:
        """
        Present the user with an edge, the presence and direction of which they should assess.

        Parameters:
            method: The method to use for producing a causal graph refinement suggestion.
            treatment: The name or tag of the treatment variable. Only applies if the method is
                `InteractiveCausalGraphRefinerMethod.LOGOS`.
            outcome: The name or tag of the outcome variable. Only applies if the method is
                `InteractiveCausalGraphRefinerMethod.LOGOS`.
            model: The model to use for the langmodel method. Only applies if the method is
                `CandidateCauseRankerMethod.LANGMODEL`.
            gpt_log_path: The path to the log file to use for the langmodel method. Only applies if
                the method is `CandidateCauseRankerMethod.LANGMODEL`.
        Returns:
            A tuple containing:
            (1) The edge to assess, as an Edge object, and
            (2) The time elapsed for generating the suggestion, as a string.
        """

        start_time = datetime.now()

        treatment_name = TagUtils.name_of(
            self._prepared_variables, treatment, "prepared"
        )
        outcome_name = TagUtils.name_of(self._prepared_variables, outcome, "prepared")

        edge = InteractiveCausalGraphRefiner.get_suggestion(
            self.prepared_log,
            method,
            self._eccs,
            treatment_name,
            outcome_name,
            self._graph,
            model,
            (
                gpt_log_path
                if (gpt_log_path is not None)
                else os.path.join(
                    self._workdir,
                    f"refiner-gpt-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log",
                )
            ),
            self.prepared_variables,
        )

        edge_tags = None
        if edge:
            edge_tags = tuple(
                TagUtils.tag_of(self._prepared_variables, x, "prepared") for x in edge
            )

        end_time = datetime.now()
        elapsed = "{:.6f}".format((end_time - start_time).total_seconds())
        Printer.printv(f"Candidate cause exploration complete in {elapsed} seconds!")

        return edge_tags, elapsed

    def suggest_next_exploration(self) -> Optional[str]:
        """
        Suggest the variable that should be explored next. Suggest the prepared variable in the partial causal graph
        that has the most (nonzero) Unexplored incoming edges, if any; otherwise suggest the prepared variable
        with the most (nonzero) Undecided incoming edges, even if it is not in the partial causal graph.

        If all edges are decided, return None.

        Returns:
            The name of the variable to explore next.
        """

        # Try to find a suggestion from the partial causal graph.
        node_names = list(self._graph.nodes)
        graph_var_indices = [self._edge_states.idx(x) for x in node_names]
        graph_var_incoming_edge_states = self._edge_states.m[:, graph_var_indices]
        undecided_edges_per_col = (
            np.sum(graph_var_incoming_edge_states == 0, axis=0)
            if len(graph_var_incoming_edge_states) > 0
            else []
        )
        max_undecided = (
            np.max(undecided_edges_per_col) if len(undecided_edges_per_col) > 0 else 0
        )

        if max_undecided > 0:
            max_undecided_idx = np.argmax(undecided_edges_per_col)
            self._next_exploration = node_names[max_undecided_idx]
            return self._next_exploration

        # If no suggestion was found, try to find a suggestion from the entire collection of prepared variables.
        undecided_edges_per_col = np.sum(self._edge_states.m == 0, axis=0)
        max_undecided = np.max(undecided_edges_per_col)

        if max_undecided > 0:
            max_undecided_idx = np.argmax(undecided_edges_per_col)
            self._next_exploration = self._prepared_variables.loc[
                max_undecided_idx, "Name"
            ]
            return self._next_exploration

        # If no suggestion was found, return None.
        self._next_exploration = None
        return None

    def discover_graph(
        self,
        method: str = "hill_climb",
        max_cond_vars: int = 3,
        model: str = "gpt-3.5-turbo",
    ) -> None:
        """
        Discover a causal graph based on the prepared table automatically.

        Parameters:
            method: The method to be used for graph discovery, among "PC", "hill_climb", "exhaustive" and "GPT".
            max_cond_vars: The maximum number of conditioning variables to be used for PC.
            model: The model to be used for GPT-based graph discovery.

        """

        if method == "PC":
            self._graph = CausalDiscoverer.pc(
                self._prepared_log, max_cond_vars=max_cond_vars
            )
        elif method == "hill_climb":
            self._graph = CausalDiscoverer.hill_climb(self._prepared_log)
        elif method == "exhaustive":
            self._graph = CausalDiscoverer.exhaustive(self._prepared_log)
        elif method == "GPT":
            self._graph = CausalDiscoverer.gpt(self._prepared_log, model=model)
        else:
            raise ValueError(f"Invalid graph discovery method {method}")

        self._edge_states.clear_and_set_from_graph(self._graph)

    def get_adjusted_ate(
        self,
        treatment: str,
        outcome: str,
        confounder: Optional[str] = None,
    ) -> float:
        """
        Calculate the adjusted ATE of `treatment` on `outcome`, given the current partial causal graph.

        Parameters:
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.
            confounder: The name or tag of a confounder variable. If specified, overrides the current partial
                causal graph in favor of a three-node graph with `treatment`, `outcome` and `confounder`.

        Returns:
            The adjusted ATE of `treatment` on `outcome`.
        """
        return ATECalculator.get_ate_and_confidence(
            self.prepared_log,
            self.prepared_variables,
            treatment,
            outcome,
            confounder,
            graph=self._graph,
            calculate_p_value=False,
            calculate_std_error=False,
        )["ATE"]

    def get_unadjusted_ate(
        self,
        treatment: str,
        outcome: str,
    ) -> float:
        """
        Calculate the unadjusted ATE of `treatment` on `outcome`, ignoring the current partial causal graph
        in favor of a two-node graph with just `treatment` and `outcome`.

        Parameters:
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.

        Returns:
            The unadjusted ATE of `treatment` on `outcome`.
        """
        return ATECalculator.get_ate_and_confidence(
            self.prepared_log,
            self.prepared_variables,
            treatment,
            outcome,
            calculate_p_value=False,
            calculate_std_error=False,
        )["ATE"]
