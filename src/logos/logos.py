import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.logos.ate_calculator import ATECalculator
from src.logos.candidate_cause_ranker import CandidateCauseRankerMethod
from src.logos.causal_explorer import CausalExplorer
from src.logos.dataset_preparer import CausalDatasetPreparer
from src.logos.exceptions import UnsupportedOperationError
from src.logos.interactive_causal_graph_refiner import (
    InteractiveCausalGraphRefinerMethod,
)
from src.logos.log_parser import LogParser
from src.logos.parsed_table_input import ParsedTableInput
from src.logos.prepared_table_input import PreparedTableInput
from src.logos.pruner import Pruner
from src.logos.types import Types
from src.logos.variable_name.prepared_variable_name import PreparedVariableName

_logger = logging.getLogger(__name__)
# Suppress LOGos debug messages by default; call set_verbose_to(True) to enable.
logging.getLogger("src.logos").setLevel(logging.WARNING)


class LOGos:
    """
    LOGos provides a high-level interface for causal analysis of event logs.
    """

    DEFAULT_REGEX_DICT = LogParser.DEFAULT_REGEX_DICT
    DEFAULT_MESSAGE_PREFIX = LogParser.DEFAULT_MESSAGE_PREFIX

    def __init__(
        self, filename: str, workdir: str, skip_writeout: bool = False
    ) -> None:
        """
        Initialize a LOGos instance, giving it the full path to the log file
        that will be analyzed.

        Parameters:
            filename: The full path to the log file that will be analyzed.
            workdir: The directory where the parsed and prepared dataframes will
                be stored.
            skip_writeout: Whether to skip writing out the parsed and prepared
                dataframes.
        """
        self._parser: Optional[LogParser | ParsedTableInput] = LogParser(
            filename, workdir, skip_writeout
        )
        self._preparer: Optional[CausalDatasetPreparer] = CausalDatasetPreparer(
            self._parser
        )
        self._explorer: Optional[CausalExplorer] = None

    @classmethod
    def _create(cls) -> "LOGos":
        """Return an uninitialised instance (used by factory methods)."""
        instance = cls.__new__(cls)
        instance._parser = None
        instance._preparer = None
        instance._explorer = None
        return instance

    @classmethod
    def from_parsed_table(
        cls,
        data: pd.DataFrame,
        workdir: str,
        source_id: str = "parsed_input",
        variable_tags: Optional[dict[str, str]] = None,
        skip_writeout: bool = False,
    ) -> "LOGos":
        """
        Create a LOGos instance from a pre-parsed DataFrame (EP-2).

        The DataFrame is treated as if it were the output of parse():
        one row per log event, one column per field.  Call
        set_causal_unit() and prepare() as normal after construction.

        Parameters:
            data: Pre-parsed table (one row per event).
            workdir: Directory for prepare-stage cache files.
            source_id: Prefix for cache filenames (analogous to the log
                filename).
            variable_tags: Optional column-name → tag mapping.
            skip_writeout: Whether to skip writing prepare cache files.
        """
        instance = cls._create()
        instance._parser = ParsedTableInput(
            data, workdir, source_id, variable_tags, skip_writeout
        )
        instance._preparer = CausalDatasetPreparer(instance._parser)
        return instance

    @classmethod
    def from_prepared_table(
        cls,
        data: pd.DataFrame,
        workdir: str,
        variable_tags: Optional[dict[str, str]] = None,
    ) -> "LOGos":
        """
        Create a LOGos instance from an already-prepared DataFrame (EP-3).

        The DataFrame is treated as if it were the output of prepare():
        one row per causal unit, one column per feature.  The instance is
        ready for exploration immediately — no parse() or prepare() calls
        are needed.

        Parameters:
            data: Prepared table (one row per causal unit).
            workdir: Directory used for GPT log files during exploration.
            variable_tags: Optional column-name → tag mapping.
        """
        instance = cls._create()
        pti = PreparedTableInput(data, workdir, variable_tags)
        instance._explorer = CausalExplorer(
            pti.prepared_log,
            pti.prepared_variables,
            pti.parsed_variables,
            pti.parsed_templates,
            workdir,
        )
        instance._explorer._init_eccs()
        return instance

    def set_verbose_to(self, val: bool) -> None:
        """Set LOGos logging verbosity (True = DEBUG, False = WARNING)."""
        level = logging.DEBUG if val else logging.WARNING
        logging.getLogger("src.logos").setLevel(level)
        if self._explorer and self._explorer._eccs:
            self._explorer._eccs.set_verbose_to(val)

    def _require_parser(self) -> LogParser:
        """Raise if the instance was not created from a raw log file."""
        if not isinstance(self._parser, LogParser):
            raise UnsupportedOperationError(
                "This operation requires a LOGos instance created from a raw "
                "log file.  Use LOGos(filename=...) instead of "
                "LOGos.from_parsed_table() or LOGos.from_prepared_table()."
            )
        return self._parser

    def _require_preparer(self) -> CausalDatasetPreparer:
        """Raise if the instance has no prepare stage (EP-3)."""
        if self._preparer is None:
            raise UnsupportedOperationError(
                "This operation is not available for instances created via "
                "LOGos.from_prepared_table()."
            )
        return self._preparer

    @property
    def parsed_log(self) -> pd.DataFrame:
        return self._parser.parsed_log

    @property
    def parsed_variables(self) -> pd.DataFrame:
        return self._parser.parsed_variables

    @property
    def parsed_templates(self) -> pd.DataFrame:
        return self._parser.parsed_templates

    @property
    def prepared_log(self) -> pd.DataFrame:
        return self._preparer.prepared_log

    @property
    def prepared_variables(self) -> pd.DataFrame:
        return self._preparer.prepared_variables

    @property
    def prepared_variable_names(self) -> list[str]:
        return self._preparer.prepared_variable_names

    @property
    def prepared_variable_tags(self) -> list[str]:
        return self._preparer.prepared_variable_tags

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
            for var in self._preparer.prepared_variable_names
            if PreparedVariableName(var).has_base_var(x)
            and PreparedVariableName(var).no_pre_post_aggs()
        ]

    @property
    def num_prepared_variables(self) -> int:
        return self._preparer.num_prepared_variables

    def parse(
        self,
        regex_dict: dict[str, str] = DEFAULT_REGEX_DICT,
        sim_thresh: float = 0.65,
        depth: int = 5,
        force: bool = False,
        message_prefix: str = DEFAULT_MESSAGE_PREFIX,
        enable_gpt_tagging: bool = False,
    ) -> str:
        """Parse the log file; see LogParser.parse() for full documentation."""
        parser = self._require_parser()
        return parser.parse(
            regex_dict,
            sim_thresh,
            depth,
            force,
            message_prefix,
            enable_gpt_tagging,
        )

    def include_in_template(
        self,
        var: str,
        enable_gpt_tagging: bool = False,
        skip_writeout: Optional[bool] = None,
    ) -> None:
        """
        Treat a parsed variable as part of its template; see
        LogParser.include_in_template().
        """
        parser = self._require_parser()
        return parser.include_in_template(
            var, enable_gpt_tagging, skip_writeout
        )

    def tag_parsed_variable(self, name: str, tag: str) -> None:
        """Tag a parsed variable."""
        parser = self._require_parser()
        return parser.tag_parsed_variable(name, tag)

    def get_tag_of_parsed(self, name: str) -> str:
        """Get the tag of a parsed variable."""
        parser = self._require_parser()
        return parser.get_tag_of_parsed(name)

    def tag_prepared_variable(self, name: str, tag: str) -> None:
        """Tag a prepared variable."""
        preparer = self._require_preparer()
        return preparer.tag_prepared_variable(name, tag)

    def get_tag_of_prepared(self, name: str) -> str:
        """Get the tag of a prepared variable."""
        preparer = self._require_preparer()
        return preparer.get_tag_of_prepared(name)

    def get_causal_unit_info(self) -> Tuple[Optional[str], Optional[int]]:
        """Get the causal unit variable and number of causal units."""
        self._require_preparer()
        assert self._preparer is not None
        return self._preparer.get_causal_unit_info()

    def suggest_causal_unit_defs(
        self,
        min_causal_units: int = 4,
        num_suggestions: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Suggest causal unit definitions based on IUS maximization."""
        self._require_preparer()
        assert self._preparer is not None
        return self._preparer.suggest_causal_unit_defs(
            min_causal_units, num_suggestions
        )

    def set_causal_unit(
        self,
        var: str,
        num_units: Optional[int] = None,
    ) -> None:
        """Set the variable used to define causal units."""
        self._require_preparer()
        assert self._preparer is not None
        return self._preparer.set_causal_unit(var, num_units)

    def prepare(
        self,
        custom_agg: Optional[dict[str, list[str]]] = None,
        custom_imp: Optional[dict[str, str]] = None,
        count_occurences: bool = False,
        ignore_uninteresting: bool = True,
        force: bool = False,
        lasso_alpha: float = Pruner.LASSO_DEFAULT_ALPHA,
        lasso_max_iter: int = Pruner.LASSO_DEFAULT_MAX_ITER,
        drop_bad_aggs: bool = True,
        reject_prunable_edges: bool = False,
    ) -> Optional[str]:
        """
        Prepare the log parsed from the table for causal analysis, using
        aggregation and imputation as needed.

        Parameters:
            custom_agg: A dictionary of custom aggregation functions to be used
                for specific variables.
            custom_imp: A dictionary of custom imputation functions to be used
                for specific variables.
            count_occurences: Whether to include extra variables counting the
                occurence of each template.
            ignore_uninteresting: Whether to ignore uninteresting variables.
            force: Whether to force re-preparation of the log.
            lasso_alpha: The alpha parameter to be used for LASSO regression.
            lasso_max_iter: The maximum number of iterations to be used for
                LASSO regression.
            drop_bad_aggs: Whether to drop prepared variables that do not add
                information compared to other variables based on the same base
                variable but using a different aggregation function.
            reject_prunable_edges: Whether to reject edges that are prunable
            based on LASSO application.

        Returns:
            The time elapsed for preparation, as a string, or `None` if the
                preparation was aborted.
        """

        start_time = datetime.now()
        if custom_agg is None:
            custom_agg = {}
        if custom_imp is None:
            custom_imp = {}
        preparer = self._require_preparer()
        assert self._parser is not None

        if not preparer.prepare(
            custom_agg,
            custom_imp,
            count_occurences,
            ignore_uninteresting,
            force,
            drop_bad_aggs,
        ):
            return None

        self._explorer = CausalExplorer(
            preparer.prepared_log,
            preparer.prepared_variables,
            self._parser.parsed_variables,
            self._parser.parsed_templates,
            self._parser.workdir,
        )
        if reject_prunable_edges:
            _logger.debug("Pruning edges...")
            self._explorer.reject_all_prunable_edges(
                also_ban=True,
                lasso_alpha=lasso_alpha,
                lasso_max_iter=lasso_max_iter,
            )
        self._explorer._init_eccs()

        elapsed = (datetime.now() - start_time).total_seconds()
        _logger.debug(
            f"Preparation complete in {elapsed:.6f}s! "
            f"{np.count_nonzero(self._explorer._edge_states.m == -1)} of the "
            f"{self._explorer.num_prepared_variables ** 2} possible edges were "
            "auto-rejected."
        )
        return elapsed

    def inspect(
        self,
        var: str,
        ref_var: Optional[str] = None,
        row_limit: Optional[int] = 10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Print information about a specific prepared variable."""
        assert self._explorer is not None
        return self._explorer.inspect(var, ref_var, row_limit)

    def clear_graph(self, clear_edge_states: bool = True) -> None:
        """Clear the graph and possibly edge states."""
        assert self._explorer is not None
        return self._explorer.clear_graph(clear_edge_states)

    def display_graph(self) -> None:
        """Display the current graph."""
        assert self._explorer is not None
        self._explorer.display_graph()

    def save_graph(self, filename: str) -> None:
        """Save the current graph to a file."""
        assert self._explorer is not None
        self._explorer.save_graph(filename)

    def accept(
        self,
        src: str,
        dst: str,
        also_fix: bool = False,
        interactive: bool = True,
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """Mark a causal graph edge as accepted."""
        assert self._explorer is not None
        return self._explorer.accept(src, dst, also_fix, interactive)

    def reject(
        self,
        src: str,
        dst: str,
        also_ban: bool = False,
        interactive: bool = True,
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """Mark a causal graph edge as rejected."""
        assert self._explorer is not None
        return self._explorer.reject(src, dst, also_ban, interactive)

    def reject_undecided_incoming(
        self, dst: str, also_ban: bool = False, interactive: bool = True
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """Mark all undecided incoming edges to a variable as rejected."""
        assert self._explorer is not None
        return self._explorer.reject_undecided_incoming(
            dst, also_ban, interactive
        )

    def reject_undecided_outgoing(
        self, src: str, also_ban: bool = False, interactive: bool = True
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """Mark all undecided outgoing edges from a variable as rejected."""
        assert self._explorer is not None
        return self._explorer.reject_undecided_outgoing(
            src, also_ban, interactive
        )

    def reject_all_prunable_edges(
        self,
        also_ban: bool = False,
        lasso_alpha: float = Pruner.LASSO_DEFAULT_ALPHA,
        lasso_max_iter: int = Pruner.LASSO_DEFAULT_MAX_ITER,
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """Reject all edges prunable by LASSO."""
        assert self._explorer is not None
        return self._explorer.reject_all_prunable_edges(
            also_ban, lasso_alpha, lasso_max_iter
        )

    @property
    def exploration_score(self) -> float:
        """Exploration score of the current partial causal graph."""
        assert self._explorer is not None
        return self._explorer.exploration_score

    def suggest_next_exploration(self) -> Optional[str]:
        """Suggest the variable that should be explored next."""
        assert self._explorer is not None
        return self._explorer.suggest_next_exploration()

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
        """Present the user with ranked candidate causes for `target`."""
        assert self._explorer is not None
        return self._explorer.rank_candidate_causes(
            target,
            ignore,
            method,
            prune_candidates,
            lasso_alpha,
            lasso_max_iter,
            model,
            gpt_log_path,
        )

    def get_causal_graph_refinement_suggestion(
        self,
        method: InteractiveCausalGraphRefinerMethod = (
            InteractiveCausalGraphRefinerMethod.LOGOS
        ),
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        model: str = "gpt-4o-mini-2024-07-18",
        gpt_log_path: Optional[str] = None,
    ) -> Tuple[Optional[Types.Edge], str]:
        """Present the user with an edge to assess."""
        assert self._explorer is not None
        return self._explorer.get_causal_graph_refinement_suggestion(
            method, treatment, outcome, model, gpt_log_path
        )

    def get_adjusted_ate(
        self,
        treatment: str,
        outcome: str,
        confounder: Optional[str] = None,
    ) -> float:
        """
        Calculate the adjusted ATE of `treatment` on `outcome`, given the
        current partial causal graph.

        Parameters:
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.
            confounder: The name or tag of a confounder variable. If specified,
                overrides the current partial causal graph in favor of a
                three-node graph with `treatment`, `outcome` and `confounder`.

        Returns:
            The adjusted ATE of `treatment` on `outcome`.
        """
        self._require_preparer()
        assert self._preparer is not None
        return ATECalculator.get_ate_and_confidence(
            self._preparer.prepared_log,
            self._preparer.prepared_variables,
            treatment,
            outcome,
            confounder,
            graph=self._explorer._graph if self._explorer else None,
            calculate_p_value=False,
            calculate_std_error=False,
        )["ATE"]

    def get_unadjusted_ate(
        self,
        treatment: str,
        outcome: str,
    ) -> float:
        """
        Calculate the unadjusted ATE of `treatment` on `outcome`, ignoring the
        current partial causal graph in favor of a two-node graph with just
        `treatment` and `outcome`.

        Parameters:
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.

        Returns:
            The unadjusted ATE of `treatment` on `outcome`.
        """
        self._require_preparer()
        assert self._preparer is not None
        return ATECalculator.get_ate_and_confidence(
            self._preparer.prepared_log,
            self._preparer.prepared_variables,
            treatment,
            outcome,
            calculate_p_value=False,
            calculate_std_error=False,
        )["ATE"]
