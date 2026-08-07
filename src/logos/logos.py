import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from logos.exceptions import UnsupportedOperationError
from logos.exploration.ate_calculator import ATECalculator
from logos.exploration.explorer import Explorer
from logos.exploration.pruner import Pruner
from logos.exploration.types import Edge
from logos.parsing.parser import Parser
from logos.parsing.parser_from_precomputed import ParserFromPrecomputed
from logos.parsing.parser_like import ParserLike
from logos.parsing.tag_utils import name_of, tag_of
from logos.preparation.prepared_variable_name import PreparedVariableName
from logos.preparation.preparer import Preparer
from logos.preparation.preparer_from_precomputed import PreparerFromPrecomputed

_logger = logging.getLogger(__name__)
# Suppress LOGos debug messages by default; call set_verbose_to(True) to enable.
logging.getLogger("src.logos").setLevel(logging.WARNING)


class Logos:
    """
    LOGos provides a high-level interface for causal analysis of event logs.
    """

    DEFAULT_REGEX_DICT = Parser.DEFAULT_REGEX_DICT
    DEFAULT_MESSAGE_PREFIX = Parser.DEFAULT_MESSAGE_PREFIX

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
        self._parser: Optional[ParserLike] = Parser(
            filename, workdir, skip_writeout
        )
        self._preparer: Optional[Preparer] = Preparer(self._parser)
        self._explorer: Optional[Explorer] = None

    @classmethod
    def _create(cls) -> "Logos":
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
        template_col: Optional[str] = None,
        passthrough_cols: Optional[List[str]] = None,
        per_unit_cols: Optional[List[str]] = None,
    ) -> "Logos":
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
            template_col: Column whose distinct values define templates.
                When provided, other columns are split into per-template
                variables (one variable per template that has non-null values
                for that column), mirroring Drain's output structure.
            passthrough_cols: Structural identifier columns kept as single
                global variables rather than being split per template (e.g.
                the causal-unit column).  Only meaningful when `template_col`
                is set.
            per_unit_cols: Per-causal-unit constant columns (outcomes,
                assignments) that should not be split per template but should
                appear as candidate variables in the ranking.  Mechanically
                equivalent to `passthrough_cols`; separate parameter for
                semantic clarity.  Only meaningful when `template_col` is set.
        """
        instance = cls._create()
        all_passthrough = list(passthrough_cols or []) + list(
            per_unit_cols or []
        )
        instance._parser = ParserFromPrecomputed(
            data,
            workdir,
            source_id,
            variable_tags,
            skip_writeout,
            template_col,
            all_passthrough,
        )
        instance._preparer = Preparer(instance._parser)
        return instance

    @classmethod
    def from_prepared_table(
        cls,
        data: pd.DataFrame,
        workdir: str,
        variable_tags: Optional[dict[str, str]] = None,
    ) -> "Logos":
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
        pti = PreparerFromPrecomputed(data, workdir, variable_tags)
        instance._explorer = Explorer(pti)
        instance._explorer._init_eccs()
        return instance

    def set_verbose_to(self, val: bool) -> None:
        """Set LOGos logging verbosity (True = DEBUG, False = WARNING)."""
        level = logging.DEBUG if val else logging.WARNING
        logging.getLogger("src.logos").setLevel(level)
        if self._explorer and self._explorer._eccs:
            self._explorer._eccs.set_verbose_to(val)

    def _require_parser(self) -> Parser:
        """Raise if the instance was not created from a raw log file."""
        if not isinstance(self._parser, Parser):
            raise UnsupportedOperationError(
                "This operation requires a LOGos instance created from a raw "
                "log file.  Use LOGos(filename=...) instead of "
                "LOGos.from_parsed_table() or LOGos.from_prepared_table()."
            )
        return self._parser

    def _require_preparer(self) -> Preparer:
        """Raise if the instance has no prepare stage (EP-3)."""
        if self._preparer is None:
            raise UnsupportedOperationError(
                "This operation is not available for instances created via "
                "LOGos.from_prepared_table()."
            )
        return self._preparer

    def _require_explorer(self) -> Explorer:
        """Raise if the exploration stage has not been initialized yet."""
        if self._explorer is None:
            raise UnsupportedOperationError(
                "This operation requires a prepared LOGos instance. "
                "Call prepare() before using exploration methods."
            )
        return self._explorer

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
        if self._preparer is not None:
            return self._preparer.prepared_log
        return self._require_explorer()._prepared_log

    @property
    def prepared_variables(self) -> pd.DataFrame:
        if self._preparer is not None:
            return self._preparer.prepared_variables
        return self._require_explorer()._prepared_variables

    @property
    def prepared_variable_names(self) -> list[str]:
        if self._preparer is not None:
            return self._preparer.prepared_variable_names
        return self._require_explorer().prepared_variable_names

    @property
    def prepared_variable_tags(self) -> list[str]:
        if self._preparer is not None:
            return self._preparer.prepared_variable_tags
        return self._require_explorer()._prepared_variables["Tag"].tolist()

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
        if self._preparer is not None:
            return self._preparer.num_prepared_variables
        return self._require_explorer().num_prepared_variables

    def parse(
        self,
        regex_dict: dict[str, str] = DEFAULT_REGEX_DICT,
        sim_thresh: float = 0.65,
        depth: int = 5,
        force: bool = False,
        message_prefix: str = DEFAULT_MESSAGE_PREFIX,
        enable_gpt_tagging: bool = False,
    ) -> None:
        """Parse the log file; see Parser.parse() for full documentation."""
        parser = self._require_parser()
        parser.parse(
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
        Parser.include_in_template().
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
        return self._require_preparer().get_causal_unit_info()

    def suggest_causal_unit_defs(
        self,
        min_causal_units: int = 4,
        num_suggestions: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Suggest causal unit definitions based on IUS maximization."""
        return self._require_preparer().suggest_causal_unit_defs(
            min_causal_units, num_suggestions
        )

    def set_causal_unit(
        self,
        var: Optional[str] = None,
        num_units: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Set the variable used to define causal units.

        If `var` is None, runs the IUS maximizer and returns ranked suggestions;
        call again with a chosen `var` to set the unit.
        """
        return self._require_preparer().set_causal_unit(var, num_units)

    def prepare(
        self,
        custom_agg: Optional[dict[str, list[str]]] = None,
        custom_imp: Optional[dict[str, str]] = None,
        count_occurrences: bool = False,
        ignore_uninteresting: bool = True,
        force: bool = False,
        lasso_alpha: float = Pruner.LASSO_DEFAULT_ALPHA,
        lasso_max_iter: int = Pruner.LASSO_DEFAULT_MAX_ITER,
        drop_bad_aggs: bool = True,
        reject_prunable_edges: bool = False,
        default_imp: str = "no_imp",
    ) -> None:
        """
        Prepare the log for causal analysis.

        Parameters:
            custom_agg: Custom aggregation functions per variable.
            custom_imp: Custom imputation functions per variable.
            count_occurrences: Whether to add template occurrence count columns.
                Evaluation-only feature — not part of the primary workflow.
            ignore_uninteresting: Whether to drop uninteresting variables.
            force: Force re-preparation even if cached results exist.
            lasso_alpha: LASSO regularization parameter (used when
                reject_prunable_edges=True).
            lasso_max_iter: Maximum LASSO iterations.
            drop_bad_aggs: Whether to drop uninformative aggregate columns.
            reject_prunable_edges: Whether to pre-reject LASSO-prunable edges.
            default_imp: Imputation applied to any variable not in `custom_imp`.
                Defaults to ``"no_imp"`` (rows with NaN are dropped by dropna).
                Set to ``"zero_imp"`` to impute all uncovered variables with 0.
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
            count_occurrences,
            ignore_uninteresting,
            force,
            drop_bad_aggs,
            default_imp=default_imp,
        ):
            return

        self._explorer = Explorer(preparer)
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

    def inspect(
        self,
        var: str,
        ref_var: Optional[str] = None,
        row_limit: Optional[int] = 10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return information DataFrames about a specific prepared variable."""
        return self._require_explorer().inspect(var, ref_var, row_limit)

    def clear_graph(self, clear_edge_states: bool = True) -> None:
        """Clear the graph and possibly edge states."""
        return self._require_explorer().clear_graph(clear_edge_states)

    @property
    def graph(self) -> nx.DiGraph:
        """The current partial causal graph as a NetworkX DiGraph."""
        return self._require_explorer().graph

    def get_graph_dict(self) -> dict:
        """Return the graph as a JSON-serializable dict (for UI/API consumers)."""
        return self._require_explorer().get_graph_dict()

    def save_graph(self, filename: str) -> None:
        """Save the current graph to a PNG file."""
        return self._require_explorer().save_graph(filename)

    def accept(
        self,
        src: str,
        dst: str,
        also_fix: bool = True,
    ) -> Tuple[float, Optional[str]]:
        """Mark a causal graph edge as accepted."""
        return self._require_explorer().accept(src, dst, also_fix)

    def reject(
        self,
        src: str,
        dst: str,
        also_ban: bool = True,
    ) -> Tuple[float, Optional[str]]:
        """Mark a causal graph edge as rejected."""
        return self._require_explorer().reject(src, dst, also_ban)

    def reject_undecided_incoming(
        self, dst: str, also_ban: bool = True
    ) -> Tuple[float, Optional[str]]:
        """Mark all undecided incoming edges to a variable as rejected."""
        return self._require_explorer().reject_undecided_incoming(dst, also_ban)

    def reject_undecided_outgoing(
        self, src: str, also_ban: bool = True
    ) -> Tuple[float, Optional[str]]:
        """Mark all undecided outgoing edges from a variable as rejected."""
        return self._require_explorer().reject_undecided_outgoing(src, also_ban)

    def reject_all_prunable_edges(
        self,
        also_ban: bool = True,
        lasso_alpha: float = Pruner.LASSO_DEFAULT_ALPHA,
        lasso_max_iter: int = Pruner.LASSO_DEFAULT_MAX_ITER,
    ) -> Tuple[float, Optional[str]]:
        """Reject all edges prunable by LASSO."""
        return self._require_explorer().reject_all_prunable_edges(
            also_ban, lasso_alpha, lasso_max_iter
        )

    @property
    def exploration_score(self) -> float:
        """Exploration score of the current partial causal graph."""
        return self._require_explorer().exploration_score

    def suggest_next_exploration(self) -> Optional[str]:
        """Suggest the variable that should be explored next."""
        return self._require_explorer().suggest_next_exploration()

    def rank_candidate_causes(
        self,
        target: Optional[str] = None,
        ignore: Optional[List[str]] = None,
        prune_candidates: bool = True,
        lasso_alpha: float = Pruner.LASSO_DEFAULT_ALPHA,
        lasso_max_iter: int = Pruner.LASSO_DEFAULT_MAX_ITER,
        autoignore_accepted_descendants: bool = True,
    ) -> pd.DataFrame:
        """Return ranked candidate causes for `target` using the LOGOS method."""
        return self._require_explorer().rank_candidate_causes(
            target,
            ignore,
            prune_candidates,
            lasso_alpha,
            lasso_max_iter,
            autoignore_accepted_descendants,
        )

    def get_causal_graph_refinement_suggestion(
        self,
        treatment: str,
        outcome: str,
    ) -> Optional[Edge]:
        """Suggest the next edge to assess using the LOGOS (ECCS) method."""
        return self._require_explorer().get_causal_graph_refinement_suggestion(
            treatment, outcome
        )

    def get_adjusted_ate(
        self,
        treatment: str,
        outcome: str,
        confounder: Optional[str] = None,
    ) -> float:
        """
        Calculate the adjusted ATE of `treatment` on `outcome` using the
        current partial causal graph for confounding adjustment.

        Parameters:
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.
            confounder: Optional explicit confounder; overrides the graph with
                a three-node treatment→outcome←confounder structure.

        Returns:
            The adjusted ATE.
        """
        preparer_log = self.prepared_log
        preparer_vars = self.prepared_variables
        explorer = self._require_explorer()
        return ATECalculator.get_ate_and_confidence(
            preparer_log,
            preparer_vars,
            treatment,
            outcome,
            confounder,
            # When a confounder is explicit, let ATECalculator build the
            # 3-node graph; otherwise use the current partial explorer graph.
            graph=explorer.graph if confounder is None else None,
            calculate_p_value=False,
            calculate_std_error=False,
        )["ATE"]

    def get_unadjusted_ate(
        self,
        treatment: str,
        outcome: str,
    ) -> float: 
        """
        Calculate the unadjusted ATE of `treatment` on `outcome`.

        Parameters:
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.

        Returns:
            The unadjusted ATE.
        """
        preparer_log = self.prepared_log
        preparer_vars = self.prepared_variables
        return ATECalculator.get_ate_and_confidence(
            preparer_log,
            preparer_vars,
            treatment,
            outcome,
            confounder=None,
            graph=None,  # No adjustment for unadjusted ATE
            calculate_p_value=False,
            calculate_std_error=False,
        )["ATE"]

    # ------------------------------------------------------------------
    # Tag management
    # ------------------------------------------------------------------

    def set_tag(self, var: str, tag: str) -> None:
        """
        Set the human-readable tag for a parsed or prepared variable.

        Resolves `var` against the parsed namespace first, then the prepared
        namespace.  Raises ValueError if `var` is not found in either.
        """
        if self._parser is not None:
            try:
                name_of(self._parser.parsed_variables, var, "parsed")
                self._require_parser().tag_parsed_variable(var, tag)
                return
            except (ValueError, KeyError):
                pass
        if self._preparer is not None:
            try:
                name_of(self._preparer.prepared_variables, var, "prepared")
                self._require_preparer().tag_prepared_variable(var, tag)
                return
            except (ValueError, KeyError):
                pass
        raise ValueError(
            f"Variable '{var}' not found in parsed or prepared namespace."
        )

    def get_tag(self, var: str) -> str:
        """
        Get the human-readable tag for a parsed or prepared variable.

        Resolves `var` against the parsed namespace first, then the prepared
        namespace.  Raises ValueError if `var` is not found in either.
        """
        if self._parser is not None:
            try:
                return (
                    tag_of(self._parser.parsed_variables, var, "parsed") or var
                )
            except (ValueError, KeyError):
                pass
        if self._preparer is not None:
            try:
                return (
                    tag_of(self._preparer.prepared_variables, var, "prepared")
                    or var
                )
            except (ValueError, KeyError):
                pass
        if self._explorer is not None:
            try:
                return (
                    tag_of(self._explorer._prepared_variables, var, "prepared")
                    or var
                )
            except (ValueError, KeyError):
                pass
        raise ValueError(f"Variable '{var}' not found.")
