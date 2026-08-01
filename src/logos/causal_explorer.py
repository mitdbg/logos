"""
Causal graph exploration: edge management, candidate ranking, ATE calculation.
"""

import logging
import multiprocessing
from datetime import datetime
from typing import List, Optional, Tuple, cast

import networkx as nx
import numpy as np
import pandas as pd
from eccs.eccs import ECCS
from tqdm.auto import tqdm

from src.logos.candidate_cause_ranker import CandidateCauseRanker
from src.logos.edge_state_matrix import EdgeStateMatrix
from src.logos.graph_renderer import GraphRenderer
from src.logos.interactive_causal_graph_refiner import (
    InteractiveCausalGraphRefiner,
)
from src.logos.prepared_source import PreparedSource
from src.logos.pruner import Pruner
from src.logos.tag_utils import TagUtils
from src.logos.types import Types
from src.logos.variable_name.prepared_variable_name import PreparedVariableName

_logger = logging.getLogger(__name__)


class CausalExplorer:
    """
    Owns graph-state and all exploration operations after data preparation.

    Call _init_eccs() once after any initial edge pruning is complete.
    """

    def __init__(self, source: PreparedSource) -> None:
        self._prepared_log = source.prepared_log
        self._prepared_variables = source.prepared_variables
        self._parsed_variables = source.parsed_variables
        self._parsed_templates = source.parsed_templates
        self._workdir = source.workdir

        self._graph: nx.DiGraph = nx.DiGraph()
        # ECCS is None until _init_eccs() is called after optional pruning.
        self._eccs: Optional[ECCS] = None
        self._next_exploration: Optional[str] = None
        self._edge_states: EdgeStateMatrix = EdgeStateMatrix(
            self.prepared_variable_names
        )

    def _init_eccs(self) -> None:
        """Initialise ECCS from the current prepared log. Call after pruning."""
        self._eccs = ECCS(self._prepared_log, nx.DiGraph())
        self._eccs.set_verbose_to(_logger.isEnabledFor(logging.DEBUG))

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def prepared_variable_names(self) -> list[str]:
        return self._prepared_variables["Name"].values.tolist()

    @property
    def num_prepared_variables(self) -> int:
        return len(self._prepared_variables)

    @property
    def graph(self) -> nx.DiGraph:
        """The current partial causal graph."""
        return self._graph

    def get_graph_dict(self) -> dict:
        """Return the graph as a JSON-serializable dict for UI/API consumers."""
        vars_df = self._prepared_variables
        nodes = [
            {
                "id": n,
                "tag": (
                    vars_df.loc[vars_df["Name"] == n, "Tag"].values[0]
                    if n in vars_df["Name"].values
                    else n
                ),
            }
            for n in self._graph.nodes
        ]
        edges = [{"src": u, "dst": v} for u, v in self._graph.edges]
        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Graph management
    # ------------------------------------------------------------------

    def inspect(
        self,
        var: str,
        ref_var: Optional[str] = None,
        row_limit: Optional[int] = 10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return information DataFrames about a specific prepared variable.

        Parameters:
            var: The name or tag of the variable.
            ref_var: The name or tag of a reference variable.
            row_limit: The number of rows of the prepared log to include.

        Returns:
            A tuple of:
                (1) Base-variable metadata from parsed_variables (empty if from regex).
                (2) Template metadata from parsed_templates (empty if from regex).
                (3) A sample of the prepared log with `row_limit` rows.
        """
        name = TagUtils.name_of(self._prepared_variables, var, "prepared")
        base_var = PreparedVariableName(name).base_var()
        from_regex = False

        base_var_info_df = pd.DataFrame()
        if base_var != "TemplateId":
            base_var_info_df = self._parsed_variables[
                self._parsed_variables["Name"] == base_var
            ]
            from_regex = (
                base_var_info_df["From regex"].values[0]
                if not base_var_info_df.empty
                else True
            )

        template_info_df = pd.DataFrame()
        if not from_regex:
            template_id = PreparedVariableName(name).template_id()
            template_info_df = self._parsed_templates[
                self._parsed_templates["TemplateId"] == template_id
            ]

        if row_limit is None:
            row_limit = len(self._prepared_log)
        col_list = [name] + ([ref_var] if ref_var is not None else [])
        prepared_log_info_df = (
            self._prepared_log[col_list].head(row_limit).copy()
        )
        col_names = [f"{name} (candidate)"] + (
            [f"{ref_var} (outcome)"] if ref_var is not None else []
        )
        prepared_log_info_df.columns = col_names

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

    def save_graph(self, filename: str) -> None:
        """
        Save the current graph to a PNG file.

        Parameters:
            filename: Destination file path.
        """
        GraphRenderer.save_graph(
            self._graph, self._prepared_variables, filename
        )

    def accept(
        self,
        src: str,
        dst: str,
        also_fix: bool = True,  # instructs ECCS to lock this edge permanently
    ) -> Tuple[float, Optional[str]]:
        """
        Mark a causal graph edge as accepted.

        This will also reject the edge from `dst` to `src` and remove any other
        variables with the same base variable as either `src` or `dst` from
        consideration for the partial causal graph.

        Parameters:
            src: The name or tag of the source variable.
            dst: The name or tag of the destination variable.
            also_fix: Whether to permanently lock this edge in ECCS so it is
                never proposed for removal. Advanced parameter — ECCS internal.

        Returns:
            A tuple of (exploration_score, suggested_next_variable).
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
        if self._eccs:
            self._eccs.remove_edge(dst_name, src_name)
            self._eccs.add_edge(src_name, dst_name)
            if also_fix:
                self._eccs.fix_edge(src_name, dst_name)

        return (self.exploration_score, self.suggest_next_exploration())

    def reject(
        self,
        src: str,
        dst: str,
        also_ban: bool = True,  # instructs ECCS to never propose this edge again
    ) -> Tuple[float, Optional[str]]:
        """
        Mark a causal graph edge as rejected.

        Parameters:
            src: The name or tag of the source variable.
            dst: The name or tag of the destination variable.
            also_ban: Whether to permanently ban this edge in ECCS so it is
                never proposed again. Advanced parameter — ECCS internal.

        Returns:
            A tuple of (exploration_score, suggested_next_variable).
        """
        src_name = TagUtils.name_of(self._prepared_variables, src, "prepared")
        dst_name = TagUtils.name_of(self._prepared_variables, dst, "prepared")
        self._edge_states.mark_edge(src_name, dst_name, "Rejected")
        if self._eccs and also_ban:
            self._eccs.ban_edge(src_name, dst_name)

        return (self.exploration_score, self.suggest_next_exploration())

    def reject_undecided_incoming(
        self, dst: str, also_ban: bool = True
    ) -> Tuple[float, Optional[str]]:
        """
        Mark all undecided incoming edges to a variable as rejected.

        Parameters:
            dst: The name or tag of the destination variable.
            also_ban: Whether to permanently ban these edges in ECCS.
                Advanced parameter — ECCS internal.

        Returns:
            A tuple of (exploration_score, suggested_next_variable).
        """
        dst_name = TagUtils.name_of(self._prepared_variables, dst, "prepared")
        for v in self.prepared_variable_names:
            if self._edge_states.get_edge_state(v, dst_name) == "Undecided":
                self._edge_states.mark_edge(v, dst_name, "Rejected")
                if self._eccs and also_ban:
                    self._eccs.ban_edge(v, dst_name)

        return (self.exploration_score, self.suggest_next_exploration())

    def reject_undecided_outgoing(
        self, src: str, also_ban: bool = True
    ) -> Tuple[float, Optional[str]]:
        """
        Mark all undecided outgoing edges from a variable as rejected.

        Parameters:
            src: The name or tag of the source variable.
            also_ban: Whether to permanently ban these edges in ECCS.
                Advanced parameter — ECCS internal.

        Returns:
            A tuple of (exploration_score, suggested_next_variable).
        """
        src_name = TagUtils.name_of(self._prepared_variables, src, "prepared")
        for v in self.prepared_variable_names:
            if self._edge_states.get_edge_state(src_name, v) == "Undecided":
                self._edge_states.mark_edge(src_name, v, "Rejected")
                if self._eccs and also_ban:
                    self._eccs.ban_edge(src_name, v)

        return (self.exploration_score, self.suggest_next_exploration())

    def reject_all_prunable_edges(
        self,
        also_ban: bool = True,  # instructs ECCS to never propose these edges again
        lasso_alpha: float = Pruner.LASSO_DEFAULT_ALPHA,
        lasso_max_iter: int = Pruner.LASSO_DEFAULT_MAX_ITER,
    ) -> Tuple[float, Optional[str]]:
        """
        Reject all incoming edges that LASSO identifies as non-predictive.

        Parameters:
            also_ban: Whether to permanently ban rejected edges in ECCS.
                Advanced parameter — ECCS internal.
            lasso_alpha: LASSO regularization parameter.
            lasso_max_iter: Maximum LASSO iterations.

        Returns:
            A tuple of (exploration_score, suggested_next_variable).
        """
        num_processors = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processors) as pool:
            all_candidates = pool.starmap(
                Pruner.prune_with_lasso,
                tqdm(
                    [
                        (
                            self._prepared_log,
                            [target],
                            lasso_alpha,
                            lasso_max_iter,
                        )
                        for target in self.prepared_variable_names
                    ],
                    total=self.num_prepared_variables,
                    desc="Finding pruned variables...",
                ),
            )

        _logger.debug(all_candidates)

        for candidates, target in zip(
            all_candidates, self.prepared_variable_names
        ):
            non_candidates = (
                set(self._prepared_log.columns) - set(candidates) - {target}
            )
            for nc in non_candidates:
                self._edge_states.mark_edge(nc, target, "Rejected")
                if self._eccs and also_ban:
                    self._eccs.ban_edge(nc, target)

        return (self.exploration_score, self.suggest_next_exploration())

    @property
    def exploration_score(self) -> float:
        """
        Calculate the exploration score of the current partial causal graph,
        based on the edge state matrix.

        Returns:
            The exploration score of the current partial causal graph.
        """
        M = self._graph.number_of_nodes()
        N = self.num_prepared_variables
        incident = M * (2 * N - M - 1)
        if incident == 0:
            return 0

        graph_var_indices = [
            self._edge_states.idx(x) for x in list(self._graph.nodes)
        ]
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

        _logger.debug(f"Considered: {considered}")
        _logger.debug(f"Incident: {incident}")

        return considered / incident

    def suggest_next_exploration(self) -> Optional[str]:
        """
        Suggest the variable that should be explored next. Suggest the prepared
        variable in the partial causal graph that has the most (nonzero)
        Unexplored incoming edges, if any; otherwise suggest the prepared
        variable with the most (nonzero) Undecided incoming edges, even if it is
        not in the partial causal graph.

        If all edges are decided, return None.

        Returns:
            The name of the variable to explore next.
        """

        node_names = list(self._graph.nodes)
        graph_var_indices = [self._edge_states.idx(x) for x in node_names]
        graph_var_incoming_edge_states = self._edge_states.m[
            :, graph_var_indices
        ]
        undecided_edges_per_col = (
            np.sum(graph_var_incoming_edge_states == 0, axis=0)
            if len(graph_var_incoming_edge_states) > 0
            else []
        )
        max_undecided = (
            np.max(undecided_edges_per_col)
            if len(undecided_edges_per_col) > 0
            else 0
        )

        if max_undecided > 0:
            max_undecided_idx = np.argmax(undecided_edges_per_col)
            self._next_exploration = node_names[max_undecided_idx]
            return self._next_exploration

        # If no suggestion was found, try to find a suggestion from the entire
        # collection of prepared variables.
        undecided_edges_per_col = np.sum(self._edge_states.m == 0, axis=0)
        max_undecided = np.max(undecided_edges_per_col)

        if max_undecided > 0:
            max_undecided_idx = np.argmax(undecided_edges_per_col)
            self._next_exploration = self._prepared_variables.loc[
                max_undecided_idx, "Name"
            ]
            return self._next_exploration

        self._next_exploration = None
        return None

    def rank_candidate_causes(
        self,
        target: Optional[str] = None,
        ignore: Optional[List[str]] = None,
        prune_candidates: bool = True,
        lasso_alpha: float = Pruner.LASSO_DEFAULT_ALPHA,
        lasso_max_iter: int = Pruner.LASSO_DEFAULT_MAX_ITER,
    ) -> pd.DataFrame:
        """
        Return ranked candidate causes for `target` using the LOGOS method.

        Parameters:
            target: The name or tag of the target variable. If None, uses the
                variable most recently suggested by suggest_next_exploration().
            ignore: Variables to exclude from consideration.
            prune_candidates: Whether to apply LASSO pruning before ranking.
            lasso_alpha: LASSO regularization parameter.
            lasso_max_iter: Maximum LASSO iterations.

        Returns:
            A DataFrame of ranked candidates with columns: Candidate,
            Candidate Tag, Target Tag, Slope, P-value,
            Candidate->Target Edge Status, Target->Candidate Edge Status.
        """
        start_time = datetime.now()

        if target is None and self._next_exploration is None:
            _logger.debug("No target specified.")
            return pd.DataFrame(columns=CandidateCauseRanker.COLUMN_ORDER)
        elif target is None:
            target = self._next_exploration
        assert target is not None

        target = TagUtils.name_of(self._prepared_variables, target, "prepared")

        lfn = f"ranker-gpt-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
        result_df, pruned = CandidateCauseRanker.rank(
            self._prepared_log,
            self._prepared_variables,
            target,
            ignore,
            prune_candidates,
            lasso_alpha,
            lasso_max_iter,
        )

        for var in pruned:
            self._edge_states.mark_edge(var, target, "Rejected")

        result_df["Candidate->Target Edge Status"] = result_df[
            "Candidate"
        ].apply(lambda x: self._edge_states.get_edge_state(x, target))
        result_df["Target->Candidate Edge Status"] = result_df[
            "Candidate"
        ].apply(lambda x: self._edge_states.get_edge_state(target, x))

        elapsed = (datetime.now() - start_time).total_seconds()
        _logger.debug(f"Candidate cause ranking complete in {elapsed:.6f}s")
        return result_df[CandidateCauseRanker.COLUMN_ORDER]

    def get_causal_graph_refinement_suggestion(
        self,
        treatment: str,
        outcome: str,
    ) -> Optional[Types.Edge]:
        """
        Suggest the next edge to assess using the LOGOS (ECCS) method.

        Parameters:
            treatment: The name or tag of the treatment variable.
            outcome: The name or tag of the outcome variable.

        Returns:
            The suggested edge as a tuple of human-readable tags, or None.
        """
        start_time = datetime.now()
        assert (
            self._eccs is not None
        ), "ECCS not initialized. Call _init_eccs() before refinement."
        treatment_name = TagUtils.name_of(
            self._prepared_variables, treatment, "prepared"
        )
        outcome_name = TagUtils.name_of(
            self._prepared_variables, outcome, "prepared"
        )

        edge = InteractiveCausalGraphRefiner.get_suggestion(
            self._eccs, treatment_name, outcome_name
        )

        edge_tags: Optional[tuple[str, str]] = None
        if edge:
            edge_tags = (
                cast(
                    str,
                    TagUtils.tag_of(
                        self._prepared_variables, edge[0], "prepared"
                    ),
                ),
                cast(
                    str,
                    TagUtils.tag_of(
                        self._prepared_variables, edge[1], "prepared"
                    ),
                ),
            )

        elapsed = (datetime.now() - start_time).total_seconds()
        _logger.debug(f"Graph refinement suggestion complete in {elapsed:.6f}s")
        return edge_tags
