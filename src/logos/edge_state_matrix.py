import numpy as np
import networkx as nx
import pandas as pd
from typing import Optional, TypeAlias
from .variable_name.prepared_variable_name import PreparedVariableName
from itertools import combinations

Edge: TypeAlias = tuple[str, str]

class EdgeStateMatrix:
    """
    A class for managing an edge state matrix.

    An edge state matrix is square, with the entry (i,j) representing the state
    of the directed edge between nodes i and j. The state of an edge is one of:
         0: The existence of the state is undecided.
        -1: The edge does not exist.
         1: The edge exists.

    Self-edges are not allowed. The presence of an edge implies the absence of
    its inverse.
    """

    def __init__(self, variables: list[str]) -> None:
        """
        Initialize the edge state matrix to the right dimensions and mark self-edges
        as rejected and all other edges as undecided.

        Parameters:
            variables: The variables to initialize the edge state matrix based on. This
                list must include variable NAMES, not tags.
        """

        n = len(variables)
        self._variables = variables
        self._m = np.zeros((n, n))
        for i in range(n):
            self._m[i, i] = -1

    @property
    def m(self) -> np.ndarray:
        """
        Returns the edge state matrix.
        """
        return self._m

    @property
    def n(self) -> int:
        """
        Returns the number of nodes.
        """
        return self._m.shape[0]

    def clear_and_set_from_graph(self, graph: nx.DiGraph) -> None:
        """
        Clear the edge state matrix and then set it based on the provided graph.
        In particular, mark all edges in the graph as accepted and all others as rejected.

        Parameters:
            graph: The graph to use to set the edge states.
        """

        self._m = np.zeros((self.n, self.n))
        for edge in graph.edges:
            print("Marking edge as accepted: ", edge)
            self._m[self.idx(edge[0]), self.idx(edge[1])] = 1

        self._m[self._m == 0] = -1

    def clear_and_set_from_matrix(self, m: np.ndarray) -> None:
        """
        Clear the edge state matrix and then set it based on the provided matrix.

        Parameters:
            m: The matrix to use to set the edge states.
        """

        self._m = m

    def idx(self, var: str) -> int:
        """
        Retrieve the index of a variable in the edge state matrix.

        Parameters:
            var: The name or tag of the variable.

        Returns:
            The index of the variable in the edge state matrix.
        """
        return self._variables.index(var)

    def get_edge_state(self, src: str, dst: str) -> str:
        """
        Get the state of a specific edge.

        Parameters:
            src: The name or tag of the source variable.
            dst: The name or tag of the destination variable.

        Returns:
            The state of the edge (Accepted, Rejected, or Undecided).
        """
        src_idx = self.idx(src)
        dst_idx = self.idx(dst)
        return self.edge_state_to_str(self._m[src_idx][dst_idx])

    def edge_state_to_str(self, state: int) -> str:
        """
        Translate between edge value and its interpretation.

        Parameters:
            state: The state of the edge represented as an integer.

        Returns:
            The state of the edge (Accepted, Rejected, or Undecided).
        """
        if state == 0:
            return "Undecided"
        elif state == -1:
            return "Rejected"
        elif state == 1:
            return "Accepted"
        else:
            raise ValueError(f"Invalid edge state {state}")

    def mark_edge(self, src: str, dst: str, state: str) -> list[str]:
        """
        Mark an edge as being in a specified state.

        Parameters:
            src: The name or tag of the source variable.
            dst: The name or tag of the destination variable.
            state: The state to mark the edge with (Accepted, Rejected, or Undecided).

        Returns:
            A list of variables that were removed from the partial causal graph as a result
            of this edge being marked as Accepted.

        Throws:
            ValueError: If `state` is not one of "Accepted", "Rejected", or "Undecided".
        """

        src_idx = self.idx(src)
        dst_idx = self.idx(dst)

        if state == "Accepted":
            self._m[src_idx][dst_idx] = 1
            self._m[dst_idx][src_idx] = -1
            return self._reject_other_variants(src, dst)
        elif state == "Rejected":
            self._m[src_idx][dst_idx] = -1
            return []
        elif state == "Undecided":
            self._m[src_idx][dst_idx] = 0
            return []
        else:
            raise ValueError(f"Invalid edge state {state}")

    def _reject_other_variants(self, src: str, dst: str) -> list[str]:
        """
        Mark any edges that touch a variable different from `src` and `dst`, but sharing
        the same base variable as `src` or `dst`, as rejected. Also remove any such variables
        from the partial causal graph.

        Parameters:
            src: The name or tag of the source variable.
            dst: The name or tag of the destination variable.

        Returns:
            A list of variables that were removed from the partial causal graph as a result
            of this edge being marked as Accepted.
        """

        src_base = PreparedVariableName(src).base_var()
        dst_base = PreparedVariableName(dst).base_var()

        l = []
        for var in self._variables:
            var_base = PreparedVariableName(var).base_var()
            if (var_base == src_base and var != src) or (
                var_base == dst_base and var != dst
            ):
                self._m[self.idx(var), :] = -1
                self._m[:, self.idx(var)] = -1
                l.append(var)

        return l 

    @staticmethod
    def enumerate_with_max_edges(n: int, max_edges: int) -> list[np.ndarray]:
        """
        Enumerate all edge state matrices of dimension `n` with at most `max_edges` accepted edges.

        Parameters:
            n: The dimension of the edge state matrices.
            max_edges: The maximum number of edges to allow in the edge state matrices.

        Returns:
            A list of edge state matrices.
        """
        valid_matrices = {0: [np.full(shape=(n, n), fill_value=-1)]}

        # Enumerate all valid matrices with k edges
        for k in range(1, max_edges + 1):
            valid_matrices[k] = []

            # For each valid matrix with k-1 edges...
            for m in valid_matrices[k - 1]:
                # ...add a new edge in every possible way
                for i in range(n):
                    for j in range(i + 1, n):
                        if m[i, j] < 0 and m[j, i] < 0:
                            forward = m.copy()
                            forward[i, j] = 1
                            valid_matrices[k].append(forward)
                            backward = m.copy()
                            backward[j, i] = 1
                            valid_matrices[k].append(backward)

        # Flatten the collection of matrices into a single list
        returned_matrices = []
        for k in range(1, max_edges + 1):
            returned_matrices.extend(valid_matrices[k])

        return returned_matrices
