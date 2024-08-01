from collections import defaultdict
import numpy as np
from typing import Optional, Any, Self
import networkx as nx
from .types import *
from .printer import Printer


class EdgeOccurrenceTree:
    """
    A tree of DAGs based on the ATE cluster they belong to.
    """

    def __init__(self, cluster_id: Optional[str] = None) -> None:
        """
        Initialize a tree node with a specific cluster id.

        Parameters:
            cluster_id: The cluster id of the DAGs that belong to this node.
        """

        self.cluster_id = cluster_id
        self.left = None
        self.right = None

    @staticmethod
    def build_tree(linked: np.ndarray, leaves: list[int]) -> tuple[Self, int]:
        """
        Build a tree from a linkage matrix.

        Parameters:
            linked: The linkage matrix.
            leaves: The list of leaf nodes.

        Returns:
            A tuple containing the root of the tree, and the index of the next cluster to be merged.
        """

        # Base case: if there is only one cluster, return it as a leaf.
        if len(leaves) == 1:
            return EdgeOccurrenceTree(cluster_id=leaves[0]), -1

        # Otherwise, build the tree recursively.
        root = EdgeOccurrenceTree()
        curr = root
        i = len(linked) - 1
        while i > -1:
            # Linked contains 4 elements: cluster1, cluster2, distance, num_observations
            # They represent the clusters that were merged, the distance between them, and
            # the number of observations in the new cluster.
            c1, c2, _, _ = linked[i]

            if c1 not in leaves and c2 not in leaves:
                curr.left, i = EdgeOccurrenceTree.build_tree(linked[:i], leaves)
                curr.right, i = EdgeOccurrenceTree.build_tree(linked[:i], leaves)
                break
            if c1 in leaves:
                curr.left = EdgeOccurrenceTree(leaves.index(c1))
                curr.right = EdgeOccurrenceTree()
                curr = curr.right
            if c2 in leaves:
                curr.right = EdgeOccurrenceTree(leaves.index(c2))
                break
            i -= 1
        root = EdgeOccurrenceTree._cleanup_tree(root)
        return root, i

    @staticmethod
    def _cleanup_tree(root: Optional[Self]) -> Self:
        """
        Clean up the tree by removing nodes that have only one child, and nodes that have no
        children and are not leaves.

        Parameters:
            root: The root of the tree.

        Returns:
            The root of the cleaned up tree.
        """

        if root is None:
            return None

        # Recursively clean up left and right subtrees
        root.left = EdgeOccurrenceTree._cleanup_tree(root.left)
        root.right = EdgeOccurrenceTree._cleanup_tree(root.right)

        # If the current node has only one child, replace the node with its child
        if root.left is None and root.right is not None:
            return root.right
        elif root.left is not None and root.right is None:
            return root.left

        # If the current node has no left and right child and is not a leaf, remove the node
        if root.left is None and root.right is None and root.cluster_id is None:
            return None

        return root

    def print_tree(self, depth: int = 0) -> None:
        """
        Print the tree in a readable format.

        Parameters:
            depth: The depth of the current node in the tree.
        """

        prefix = ""
        for _ in range(depth):
            prefix += "-"
        if self.cluster_id is not None:
            Printer.printv(prefix + str(self.cluster_id))
        else:
            Printer.printv(prefix + "node")
        if self.left:
            self.left.print_tree(depth + 1)
        if self.right:
            self.right.print_tree(depth + 1)

    def assign_dags_to_nodes(self, cluster_mapping: dict[nx.DiGraph, int]) -> None:
        """
        Assign each DAG to the node it belongs to, based on `cluster_mapping`.

        Parameters:
            cluster_mapping: A dictionary mapping DAGs to cluster id's.
        """
        self.num_dags = 0

        # If leaf, assign DAGs and set count.
        if self.cluster_id is not None:
            self.dags = [
                key
                for key in cluster_mapping.keys()
                if cluster_mapping[key] == self.cluster_id
            ]
            self.num_dags = len(self.dags)

        # Otherwise, recurse for children and retireve counts.
        if self.left:
            self.left.assign_dags_to_nodes(cluster_mapping)
            self.num_dags += self.left.num_dags
        if self.right:
            self.right.assign_dags_to_nodes(cluster_mapping)
            self.num_dags += self.right.num_dags

    def count_edge_occurrences(
        self, treatment: str, outcome: str, dag: nx.DiGraph
    ) -> None:
        """
        Recursively count the number of times each edge occurs amongst the DAGs
        assigned to all the children of this node, omitting the edge from treatment -> outcome,
        since this always exists. If a DAG is passed in, ignore the edges in that DAG as well.

        Parameters:
            treatment: The treatment variable.
            outcome: The outcome variable.
            dag: The optional dag structure to ignore.
        """
        self.edge_counts: Types.EdgeCountDict = defaultdict(int)

        # If leaf, actually compute count.
        if self.cluster_id is not None:
            edges_to_ignore = [(treatment, outcome)]
            if dag:
                edges_to_ignore.extend(dag.edges)
            for graph in self.dags:
                for edge in graph.edges:
                    if edge not in edges_to_ignore:
                        self.edge_counts[edge] += 1

        # Otherwise, derive counts from children.
        if self.left:
            self.left.count_edge_occurrences(treatment, outcome, dag)
            for key in self.left.edge_counts.keys():
                self.edge_counts[key] += self.left.edge_counts[key]
        if self.right:
            self.right.count_edge_occurrences(treatment, outcome, dag)
            for key in self.right.edge_counts.keys():
                self.edge_counts[key] += self.right.edge_counts[key]

        # Compute statistics.
        freq_counts = list(self.edge_counts.values())
        if len(freq_counts) == 0:
            self.mean = None
            self.std_dev = None
        else:
            self.mean = np.mean(freq_counts)
            self.std_dev = np.std(freq_counts)

    def calculate_edge_expectancy(
        self, totals: tuple[int, Types.EdgeCountDict] = None
    ) -> None:
        """
        For each edge at each node, calculate what percent over or under
        expectancy the edge is at in relationship to its parent.

        Parameters:
            totals: A tuple containing the total number of DAGs and the mapping from
                edges to their counts for the parent of this node.
        """
        # At root node, calculate expectancy
        if totals is None:
            totals = (self.num_dags, self.edge_counts)

        # Otherwise, calculate expectancy based on parent.
        total_dags, total_edges = totals
        self.percent_expectancy = defaultdict(float)

        for edge in self.edge_counts.keys():
            expected = self.num_dags / total_dags * total_edges[edge]
            self.percent_expectancy[edge] = (
                self.edge_counts[edge] - expected
            ) / expected

        # Recurse for children.
        if self.left:
            self.left.calculate_edge_expectancy((self.num_dags, self.edge_counts))
        if self.right:
            self.right.calculate_edge_expectancy((self.num_dags, self.edge_counts))

    def find_outliers_in_tree(self, threshold: float = 0) -> None:
        """
        Find outlier edges, based on the percent expectancy of each edge. Define an outlier as an
        edge that is below expectancy on one side of the tree, and above on the other side, and
        optionally, over some threshold on both sides.

        Parameters:
            threshold: The threshold for an edge to be considered an outlier.
        """

        # If able to compare, find outliers.
        if self.left and self.right:
            self.left.outliers = {}
            self.right.outliers = {}
            edges = set(self.left.edge_counts.keys()).union(
                set(self.right.edge_counts.keys())
            )
            for edge in edges:
                if (
                    np.sign(self.left.percent_expectancy[edge])
                    != np.sign(self.right.percent_expectancy[edge])
                    and abs(self.left.percent_expectancy[edge]) > threshold
                    and abs(self.right.percent_expectancy[edge]) > threshold
                ):
                    self.left.outliers[edge] = self.left.percent_expectancy[edge]
                    self.right.outliers[edge] = self.right.percent_expectancy[edge]

        # Recurse for children.
        if self.left:
            self.left.find_outliers_in_tree(threshold)
        if self.right:
            self.right.find_outliers_in_tree(threshold)

    def find_outliers_per_cluster(
        self,
        dag: nx.DiGraph,
    ) -> tuple[Types.EdgeCountDict, dict[Types.Edge, float]]:
        """
        Collect the edge counts and outliers found earlier into appropriate dictionaries
        per cluster.

        Parameters:
            dag: The DAG to ignore when collecting outliers.

        Returns:
            A tuple containing the following: a dictionary mapping cluster id's to edge counts,
            and a dictionary mapping cluster id's to outlier edges.
        """

        cluster_edge_count = {}
        cluster_outliers = {}

        # If leaf, add to cluster counts.
        if self.cluster_id is not None:
            cluster_edge_count[self.cluster_id] = self.edge_counts
            edges_to_ignore = dag.edges if dag is not None else []
            cluster_outliers[self.cluster_id] = {
                edge: self.outliers[edge]
                for edge in self.outliers
                if edge not in edges_to_ignore
            }

        # Otherwise, recurse for children.
        if self.left:
            lec, lo = self.left.find_outliers_per_cluster(dag)
            cluster_edge_count.update(lec)
            cluster_outliers.update(lo)
        if self.right:
            rec, ro = self.right.find_outliers_per_cluster(dag)
            cluster_edge_count.update(rec)
            cluster_outliers.update(ro)

        return cluster_edge_count, cluster_outliers
