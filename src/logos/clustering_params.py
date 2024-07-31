from typing import Optional


class ClusteringParams:
    """
    A class to conveniently hold all the parameters required by the clustering
    approach to challenging the ATE.
    """

    def __init__(
        self,
        top_n: int = 10,
        num_edges: int = 3,
        ignore_ts: bool = True,
        var_pruning_method: Optional[str] = None,
        triangle_n: int = 6,
        force: bool = False,
        force_triangle: bool = False,
        num_clusters: Optional[int] = None,
        threshold: float = 0,
    ) -> None:
        """
        Initializes a ClusteringParams object.

        Parameters:
            top_n: The number of top edges to identify.
            num_edges: The maximum number of edges to use when enumerating DAGs.
            ignore_ts: Whether to ignore timestamp variables.
            var_pruning_method: The pruning method to use. Can be either "lasso" or "triangle".
            triangle_n: The number of variables to use for the triangle method.
            force: Whether to force recalculation.
            force_triangle: Whether to force the triangle method to be recalculated, if selected.
            num_clusters: The number of clusters to use. If None, will try to find the optimal number.
            threshold: The threshold to use when finding outlier edges.

        """
        self.top_n = top_n
        self.num_edges = num_edges
        self.ignore_ts = ignore_ts
        self.var_pruning_method = var_pruning_method
        self.triangle_n = triangle_n
        self.force = force
        self.force_triangle = force_triangle
        self.num_clusters = num_clusters
        self.threshold = threshold
