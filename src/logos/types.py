from typing import Callable
from collections import defaultdict
from typing import Self

class Types:
    Edge = tuple[str, str]
    """Type alias for a directed edge."""

    LeafLabelingFunction = Callable[[int], str]
    """Type alias for a leaf labeling function in `ATE`."""

    EdgeCountDict = defaultdict[Edge, int]
    """Type alias for a dictionary counting edge occurrences."""
