"""
A module for the interactive causal graph refiner functionality.
"""

from datetime import datetime
from typing import Optional, cast

import networkx as nx
import pandas as pd
from eccs.eccs import ECCS

from logos.llm import get_openai_client
from logos.regression import get_normalized_copy, ols
from logos.tag_utils import name_of, tag_of
from logos.types import Edge


class GraphRefiner:

    @staticmethod
    def get_suggestion(
        eccs: ECCS,
        treatment_name: str,
        outcome_name: str,
    ) -> Optional[Edge]:
        """Suggest the next edge to assess using the LOGOS (ECCS) method.

        Parameters:
            eccs: The ECCS object.
            treatment_name: The name of the treatment variable.
            outcome_name: The name of the outcome variable.

        Returns:
            The next edge for which the user should produce a judgment, or None.
        """
        return GraphRefiner._get_suggestion_logos(
            eccs, treatment_name, outcome_name
        )

    @staticmethod
    def _get_suggestion_logos(
        eccs: ECCS, treatment_name: str, outcome_name: str
    ) -> Edge:
        """
        Implement `get_suggestion()` for the `LOGOS` method.

        Parameters:
            eccs: The ECCS object to use for suggesting the next edge.
            treatment_name: The name of the treatment variable.
            outcome_name: The name of the outcome variable.

        Returns:
            The next edge for which the user should produce a judgment.
        """
        eccs.set_treatment(treatment_name)
        eccs.set_outcome(outcome_name)
        edge_edits, _, _ = eccs.suggest_best_single_adjustment_set_change(
            max_results=1, use_optimized=True
        )
        return (
            edge_edits[0].edge if (edge_edits and len(edge_edits) > 0) else None
        )

    most_recent_graph = None
    cache: list[Edge] = []

    @classmethod
    def get_suggestion_regression(
        cls,
        data: pd.DataFrame,
        graph: nx.DiGraph,
        data_tags: Optional[pd.DataFrame] = None,
    ) -> Optional[Edge]:
        """For evaluation: suggest next edge using OLS regression.

        Recomputes a full pairwise ranking on every call.  Pass `data_tags` to
        receive the edge expressed as human-readable tags; omit it for raw
        variable names.
        """
        if graph != cls.most_recent_graph:
            cls.most_recent_graph = graph
            cls.cache = []
        if cls.cache:
            edge = cls.cache.pop(0)
        else:
            pairs: list[tuple[Edge, float]] = []
            data_norm, _ = get_normalized_copy(data)
            for v in graph.nodes:
                for w in set(data_norm.columns) - set(graph.neighbors(v)) - {v}:
                    d = ols(w, data_norm[w], data_norm[v])
                    slope = d["Slope"]
                    abs_slope = abs(slope) if slope is not None else 0.0
                    pairs.append((cast(Edge, (w, v)), abs_slope))
            if not pairs:
                return None
            pairs.sort(key=lambda x: x[1], reverse=True)
            cls.cache = [row[0] for row in pairs[1:]]
            edge = pairs[0][0]

        if data_tags is not None:
            return (
                cast(str, tag_of(data_tags, edge[0], "prepared")),
                cast(str, tag_of(data_tags, edge[1], "prepared")),
            )
        return edge

    @classmethod
    def get_suggestion_langmodel(
        cls,
        data: pd.DataFrame,
        data_tags: pd.DataFrame,
        treatment_name: str,
        outcome_name: str,
        graph: nx.DiGraph,
        model: str = "gpt-4o-mini-2024-07-18",
        gpt_log_path: Optional[str] = None,
        return_tags: bool = True,
    ) -> Optional[Edge]:
        """For evaluation: suggest next edge using an LLM.

        Set `return_tags=False` to receive raw variable names instead of tags.
        """
        return GraphRefiner._get_suggestion_langmodel(
            data,
            data_tags,
            treatment_name,
            outcome_name,
            graph,
            model,
            gpt_log_path,
            return_tags=return_tags,
        )

    @classmethod
    def _get_suggestion_langmodel(
        cls,
        data: pd.DataFrame,
        data_tags: pd.DataFrame,
        treatment_name: str,
        outcome_name: str,
        graph: nx.DiGraph,
        model: str = "gpt-4o-mini-2024-07-18",
        gpt_log_path: Optional[str] = None,
        return_tags: bool = True,
    ) -> Optional[Edge]:
        """Implementation of the LANGMODEL suggestion method."""
        if graph != cls.most_recent_graph:
            cls.most_recent_graph = graph
            cls.cache = []
        if cls.cache:
            return cls.cache.pop(0)

        client = get_openai_client()

        treatment_tag = tag_of(data_tags, treatment_name, "prepared")
        outcome_tag = tag_of(data_tags, outcome_name, "prepared")

        num_samples_per_var = 3

        if gpt_log_path is None:
            gpt_log_path = (
                f"ranker-gpt-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
            )

        # Prepare some substrings for the prompt
        def tag_func(x: str) -> Optional[str]:
            return tag_of(data_tags, x, "prepared")

        vars_to_examples = {
            v: data[v].unique().tolist()[:num_samples_per_var]
            for v in data.columns
        }
        vars_and_examples_s = ", ".join(
            [
                f'{tag_func(v)}: [{", ".join(str(x) for x in vars_to_examples[v])}]'
                for v in data.columns
            ]
        )
        directed_edges_s = ", ".join(
            [f"({tag_func(u)}, {tag_func(v)})" for u, v in graph.edges]
        )

        with open(gpt_log_path, "w+", encoding="utf-8") as f:
            # Define the messages to send to the model
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for causal reasoning.",
                },
                {
                    "role": "user",
                    "content": """Below is a list of variable names and some example distinct """
                    """values for each. The lists are not sorted in compatible ways, so that """
                    """elements in the same position may not correspond to the same entity. """
                    f"""{vars_and_examples_s}\n\n"""
                    """I have constructed a partial causal graph over these variables. Here is """
                    f"""the list of directed edges: [{directed_edges_s}]\n\n"""
                    f"""I plan to use this causal graph to calculate the ATE of {treatment_tag} """
                    f"""on {outcome_tag}. However, I'm not sure of its correctness nor """
                    """completeness. I want you to rank pairs of variables from this collection """
                    """of variables, based on how important it is for me to either add or remove """
                    """an edge between them in the graph for the accuracy of my ATE calculation. """
                    """I understand that you may think this is speculative, but I want you to do """
                    """your best to come up with such a ranked list ALWAYS. I will interpret any """
                    """results you give me knowing that you may not be sure about them. Only """
                    """return the ranked answers, one per line, preceded by a number and a """
                    """period. Separate each variable in a pair with a comma. Do not return any """
                    """other text before or after the list.""",
                },
            ]

            reply = (
                client.chat.completions.create(model=model, messages=messages)  # type: ignore
                .choices[0]
                .message.content
            )

            # Log the messages and the reply
            f.write(f"{datetime.now()}\n")
            f.write("Messages:\n")
            for message in messages:
                f.write(f"{message['role']}: {message['content']}\n")
            f.write("----------------\n")
            f.write(f"Reply: {reply}\n\n")
            f.write("================\n")
            f.flush()

        # Combat hallucinations
        reply_rows = cast(str, reply).split("\n")
        reply_rows = [
            row for row in reply_rows if row.strip() != "" and row[0].isdigit()
        ]
        possibly_ranked_edges = [
            [v.strip() for v in ".".join(row.split(".")[1:]).strip().split(",")]
            for row in reply_rows
        ]
        ranked_edges = []
        tags = data_tags["Tag"].values
        for edge in possibly_ranked_edges:
            if len(edge) != 2:
                continue

            left = None
            right = None

            if edge[0] in tags:
                left = edge[0]
            elif f"{edge[0]} mean" in tags:
                left = f"{edge[0]} mean"

            if edge[1] in tags:
                right = edge[1]
            elif f"{edge[1]} mean" in tags:
                right = f"{edge[1]} mean"

            if left is not None and right is not None:
                ranked_edges.append(cast(Edge, (left, right)))

        if not ranked_edges:
            return None
        cls.cache = list(ranked_edges[1:])
        edge = ranked_edges[0]
        if not return_tags:
            return (
                cast(str, name_of(data_tags, edge[0], "prepared")),
                cast(str, name_of(data_tags, edge[1], "prepared")),
            )
        return edge
