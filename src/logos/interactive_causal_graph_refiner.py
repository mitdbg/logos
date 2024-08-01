from eccs.eccs import ECCS
import enum

import pandas as pd
import networkx as nx

from datetime import datetime
from openai import OpenAI
from typing import Self, Optional

from .edge_state_matrix import Edge
from .regression import Regression
from .tag_utils import TagUtils


class InteractiveCausalGraphRefinerMethod(str, enum.Enum):
    LOGOS = "logos"
    REGRESSION = "regression"
    LANGMODEL = "langmodel"

    @staticmethod
    def from_str(method: str) -> Self:
        if method == InteractiveCausalGraphRefinerMethod.LOGOS.value:
            return InteractiveCausalGraphRefinerMethod.LOGOS
        elif method == InteractiveCausalGraphRefinerMethod.REGRESSION.value:
            return InteractiveCausalGraphRefinerMethod.REGRESSION
        elif method == InteractiveCausalGraphRefinerMethod.LANGMODEL.value:
            return InteractiveCausalGraphRefinerMethod.LANGMODEL
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def methods() -> list[Self]:
        return [method for method in InteractiveCausalGraphRefinerMethod]

    @staticmethod
    def methods_str() -> list[str]:
        return [method.value for method in InteractiveCausalGraphRefinerMethod]


class InteractiveCausalGraphRefiner:

    @staticmethod
    def get_suggestion(
        data: pd.DataFrame,
        method: InteractiveCausalGraphRefinerMethod,
        eccs: Optional[ECCS] = None,
        treatment_name: Optional[str] = None,
        outcome_name: Optional[str] = None,
        graph: Optional[nx.DiGraph] = None,
        model: Optional[str] = None,
        gpt_log_path: Optional[str] = None,
        data_tags: Optional[pd.DataFrame] = None,
    ) -> Edge:
        """
        Get the next edge for which the user should porduce a judgment, in the
        process of refining a causal graph.

        Parameters:
            data: The dataframe containing the data.
            method: The method to use for suggesting the next edge.
            eccs: The ECCS object to use for suggesting the next edge. Only applies
                if `method` is `InteractiveCausalGraphRefinerMethod.LOGOS`.
            treatment_name: The name of the treatment variable. Only applies if
                `method` is `InteractiveCausalGraphRefinerMethod.LOGOS` or
                `InteractiveCausalGraphRefinerMethod.LANGMODEL`.
            outcome_name: The name of the outcome variable. Only applies if
                `method` is `InteractiveCausalGraphRefinerMethod.LOGOS` or
                `InteractiveCausalGraphRefinerMethod.LANGMODEL`.
            graph: The graph to use for suggesting the next edge. Only applies if
                `method` is `InteractiveCausalGraphRefinerMethod.REGRESSION` or
                `InteractiveCausalGraphRefinerMethod.LANGMODEL`.
            model: The model to use for suggesting the next edge. Only applies if
                `method` is not `InteractiveCausalGraphRefinerMethod.LANGMODEL`.
            gpt_log_path: The path to the GPT log file. Only applies if `method` is
                `InteractiveCausalGraphRefinerMethod.LANGMODEL`.
            data_tags: The dataframe containing the data tags. Only applies if `method`
                is `InteractiveCausalGraphRefinerMethod.LANGMODEL`.

        Returns:
            The next edge for which the user should produce a judgment.
        """
        if method == InteractiveCausalGraphRefinerMethod.LOGOS:
            return InteractiveCausalGraphRefiner._get_suggestion_logos(
                eccs, treatment_name, outcome_name
            )
        elif method == InteractiveCausalGraphRefinerMethod.REGRESSION:
            return InteractiveCausalGraphRefiner._get_suggestion_regression(data, graph)
        elif method == InteractiveCausalGraphRefinerMethod.LANGMODEL:
            return InteractiveCausalGraphRefiner._get_suggestion_langmodel(
                data,
                data_tags,
                treatment_name,
                outcome_name,
                graph,
                model,
                gpt_log_path,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

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
        return edge_edits[0].edge if (edge_edits and len(edge_edits) > 0) else None

    most_recent_graph = None
    cache = []

    @classmethod
    def _get_suggestion_regression(cls, data: pd.DataFrame, graph: nx.DiGraph) -> Edge:
        """
        Implement `get_suggestion()` for the `REGRESSION` method.

        Parameters:
            data: The dataframe containing the data.
            graph: The graph to use for suggesting the next edge.
        """
        if graph != cls.most_recent_graph:
            cls.most_recent_graph = graph
            cls.cache = []
        if len(cls.cache) > 0:
            return cls.cache.pop(0)

        l = []

        data, _ = Regression.get_normalized_copy(data)

        for v in graph.nodes:
            for w in set(data.columns) - set(graph.neighbors(v)) - set([v]):
                d = Regression.ols(w, data[w], data[v])
                abs_slope = abs(d["Slope"])
                l.append((Edge((w, v)), abs_slope))

        l.sort(key=lambda x: x[1], reverse=True)
        cls.cache = [row[0] for row in l[1:]]

        return l[0][0]

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
    ) -> Edge:
        """
        Implement `get_suggestion()` for the `LANGMODEL` method.

        Parameters:
            data: The dataframe containing the data.
            treatment_name: The name of the treatment variable.
            outcome_name: The name of the outcome variable.
            graph: The graph to use for suggesting the next edge.
            model: The model to use for suggesting the next edge.
            gpt_log_path: The path to the GPT log file.
            data_tags: The dataframe containing the data tags.
        """
        if graph != cls.most_recent_graph:
            cls.most_recent_graph = graph
            cls.cache = []
        if len(cls.cache) > 0:
            return cls.cache.pop(0)

        client = OpenAI()

        treatment_tag = TagUtils.tag_of(data_tags, treatment_name, "prepared")
        outcome_tag = TagUtils.tag_of(data_tags, outcome_name, "prepared")

        num_samples_per_var = 3

        if gpt_log_path == None:
            gpt_log_path = (
                f"ranker-gpt-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
            )
        with open(gpt_log_path, "w+") as f:

            # Define the messages to send to the model
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for causal reasoning.",
                },
                {
                    "role": "user",
                    "content": f"""Below is a list of variable names and some example distinct values for each. """
                    f"""The lists are not sorted in compatible ways, so that elements in the same position may not correspond to the same entity. """
                    f"""{', '.join([f'{TagUtils.tag_of(data_tags, v, "prepared")}: [{", ".join(str(x) for x in data[v].unique().tolist()[:num_samples_per_var])}]' for v in data.columns])}"""
                    """\n\n"""
                    """I have constructed a partial causal graph over these variables. Here is the list of directed edges: """
                    f"""[{', '.join([f'({TagUtils.tag_of(data_tags, u, "prepared")}, {TagUtils.tag_of(data_tags, v, "prepared")})' for u, v in graph.edges])}]"""
                    """\n\n"""
                    f"""I plan to use this causal graph to calculate the ATE of {treatment_tag} on {outcome_tag}. """
                    """However, I'm not sure of its correctness nor completeness. """
                    """I want you to rank pairs of variables from this collection of variables, based on how important it is for me to either add or remove an edge between them in the graph"""
                    """ for the accuracy of my ATE calculation. """
                    """I understand that you may think this is speculative, but I want you to do your best to come up with such a ranked list ALWAYS. """
                    """I will interpret any results you give me knowing that you may not be sure about them. """
                    """Only return the ranked answers, one per line, preceded by a number and a period. Separate each variable in a pair with a comma. """
                    """Do not return any other text before or after the list.""",
                },
            ]

            reply = (
                client.chat.completions.create(model=model, messages=messages)
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
            f.close()

        # Combat hallucinations
        reply_rows = reply.split("\n")
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
                ranked_edges.append(Edge((left, right)))

        cls.cache = ranked_edges[1:]
        return ranked_edges[0]
