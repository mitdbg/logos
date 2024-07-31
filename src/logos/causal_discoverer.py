import pandas as pd
import numpy as np
import networkx as nx
from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
from pgmpy.estimators import K2Score
from pgmpy.base import DAG
from openai import OpenAI
from tqdm.auto import tqdm
from datetime import datetime
from typing import Optional, Tuple
from .tag_utils import TagUtils


class CausalDiscoverer:
    """
    Provides various methods for automatic causal discovery based on a dataframe.

    Within LOGos, the expectation is that the passed dataframe will contain the prepared variables.
    """

    @staticmethod
    def _pgmpy_dag_to_digraph(dag: DAG) -> nx.DiGraph:
        """
        Converts a pgmpy DAG to a networkx DiGraph.

        Parameters:
            dag: The pgmpy DAG.

        Returns:
            The networkx DiGraph.
        """

        return nx.DiGraph(dag.edges())

    @staticmethod
    def pc(df: pd.DataFrame, max_cond_vars: int = 3) -> nx.DiGraph:
        """
        Runs the PC algorithm on a dataframe.

        Parameters:
            df: The dataframe on which to run the PC algorithm.
            max_cond_vars: The maximum number of conditioning variables to use.

        Returns:
            The causal graph learned by the PC algorithm.
        """

        pc = PC(data=df)
        model = pc.estimate(variant="parallel", max_cond_vars=max_cond_vars)
        return CausalDiscoverer._pgmpy_dag_to_digraph(model)

    @staticmethod
    def hill_climb(df: pd.DataFrame) -> nx.DiGraph:
        """
        Runs the hill climb algorithm on a dataframe.

        Parameters:
            df: The dataframe on which to run the hill climb algorithm.

        Returns:
            The causal graph learned by the hill climb algorithm.
        """

        scoring_method = K2Score(data=df)
        hcs = HillClimbSearch(data=df)
        model = hcs.estimate(
            scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
        )
        return CausalDiscoverer._pgmpy_dag_to_digraph(model)

    @staticmethod
    def exhaustive(df: pd.DataFrame) -> nx.DiGraph:
        """
        Runs the exhaustive search algorithm on a dataframe.

        Parameters:
            df: The dataframe on which to run the exhaustive search algorithm.

        Returns:
            The causal graph learned by the exhaustive search algorithm.
        """

        scoring_method = K2Score(data=df)
        exh = ExhaustiveSearch(data=df, complete_samples_only=False)
        model = exh.estimate()
        return CausalDiscoverer._pgmpy_dag_to_digraph(model)

    @staticmethod
    def gpt(
        data_df: pd.DataFrame,
        model: str = "gpt-3.5-turbo",
        vars_df: Optional[pd.DataFrame] = None,
    ) -> nx.DiGraph:
        """
        Consults GPT to determine the causal graph of the variables in the dataframe.

        Parameters:
            data_df: The dataframe based on which to construct a causal graph.
            model: The GPT model to use.
            vars_df: The dataframe containing the variable names and tags.

        Returns:
            The causal graph learned by consulting GPT.
        """

        # Open a file for logging, with the model and the timestamp in the name
        log_file = open(
            f"/../../evaluation/gpt-logs/{model}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt",
            "w",
        )

        client = OpenAI()
        graph = nx.DiGraph()

        for i in tqdm(
            range(len(data_df.columns)), desc="Outer edge-finding loop using GPT..."
        ):
            for j in range(i + 1, len(data_df.columns)):
                var_a = data_df.columns[i]
                var_b = data_df.columns[j]

                example_rows = data_df[[var_a, var_b]].dropna().sample(3)
                examples_a = ", ".join(str(x) for x in example_rows[var_a].tolist())
                examples_b = ", ".join(str(x) for x in example_rows[var_b].tolist())

                tag_a = (
                    var_a
                    if vars_df is None
                    else TagUtils.get_tag(vars_df, var_a, "prepared")
                )
                tag_b = (
                    var_b
                    if vars_df is None
                    else TagUtils.get_tag(vars_df, var_b, "prepared")
                )

                # Define the messages to send to the model
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for causal reasoning.",
                    },
                    {
                        "role": "user",
                        "content": f"""Which cause-and-effect relationship is more likely? """
                        f"""A. changing {tag_a} causes a change in {tag_b}. """
                        f"""B. changing {tag_b} causes a change in {tag_a}. """
                        f"""C. Neither of the two. """
                        f""" Here are some example values of {tag_a} : [{examples_a}]"""
                        f""" Here are the corresponding values of {tag_b} : [{examples_b}]"""
                        """Let's work this out in a step by step way to be sure that we have the right answer. """
                        """Then provide your ﬁnal answer within the tags <Answer>A/B/C</Answer>.""",
                    },
                ]

                reply = (
                    client.chat.completions.create(model=model, messages=messages)
                    .choices[0]
                    .message.content
                )

                # Log the messages and the reply
                log_file.write(f"{datetime.now()}\n")
                log_file.write("Messages:\n")
                for message in messages:
                    log_file.write(f"{message['role']}: {message['content']}\n")
                log_file.write("----------------\n")
                log_file.write(f"Reply: {reply}\n\n")
                log_file.write("================\n")
                log_file.flush()

                # Find the part of the reply that contains the answer
                start_idx = reply.find("<Answer>") + len("<Answer>")
                end_idx = reply.find("</Answer>")
                answer = reply[start_idx:end_idx]

                # Add the edge to the graph
                if answer == "A":
                    graph.add_edge(var_a, var_b)
                elif answer == "B":
                    graph.add_edge(var_b, var_a)
        log_file.close()
        return graph


        
