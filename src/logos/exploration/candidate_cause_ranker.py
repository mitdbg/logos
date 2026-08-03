"""
Functionality for ranking candidate causes.
"""

import enum
import logging
import multiprocessing
from typing import Optional, Tuple, cast

import pandas as pd

from logos.llm import get_openai_client
from logos.exploration.pruner import Pruner
from logos.exploration.regression import multi_ols, ols
from logos.parsing.tag_utils import name_of, tag_of

_logger = logging.getLogger(__name__)


class CandidateCauseRanker:
    """
    A class to rank candidate causes for a target variable.
    """

    COLUMN_ORDER = [
        "Candidate",
        "Candidate Tag",
        "Target Tag",
        "Slope",
        "P-value",
        "Candidate->Target Edge Status",
        "Target->Candidate Edge Status",
    ]

    INTERNAL_COLUMN_ORDER = [
        "Candidate",
        "Candidate Tag",
        "Target Tag",
        "Slope",
        "P-value",
    ]

    @staticmethod
    def rank(
        data: pd.DataFrame,
        data_tags_df: pd.DataFrame,
        target_name: str,
        ignore: Optional[list[str]] = None,
        prune_candidates: bool = True,
        lasso_alpha: float = Pruner.LASSO_DEFAULT_ALPHA,
        lasso_max_iter: int = Pruner.LASSO_DEFAULT_MAX_ITER,
    ) -> Tuple[pd.DataFrame, list[str]]:
        """Rank candidate causes for `target_name` using the LOGOS method.

        Parameters:
            data: The data based on which to explore candidate causes.
            data_tags_df: A dataframe containing tags for the data.
            target_name: The name of the target variable.
            ignore: A list of variables to ignore.
            prune_candidates: Whether to prune candidates by Lasso regression.
            lasso_alpha: The Lasso regularization parameter.
            lasso_max_iter: The maximum number of Lasso iterations.

        Returns:
            results_df: A dataframe of ranked candidate causes.
            pruned: A list of pruned (Lasso-rejected) variables.
        """
        if ignore is None:
            ignore = []
        non_ignore = [col for col in data.columns if col not in ignore]
        return CandidateCauseRanker._rank_logos(
            data[non_ignore],
            data_tags_df,
            target_name,
            prune_candidates,
            lasso_alpha,
            lasso_max_iter,
        )

    @staticmethod
    def _rank_logos(
        data: pd.DataFrame,
        data_tags_df: pd.DataFrame,
        target_name: str,
        prune_candidates: bool = True,
        lasso_alpha: float = Pruner.LASSO_DEFAULT_ALPHA,
        lasso_max_iter: int = Pruner.LASSO_DEFAULT_MAX_ITER,
    ) -> Tuple[pd.DataFrame, list[str]]:
        """
        Implement `rank()` for the LOGOS method.

        Parameters:
            data: The data based on which to explore candidate causes.
            data_tags_df: A dataframe containing tags for the data.
            target_name: The name of the target variable.
            prune_candidates: Whether to prune the candidate causes using Lasso regression.
            lasso_alpha: The alpha parameter to be used for Lasso regression.
            lasso_max_iter: The maximum number of iterations to be used for Lasso regression.

        Returns:
            results_df: A dataframe containing the candidate causal graph neighbors for `target`
            pruned: A list of pruned candidate causes, if any.
        """

        # Use Lasso to get a pruned list of neighbors
        candidates = (
            Pruner.prune_with_lasso(
                data,
                [target_name],
                alpha=lasso_alpha,
                max_iter=lasso_max_iter,
            )
            if prune_candidates
            else [c for c in data.columns if c != target_name]
        )
        _logger.debug(f"Candidates: {candidates}")
        pruned = list(set(data.columns) - set(candidates) - set([target_name]))

        # Stop if there are no candidates
        if len(candidates) == 0:
            print("No candidates found.")
            return (
                pd.DataFrame(columns=CandidateCauseRanker.COLUMN_ORDER),
                pruned,
            )

        # For each candidate, calculate the slope and p-value of
        # a linear regression with target (in parallel)
        num_processors = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processors) as pool:
            results = pool.starmap(
                ols,
                [(col, data[col], data[target_name]) for col in candidates],
            )

        # Prepare return values.
        result_df = (
            pd.DataFrame(results)
            .sort_values(by="P-value", ascending=True)
            .reset_index(drop=True)
        )
        result_df["Target Tag"] = tag_of(data_tags_df, target_name, "prepared")
        result_df["Candidate Tag"] = result_df["Candidate"].apply(
            lambda x: tag_of(data_tags_df, x, "prepared")
        )
        result_df = result_df[CandidateCauseRanker.INTERNAL_COLUMN_ORDER]

        return result_df, pruned

    @staticmethod
    def rank_regression(
        data: pd.DataFrame,
        data_tags_df: pd.DataFrame = None,
        target_name: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, list[str]]:
        """
        Implement `rank()` for the REGRESSION method.

        Parameters:
            data: The data based on which to explore candidate causes.
            data_tags_df: A dataframe containing tags for the data.
            target_name: The name of the target variable.

        Returns:
            results_df: A dataframe containing the candidate causal graph neighbors for `target`
            pruned: A list of pruned candidate causes, if any. #TODO: Prune based on cutoff?
        """

        candidates = [c for c in data.columns if c != target_name]
        result_df = multi_ols(candidates, data[candidates], data[target_name])
        result_df = (
            result_df.sort_values(
                by="Absolute Normalized Slope", ascending=False
            )
            .drop(columns=["Normalized Slope", "Absolute Normalized Slope"])
            .reset_index(drop=True)
        )

        result_df["Target Tag"] = tag_of(data_tags_df, target_name, "prepared")
        result_df["Candidate Tag"] = result_df["Candidate"].apply(
            lambda x: tag_of(data_tags_df, x, "prepared")
        )
        result_df = result_df[CandidateCauseRanker.INTERNAL_COLUMN_ORDER]

        return result_df, []

    @staticmethod
    def rank_langmodel(
        data: pd.DataFrame,
        data_tags_df: pd.DataFrame = None,
        target_name: Optional[str] = None,
        model: str = "gpt-4o-mini-2024-07-18",
        gpt_log_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, list[str]]:
        """
        Implement `rank()` for the LANGMODEL method.

        Parameters:
            data: The data based on which to explore candidate causes.
            data_tags_df: A dataframe containing tags for the data.
            target_name: The name of the target variable.
            model: The model to use for the langmodel method.
            gpt_log_path: The path to the log file for the prompt and reply.

        Returns:
            results_df: A dataframe containing the candidate causal graph neighbors for `target`
            pruned: A list of pruned candidate causes, if any.
        """

        client = get_openai_client()

        target_tag = tag_of(data_tags_df, target_name, "prepared")
        nspv = 3  # Number of sample values to show per variable

        if gpt_log_path is None:
            gpt_log_path = (
                f"ranker-gpt-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
            )
        with open(cast(str, gpt_log_path), "w+", encoding="utf-8") as f:

            # Define the messages to send to the model
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for causal reasoning.",
                },
                {
                    "role": "user",
                    "content": f"""Below is a list of variable names and some example distinct """
                    f"""values for each. The lists are not sorted in compatible ways, so that """
                    f"""elements in the same position may not correspond to the same entity. """
                    f"""I want you to assess the likelihood of each of these variables as a """
                    f"""cause for variable '{target_tag}' and return them as a ranked list. """
                    """I understand that you may think this is speculative, but I want you to do """
                    """your best to come up with such a list ALWAYS. I will interpret any """
                    """results you give me knowing that you may not be sure about them. """
                    """Only return the ranked answers, one per line, preceded by a number and a """
                    """period. Rank as many of the given variables as you see fit, except """
                    f"""'{target_tag}' itself. Do not return any other text before or after the """
                    """list. Here are the variables: """
                    ", ".join(
                        [
                            f"""{tag_of(data_tags_df, v, "prepared")}: """
                            f"""[{", ".join(str(x) for x in data[v].unique().tolist()[:nspv])}]"""
                            for v in data.columns
                        ]
                    ),
                },
            ]

            reply = cast(
                str,
                client.chat.completions.create(model=model, messages=messages)  # type: ignore
                .choices[0]
                .message.content,
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
        possibly_candidate_tags = [
            ".".join(row.split(".")[1:]).split(":", maxsplit=1)[0].strip()
            for row in reply_rows
        ]
        candidate_tags = [
            tag
            for tag in possibly_candidate_tags
            if tag in data_tags_df["Tag"].values
        ]

        d = {
            "Candidate Tag": candidate_tags,
            "Slope": [None for _ in range(len(candidate_tags))],
            "P-value": [None for _ in range(len(candidate_tags))],
        }
        result_df = pd.DataFrame(d)
        result_df["Target Tag"] = tag_of(data_tags_df, target_name, "prepared")
        result_df["Candidate"] = result_df["Candidate Tag"].apply(
            lambda x: name_of(data_tags_df, x.split(":")[0], "prepared")
        )
        result_df = result_df[CandidateCauseRanker.INTERNAL_COLUMN_ORDER]

        pruned = list(
            set(data.columns) - set(result_df["Candidate"]) - set([target_name])
        )

        return result_df, pruned
