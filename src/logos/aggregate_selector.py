import pandas as pd
import numpy as np
from .aggimp.agg_funcs import *
from .variable_name.prepared_variable_name import PreparedVariableName


class AggregateSelector:
    DEFAULT_AGGREGATES = {
        "num": [
            "mean",
            "max",
            "min",
        ],
        "str": [
            "last",
            "mode",
            "first",
        ],
    }

    def _entropy(col: pd.Series) -> float:
        """
        Calculates the entropy of a column.

        Parameters:
            col: The column for which to calculate the entropy.

        Returns:
            The entropy of `col`.
        """

        rel_value_counts = col.value_counts(normalize=True)
        if rel_value_counts.empty:
            return 0
        return -np.sum(rel_value_counts * np.log2(rel_value_counts))

    def find_uninformative_aggregates(
        prepared_log: pd.DataFrame, parsed_variables: pd.DataFrame, causal_unit_var: str
    ) -> list[str]:
        """
        Find aggregates that are uninformative for each column in `prepared_log`.
        Aggregates are uninformative unless they maximize the empirical entropy across causal units.

        Parameters:
            prepared_log: The prepared log.
            parsed_variables: The parsed variables.
            causal_unit_var: The name of the causal unit variable.

        Returns:
            A list of uninformative aggregates for `prepared_log`.
        """

        drop_list = []

        for row in parsed_variables.itertuples():
            aggs = row.Aggregates
            if len(aggs) == 0 or row.Name == causal_unit_var:
                continue

            vars = [f"{row.Name}+{agg}" for agg in aggs]
            best_var = f"{row.Name}+{AggregateSelector.DEFAULT_AGGREGATES[row.Type][0]}"
            max_entropy = -np.inf

            for var in vars:
                entropy = AggregateSelector._entropy(prepared_log[var])

                if entropy > max_entropy:
                    best_var = var
                    max_entropy = entropy

            drop_list.extend([var for var in vars if var != best_var])

        return drop_list
