import pandas as pd
from typing import Optional


class CausalUnitSuggester:
    """
    This class is responsible for suggesting causal units to the user.
    """

    @staticmethod
    def _discretize(col: pd.Series, col_type: str, bins: int = 0) -> pd.Series:
        """
        Discretize an unsorted `col` based on its type. If `col_type` is 'num', then
        return labels for each of `bins` equi-depth bins. If `col_type` is 'str,
        then return a unique label for each unique value. Nulls in `col` are assigned
        to bin -1.

        Parameters:
            col: The column to discretize.
            col_type: The type of the column.
            bins: The number of bins to use when discretizing the column.

        Returns:
            A vector of length len(`col`) with the labels of each value in `col`.
        """
        if col_type == "num":
            return (
                pd.qcut(col, bins, labels=False, duplicates="drop")
                .fillna(-1)
                .astype(int)
            )
        elif col_type == "str":
            return pd.factorize(col, use_na_sentinel=True)[0]
        else:
            raise ValueError(f"Unknown column type: {col_type}")

    @staticmethod
    def _get_all_discretizations(
        col: pd.Series, col_type: str, k: int
    ) -> list[pd.Series]:
        """
        Return a list of all possible discretizations of `col` based on its type.
        If `col_type` is 'num', then return discretizations with `k`, `2k` and `10k` bins.
        If `col_type` is 'str', then return a discretization with a unique label for
        each unique value in `col`.

        Parameters:
            col: The column to discretize.
            col_type: The type of the column.
            k: A parameter indirectly controlling the number of bins to use when discretizing
                a numeric column (see above).

        Returns:
            A list of all desired discretizations of `col`.
        """

        if col_type == "num":
            l = []
            if len(col) >= k:
                l.append(CausalUnitSuggester._discretize(col, col_type, k))
            if len(col) >= 2 * k:
                l.append(CausalUnitSuggester._discretize(col, col_type, 2 * k))
            if len(col) >= 10 * k:
                l.append(CausalUnitSuggester._discretize(col, col_type, 10 * k))
            return l
        elif col_type == "str":
            return [CausalUnitSuggester._discretize(col, col_type)]
        else:
            raise ValueError(f"Unknown column type: {col_type}")

    @staticmethod
    def _calculate_IUS(df: pd.DataFrame, discretization: pd.Series) -> float:
        """
        Calculate the Information Utilization Score of `df` if each row belongs
        to the causal unit specified by `discretization`. The unit labelled -1
        contails rows with null value for the causal unit column, so the corresponding
        rows in `df` are ignored.

        Parameters:
            df: The DataFrame to calculate the Information Utilization Score of.
            discretization: The causal unit of each row.

        Returns:
            The Information Utilization Score of `df`.
        """

        grouped = df.groupby(discretization)  # TODO: handle nulls
        ius = 0

        for group_id, group_data in grouped:
            if group_id == -1:
                continue
            columns_with_non_nulls = group_data.notna().any(axis=0).sum()
            ius += columns_with_non_nulls * len(group_data)

        return ius / (len(df.columns) * len(df))

    @staticmethod
    def suggest_causal_unit_defs(
        data_df: pd.DataFrame,
        var_df: pd.DataFrame,
        min_causal_units: int = 4,
        num_suggestions: int = 10,
    ) -> Optional[pd.DataFrame]:
        """
        Suggest at most `num_suggestions` causal unit definitions for `data_df` based on ius
        maximization, while returning at least `min_causal_units` causal units. `var_df` provides
        information on the type of each variable.

        Parameters:
            data_df: The DataFrame to suggest causal unit definitions for.
            var_df: A DataFrame with one row for each variable in `data_df` that includes variable type information.
            min_causal_units: The minimum number of causal units that a suggested definition should create.
            num_suggestions: The maximum number of causal unit definitions to suggest.

        Returns:
            A DataFrame with one row for each suggested causal unit definition, or `None` if no suggestions were made.
        """

        list_of_suggestions = []

        for col in data_df.columns:
            discretizations = CausalUnitSuggester._get_all_discretizations(
                data_df[col],
                var_df[var_df["Name"] == col]["Type"].values[0],
                k=min_causal_units,
            )
            for disc in discretizations:
                # Ensure that the unique values in disc, excluding -1 if it exists, are at least min_causal_units
                if disc.max() >= (min_causal_units - 1):
                    list_of_suggestions.append(
                        {
                            "Variable": col,
                            "Type": var_df[var_df["Name"] == col]["Type"].values[0],
                            "Num Units": disc.max() + 1,
                            "IUS": CausalUnitSuggester._calculate_IUS(data_df, disc),
                        }
                    )

        df_of_suggestions = pd.DataFrame(list_of_suggestions)
        if len(df_of_suggestions) == 0:
            return None
        return df_of_suggestions.sort_values(by=["IUS"], ascending=False).head(
            num_suggestions
        )
