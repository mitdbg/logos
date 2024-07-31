import pandas as pd
import networkx as nx
from typing import Optional, Any
import os
import pickle
from .variable_name.prepared_variable_name import PreparedVariableName
from .printer import Printer
from tqdm.auto import tqdm
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from .pickler import Pickler
from .ate_calculator import ATECalculator


class Pruner:
    LASSO_DEFAULT_ALPHA = 0.1
    LASSO_DEFAULT_MAX_ITER = 100000

    """
    A collection of pruning functions for prepared variables,
    used for pruning and candidate suggestion.
    """

    @staticmethod
    def prune_with_lasso(
        data: pd.DataFrame,
        outcome_cols: list[str],
        alpha: float = LASSO_DEFAULT_ALPHA,
        max_iter: int = LASSO_DEFAULT_MAX_ITER,
        top_n: int = 0,
        ignore: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Prune variables using Lasso regression.

        Parameters:
            data: The dataframe containing the data.
            outcome_cols: The names of the target variables.
            alpha: The Lasso regularization parameter.
            max_iter: The maximum number of iterations for Lasso.
            top_n: The number of variables to return. If 0, return all variables.
            ignore: The names of the variables to ignore.

        Returns:
            The names of the variables that Lasso identified as impactful, optionally
            limited to the top `n` variables by absolute coefficient.
        """

        # TODO: do this properly wherever this is called
        outcome_col = outcome_cols[0]

        # Separate the target variable and predictor variables.
        # Optionally, do not consider variables already in the graph.
        y = data[outcome_cols]
        drop_cols = [] if ignore is None else ignore
        to_ignore = outcome_cols
        drop_cols.extend(to_ignore)

        # Do not consider variables with the same base variable as an ignored variable.
        for v in to_ignore:
            vp = PreparedVariableName(v)
            if vp.base_var() != "TemplateId":
                drop_cols.extend([c for c in data.columns if vp.base_var() in c])
        drop_cols = list(set(drop_cols))

        # Iterate until multiple prepared variables with the same base variable are eliminated.
        done = False

        while not done:
            Printer.printv(f"Variables that Lasso will ignore: {drop_cols}")
            X = data.drop(drop_cols, axis=1)
            X_cols = X.columns
            if X.empty:
                return []

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Fit a Lasso model to the data
            lasso = Lasso(alpha=alpha, max_iter=max_iter)
            lasso.fit(X, y)
            Printer.printv(f"Lasso coefficients : {lasso.coef_}")
            Printer.printv(f"Scale: {scaler.scale_}")
            final_coefs = lasso.coef_ / scaler.scale_
            abs_coefs = np.abs(final_coefs)
            Printer.printv(f"Lasso coefficients unscaled: {final_coefs}")

            # Mask for nonzero elements
            nonzero_mask = final_coefs != 0

            # Mask for top n largest elements by absolute value
            # Create an array of False values with the same shape as the coefficients
            top_n_mask = [False] * len(final_coefs)
            for i in np.argsort(abs_coefs)[-top_n:]:
                top_n_mask[i] = True

            # Retrieve columns based on conditions above
            selected_names = list(X_cols[nonzero_mask & top_n_mask])

            # Only keep one aggregate per variable
            d = set()
            done = True
            for var in selected_names:
                base_var = PreparedVariableName(var).base_var()
                if base_var in d:
                    drop_cols.append(var)
                    done = False
                else:
                    d.add(base_var)

        Printer.printv("Lasso identified the following impactful variables:")
        Printer.printv(selected_names)

        return selected_names

    @staticmethod
    def prune_with_triangle(
        data: pd.DataFrame,
        vars: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        work_dir: str,
        top_n: int = 0,
        force: bool = False,
    ) -> list[str]:
        """
        Prune variables using triangle method.

        Parameters:
            data: The dataframe containing the data.
            vars: The dataframe containing the variables.
            treatment_col: The name of the treatment variable.
            outcome_col: The name of the outcome variable.
            work_dir: The directory to store intermediate files in.
            top_n: The number of variables to return. If 0, return all variables.
            force: Whether to force recalculation of the triangle method.

        Returns:
            The names of the variables that triangle method identified as impactful, optionally
            limited to the top `n` variables.
        """

        # Check whether we can use pre-calculated results
        filename = os.path.join(
            work_dir, f"pickles/triangle_dags/{treatment_col}_{outcome_col}.pkl"
        )
        if os.path.isfile(filename) and not force:
            df = pickle.load(open(filename, "rb"))
            print("Found pickled file")
            return list(df.index[:top_n].values)

        Printer.printv("Starting to prune using triangle method")
        max_diffs = {}
        base_ate = ATECalculator.get_ate_and_confidence(
            data, vars, treatment_col, outcome_col, calculate_std_error=False
        )["ATE"]

        for var in tqdm(data.columns, "Processing triangle dags"):
            if var == treatment_col or var == outcome_col:
                continue

            # Construct the graphs to consider
            graphs = []
            # Second cause
            graphs.append(
                nx.DiGraph([(treatment_col, outcome_col), (var, outcome_col)])
            )
            # Confounder
            graphs.append(
                nx.DiGraph(
                    [
                        (treatment_col, outcome_col),
                        (var, treatment_col),
                        (var, outcome_col),
                    ]
                )
            )
            # Mediator with direct path
            graphs.append(
                nx.DiGraph(
                    [
                        (treatment_col, outcome_col),
                        (treatment_col, var),
                        (var, outcome_col),
                    ]
                )
            )
            # Mediator without direct path
            graphs.append(nx.DiGraph([(treatment_col, var), (var, outcome_col)]))

            # Calculate the corrsponding ATEs
            ates = [base_ate]
            for G in graphs:
                try:
                    ates.append(
                        ATECalculator.get_ate_and_confidence(
                            data,
                            vars,
                            treatment_col,
                            outcome_col,
                            graph=G,
                            calculate_std_error=False,
                        )["ATE"]
                    )
                except:
                    pass
            max_diffs[var] = max(ates) - min(ates)
        max_diffs = max_diffs
        df = pd.DataFrame.from_dict(max_diffs, orient="index", columns=["max_diff"])
        df = df.sort_values(by="max_diff", ascending=False)

        Pickler.dump(df, filename)

        return list(df.index[:top_n].values)

