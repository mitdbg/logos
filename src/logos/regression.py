import pandas as pd
import statsmodels.api as sm


class Regression:
    """
    A collection of regression-related functions.
    """

    @staticmethod
    def ols(X_name: str, X_data: pd.Series, Y_data: pd.Series) -> dict:
        """
        Calculate the slope and p-value of a linear regression of `X` on `Y`.

        Parameters:
            X_name: The name of the predictor variable.
            X_data: The data for the predictor variable.
            Y_data: The data for the target variable.

        Returns:
            A dictionary containing the slope and p-value of the regression. If
            there is no slope parameter because X_data does not vary enough,
            the slope and p-value will be None.
        """
        X_data = sm.add_constant(X_data)
        model = sm.OLS(Y_data, X_data).fit()
        slope = None
        p_value = None
        if len(model.params) > 1:
            slope = model.params.iloc[1]
            p_value = model.pvalues.iloc[1]
        return {
            "Candidate": X_name,
            "Slope": slope,
            "P-value": p_value,
        }

    @staticmethod
    def get_normalized_copy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return a normalized copy of the input data, with zero mean
        and unit variance.

        Parameters:
            data: The data to normalize.

        Returns:
            A normalized copy of the input data.
            The original standard deviations of the columns of the input data.
        """
        data = data.copy(deep=True)
        stdevs = data.std()

        # Cast all columns to float64 to avoid numpy warnings
        data = data.astype("float64")

        for column in data.columns:
            if stdevs[column] == 0:
                data.loc[:, column] = 0
            else:
                data.loc[:, column] = (data[column] - data[column].mean()) / stdevs[
                    column
                ]
        return data, stdevs

    @staticmethod
    def multi_ols(
        X_names: list[str], X_data: pd.DataFrame, Y_data: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate the slopes and p-values of a multivariate linear regression
        of the variables in `X` on `Y`. Normalize each column to zero mean and
        unit variance before running the regression. Return both the normalized
        and unnormalized slopes.

        Parameters:
            X_names: The names of the predictor variables.
            X_data: The data for the predictor variables.
            Y_data: The data for the target variable.

        Returns:
            A dataframe with the names, slopes, and p-values of the regressions.
        """
        X_data, stdevs = Regression.get_normalized_copy(X_data)

        X_data = sm.add_constant(X_data)
        model = sm.OLS(Y_data, X_data).fit()

        # Get the coefficients and p-values, ignoring the constant
        coefficients = model.params.iloc[1:]
        p_values = model.pvalues.iloc[1:]

        # Unnormalize the slopes
        coefficients_unnormalized = coefficients.copy()
        for coeff in coefficients_unnormalized.index:
            coefficients_unnormalized[coeff] = (
                coefficients[coeff] / stdevs[coeff] if stdevs[coeff] != 0 else 0
            )

        return pd.DataFrame(
            {
                "Candidate": coefficients.index,
                "Slope": coefficients_unnormalized.values,
                "P-value": p_values.values,
                "Normalized Slope": coefficients.values,
                "Absolute Normalized Slope": coefficients.abs().values,
            }
        )
