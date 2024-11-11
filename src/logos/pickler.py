import os
import pickle

import pandas as pd


class Pickler:
    """
    A class for loading and dumping dataframes to and from pkl files.
    """

    @staticmethod
    def load(filename: str) -> pd.DataFrame:
        """
        Loads a dataframe from a pkl file.

        Parameters:
            filename: The name of the pkl file.

        Returns:
            The dataframe loaded from the pkl file.
        """
        df = pd.DataFrame()
        with open(filename, "rb") as f:
            df = pickle.load(f)
        return df

    @staticmethod
    def dump(df: pd.DataFrame, filename: str) -> None:
        """
        Dumps a dataframe to a pkl file.

        Parameters:
            df: The dataframe to be dumped.
            filename: The name of the pkl file.
        """

        if "/" in filename:
            path = filename[: filename.rindex("/")]
            os.makedirs(path, exist_ok=True)

        with open(filename, "wb+") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
