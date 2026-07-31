import os
import pickle

import pandas as pd


class Cache:
    """
    Serialisation for LOGos artifacts. Writes Parquet/JSON; falls back to legacy 
    .pkl on read.
    """

    @staticmethod
    def dump_dataframe(df: pd.DataFrame, path: str) -> None:
        """Write a tabular DataFrame to Parquet."""
        _ensure_parent(path)
        df.to_parquet(path, index=True)

    @staticmethod
    def load_dataframe(path: str) -> pd.DataFrame:
        """
        Read from Parquet, falling back to a legacy .pkl file if Parquet is 
        absent.
        """
        if os.path.isfile(path):
            return pd.read_parquet(path)
        with open(_pkl_equivalent(path), "rb") as f:
            return pickle.load(f)  # legacy read path only, no new .pkl files

    @staticmethod
    def dump_metadata(df: pd.DataFrame, path: str) -> None:
        """Write a metadata DataFrame (may contain list-typed cells) to JSON."""
        _ensure_parent(path)
        df.to_json(path, orient="records", indent=2)

    @staticmethod
    def load_metadata(path: str) -> pd.DataFrame:
        """
        Read from JSON, falling back to a legacy .pkl file if JSON is absent.
        """
        if os.path.isfile(path):
            return pd.read_json(path, orient="records")
        with open(_pkl_equivalent(path), "rb") as f:
            return pickle.load(f)  # legacy read path only, no new .pkl files

    @staticmethod
    def artifact_exists(path: str) -> bool:
        """
        True if the artifact exists in the new format or as a legacy .pkl file.
        """
        return os.path.isfile(path) or os.path.isfile(_pkl_equivalent(path))


def _pkl_equivalent(path: str) -> str:
    """
    Derive the legacy .pkl path from a new-format path.
    """
    for ext in (".parquet", ".json"):
        if path.endswith(ext):
            return path[: -len(ext)] + ".pkl"
    return path + ".pkl"


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
