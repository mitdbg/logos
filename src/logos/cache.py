import os
import pickle

import pandas as pd


def dump_dataframe(df: pd.DataFrame, path: str) -> None:
    """Write a tabular DataFrame to Parquet."""
    _ensure_parent(path)
    df.to_parquet(path, index=True)


def load_dataframe(path: str) -> pd.DataFrame:
    """
    Read from Parquet, falling back to a legacy .pkl file if Parquet is
    absent.
    """
    if os.path.isfile(path):
        return pd.read_parquet(path)
    for fallback in (_pkl_equivalent(path), _legacy_none_none_pkl(path)):
        if os.path.isfile(fallback):
            with open(fallback, "rb") as f:
                return pickle.load(f)  # noqa: S301 — legacy read path only
    raise FileNotFoundError(f"No cache file found for {path}")


def dump_metadata(df: pd.DataFrame, path: str) -> None:
    """Write a metadata DataFrame (may contain list-typed cells) to JSON."""
    _ensure_parent(path)
    df.to_json(path, orient="records", indent=2)


def load_metadata(path: str) -> pd.DataFrame:
    """
    Read from JSON, falling back to a legacy .pkl file if JSON is absent.
    """
    if os.path.isfile(path):
        return pd.read_json(path, orient="records")
    for fallback in (_pkl_equivalent(path), _legacy_none_none_pkl(path)):
        if os.path.isfile(fallback):
            with open(fallback, "rb") as f:
                return pickle.load(f)  # noqa: S301 — legacy read path only
    raise FileNotFoundError(f"No cache file found for {path}")


def artifact_exists(path: str) -> bool:
    """True if the artifact exists in new format or as any legacy .pkl variant."""
    return (
        os.path.isfile(path)
        or os.path.isfile(_pkl_equivalent(path))
        or os.path.isfile(_legacy_none_none_pkl(path))
    )


def _legacy_none_none_pkl(path: str) -> str:
    """
    Older parse artifact path (had _None_None suffix before extension).
    """
    for ext in (".parquet", ".json"):
        if path.endswith(ext):
            return path[: -len(ext)] + "_None_None.pkl"
    return path + "_None_None.pkl"


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
