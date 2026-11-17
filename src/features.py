"""Domain-driven feature engineering for tabular competition data."""
import pandas as pd


def add_aggregates(df: pd.DataFrame, group: str, cols: list[str]) -> pd.DataFrame:
    aggs = df.groupby(group)[cols].agg(["mean", "std", "max"]).reset_index()
    aggs.columns = [f"{group}_{c}_{s}" for c, s in aggs.columns[1:]]
    return df.merge(aggs, on=group, how="left")


def clip_outliers(df: pd.DataFrame, cols: list[str], z: float = 4.0) -> pd.DataFrame:
    for c in cols:
        lo = df[c].mean() - z * df[c].std()
        hi = df[c].mean() + z * df[c].std()
        df[c] = df[c].clip(lo, hi)
    return df
