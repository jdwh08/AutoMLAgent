#####################################################
# AutoMLAgent [EDA COLUMN STATS]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Summary statistics for EDA columns."""

#####################################################
### BOARD

#####################################################
### IMPORTS

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
import polars as pl

### OWN MODULES
from automlagent.eda.column.column_utils import column_filter_out_missing

if TYPE_CHECKING:
    import datetime

#####################################################
### SETTINGS


#####################################################
### CODE
@mlflow.trace(name="get_numerical_stats_for_column", span_type="func")
def get_numerical_stats_for_column(
    df: pl.DataFrame, column_name: str
) -> dict[str, float]:
    """Create a dictionary of numerical statistics for a column.

    Args:
        df (pl.DataFrame): DataFrame containing the data
        column_name (str): Name of the column to analyze

    Returns:
        dict[str, float]: Dictionary of numerical statistics for the column

    """
    # Filter out missing values
    df_filtered = column_filter_out_missing(df, column_name)
    stats = {
        "min": df_filtered.select(pl.col(column_name).min()).item(),
        "max": df_filtered.select(pl.col(column_name).max()).item(),
        "mean": df_filtered.select(pl.col(column_name).mean()).item(),
        "median": df_filtered.select(pl.col(column_name).median()).item(),
        # NOTE(jdwh08): "linear" approximates values like NumPy
        "p5": df_filtered.select(
            pl.col(column_name).quantile(0.05, interpolation="linear")
        ).item(),
        "q1": df_filtered.select(
            pl.col(column_name).quantile(0.25, interpolation="linear")
        ).item(),
        "q3": df_filtered.select(
            pl.col(column_name).quantile(0.75, interpolation="linear")
        ).item(),
        "p95": df_filtered.select(
            pl.col(column_name).quantile(0.95, interpolation="linear")
        ).item(),
        "std": df_filtered.select(pl.col(column_name).std()).item(),
        "skewness": df_filtered.select(pl.col(column_name).skew()).item(),
        "kurtosis": df_filtered.select(pl.col(column_name).kurtosis()).item(),
    }
    return stats


@mlflow.trace(name="get_string_stats_for_column", span_type="func")
def get_string_stats_for_column(df: pl.DataFrame, column_name: str) -> dict[str, float]:
    """Create a dictionary of string statistics for a column.

    Args:
        df (pl.DataFrame): DataFrame containing the data
        column_name (str): Name of the column to analyze

    Returns:
        dict[str, float]: Dictionary of string statistics for the column

    """
    # Filter out missing values
    df_filtered = column_filter_out_missing(df, column_name)
    char_len = df_filtered.select(pl.col(column_name).str.len_chars())
    stats = {
        "char_length_mean": char_len.select(pl.col(column_name).mean()).item(),
        "char_length_min": char_len.select(pl.col(column_name).min()).item(),
        "char_length_max": char_len.select(pl.col(column_name).max()).item(),
        "char_length_std": char_len.select(pl.col(column_name).std()).item(),
    }
    return stats


@mlflow.trace(name="get_temporal_stats_for_column", span_type="func")
def get_temporal_stats_for_column(
    df: pl.DataFrame, column_name: str
) -> dict[str, datetime.datetime | datetime.date | datetime.timedelta]:
    """Create a dictionary of temporal statistics for a column.

    Args:
        df (pl.DataFrame): DataFrame containing the data
        column_name (str): Name of the column to analyze

    Returns:
        dict[str, float]: Dictionary of temporal statistics for the column

    """
    stats = {
        "temporal_min": df.select(pl.col(column_name).min()).item(),
        "temporal_max": df.select(pl.col(column_name).max()).item(),
        "temporal_diff": df.select(
            pl.col(column_name).max() - pl.col(column_name).min()
        ).item(),
    }
    return stats
