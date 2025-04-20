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

import datetime

import mlflow
import polars as pl

### OWN MODULES
from automlagent.eda.column.column_utils import (
    MAX_CATEGORIES_FOR_LEVEL,
    column_filter_out_missing,
)
from automlagent.logger.mlflow_logger import get_mlflow_logger

#####################################################
### SETTINGS


#####################################################
### CODE
@mlflow.trace(name="get_histogram_bins_for_column", span_type="func")
def get_histogram_bins_for_column(
    df: pl.DataFrame, column_name: str
) -> dict[str, list[float] | list[int]] | None:
    """Create a histogram for a numeric variable in a dataframe.

    Args:
        df (pl.DataFrame): The dataframe to analyze.
        column_name (str): The column to analyze.

    Returns:
        dict[str, list[float] | list[int]]: The bin edges
            and the count of rows in each bin.
            If failed to create, dictionary is empty.

    """
    logger = get_mlflow_logger()

    output: dict[str, list[float] | list[int]] = {}
    histogram_bins: list[float] = []
    histogram_counts: list[int] = []

    if column_name not in df.columns:
        msg = f"Column '{column_name}' not found in DataFrame."
        raise KeyError(msg)
    if not df[column_name].dtype.is_numeric():
        msg = f"Column '{column_name}' must be numeric."
        raise TypeError(msg)
    if df.select(pl.col(column_name)).is_empty():
        return {
            "bin_edges": histogram_bins,
            "counts": histogram_counts,
        }

    try:
        histogram = (
            df.select(pl.col(column_name).hist(include_category=True))
            .unnest(column_name)
            .sort("category")
        )
        if (
            histogram is None or len(histogram) <= 0
        ):  # pragma: no cover  # defensive typing polars return
            msg = f"Failed to create histogram for {column_name}."
            logger.warning(msg, stacklevel=2)
            return None

        # Convert interval strings to numeric bin edges
        # Sort to ensure we process bins in order
        for row in histogram.rows():
            category, count = row

            # Extract numeric values from interval notation
            # Format is typically "(lower, upper]" or "[lower, upper]"
            if not category or not isinstance(category, str):
                msg = f"Failed to parse bin for {column_name}: {category}"
                logger.warning(msg, stacklevel=2)
                continue

            # Strip brackets and split by comma
            expected_num_bounds = 2
            bounds = category.strip("()[]").split(",")
            if (
                len(bounds) != expected_num_bounds
            ):  # pragma: no cover  # defensive polars output format return
                msg = f"Failed to parse bin for {column_name}: {category}"
                logger.warning(msg, stacklevel=2)
                continue

            try:
                lower = float(bounds[0].strip())
                upper = float(bounds[1].strip())
                # For first bin, add the lower edge
                if not histogram_bins:
                    histogram_bins.append(lower)

                # Always add the upper edge
                histogram_bins.append(upper)
                histogram_counts.append(count)
            except (
                ValueError
            ):  # pragma: no cover  # defensive polars output format parsing
                logger.exception(f"Failed to parse bin for {column_name}")
                continue

        # Only store if we successfully parsed bins
        if (
            histogram_bins
            and histogram_counts
            and len(histogram_bins) == len(histogram_counts) + 1
        ):
            output = {"bin_edges": histogram_bins, "counts": histogram_counts}
            return output

        # pragma: no cover  # defensive polars output format parsing
        logger.exception(f"Failed to create histogram for {column_name}")
    except Exception:
        logger.exception(f"Failed to create histogram for {column_name}")
    return None


@mlflow.trace(name="get_category_levels_for_column", span_type="func")
def get_category_levels_for_column(
    df: pl.DataFrame, column_name: str
) -> dict[str, int]:
    """Create a histogram for a numeric variable in a dataframe.

    Args:
        df (pl.DataFrame): The dataframe to analyze.
        column_name (str): The column to analyze.

    Returns:
        dict[str, int]: The category levels and their counts.
            If failed to create, dictionary is empty.

    """
    category_counts: dict[str, int] = {}
    try:
        value_counts = df.select(
            pl.col(column_name).value_counts(sort=True, name="counts")
        )

        # Build category counts dictionary
        if len(value_counts) > 0:
            # Convert to dict, limiting to reasonable number
            for row_idx, row in enumerate(value_counts.rows()):
                if row_idx >= MAX_CATEGORIES_FOR_LEVEL:
                    break
                row_dict = row[0]
                value = row_dict.get(f"{column_name}")
                count = row_dict.get("counts")
                category_counts[str(value)] = count
    except Exception:
        logger = get_mlflow_logger()
        logger.exception(f"Failed to create histogram for {column_name}")
    return category_counts


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
        "q1": df_filtered.select(pl.col(column_name).quantile(0.25)).item(),
        "q3": df_filtered.select(pl.col(column_name).quantile(0.75)).item(),
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
