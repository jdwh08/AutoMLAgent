#####################################################
# AutoMLAgent [EDA COLUMN QUALITY]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Data quality analysis for EDA columns."""

#####################################################
### BOARD

#####################################################
### IMPORTS

from __future__ import annotations

from collections.abc import Callable
from math import ceil, floor

import mlflow
import polars as pl

### OWN MODULES
from AutoMLAgent.dataclass.column_info import ColumnInfo
from AutoMLAgent.dataclass.column_type import ColumnType
from AutoMLAgent.eda.column.column_types import get_type_for_column
from AutoMLAgent.logger.mlflow_logger import logger

#####################################################
### SETTINGS
LOW_VARIATION_THRESHOLD = 0.05  # x% not majority class

#####################################################
### CODE


@mlflow.trace(name="get_data_quality_for_column_numeric", span_type="func")
def get_data_quality_for_column_numeric(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo | None = None,
    outlier_zscore_threshold: float = 3.0,
    low_variance_threshold: float = 0.01,
) -> dict[str, int | float | bool]:
    """Analyze data quality for a numeric column (INTEGER, FLOAT).

    Args:
        df: The input dataframe
        column_name: Name of the column to analyze
        column_info: Optional ColumnInfo object
        outlier_zscore_threshold: Z-score threshold for outlier detection
        low_variance_threshold: Variance threshold for low variation

    Returns:
        dict with data quality metrics

    """
    result: dict[str, int | float | bool] = {}
    col = df[column_name]

    # Check if all nulls
    if col.is_null().all():
        result["outlier_count"] = 0
        result["outlier_rate"] = 0.0
        result["has_low_variation"] = True
        return result

    try:
        mean: float
        std: float
        if column_info is not None:
            mean = column_info.mean or float(col.mean())
            std = column_info.std or float(col.std())
        else:
            mean = float(col.mean())
            std = float(col.std())

        z_scores = df.select(
            ((pl.col(column_name) - mean) / std).abs().alias("z_score")
        )
        outlier_mask = z_scores.filter(pl.col("z_score") > outlier_zscore_threshold)
        outlier_count = outlier_mask.height
        result["outlier_count"] = outlier_count
        result["outlier_rate"] = outlier_count / df.height if df.height > 0 else 0.0

        variance = std**2
        result["has_low_variation"] = variance < low_variance_threshold
    except Exception:
        logger.exception(f"Error analyzing quality for numeric column {column_name}")
    return result


@mlflow.trace(name="get_data_quality_for_column_categorical", span_type="func")
def get_data_quality_for_column_categorical(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo | None = None,
) -> dict[str, int | float | bool]:
    """Analyze data quality for a categorical column (CATEGORICAL, TEXT).

    Args:
        df: The input dataframe
        column_name: Name of the column to analyze
        column_info: Optional ColumnInfo object

    Returns:
        dict with data quality metrics

    """
    result: dict[str, int | float | bool] = {}
    try:
        top_count = None
        if column_info is not None and getattr(column_info, "category_counts", None):
            category_counts = column_info.category_counts
            top_count = max(category_counts.values()) if category_counts else None
        else:
            value_counts = df.select(
                pl.col(column_name).value_counts(sort=True, name="counts")
            ).unnest(column_name)
            top_count = value_counts.select(pl.col("counts").max()).item()
        if top_count is not None:
            has_low_variation = (top_count / df.height) > min(
                1 - LOW_VARIATION_THRESHOLD, 1
            )
            result["has_low_variation"] = has_low_variation
        result["outlier_count"] = 0
        result["outlier_rate"] = 0.0
    except Exception:
        logger.exception(
            f"Error analyzing quality for categorical column {column_name}"
        )
    return result


@mlflow.trace(name="get_data_quality_for_column_temporal", span_type="func")
def get_data_quality_for_column_temporal(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo | None = None,
) -> dict[str, int | float | bool]:
    """Analyze data quality for a temporal column (DATETIME, DATE, TIME).

    Args:
        df: The input dataframe
        column_name: Name of the column to analyze
        column_info: Optional ColumnInfo object

    Returns:
        dict with data quality metrics

    """
    result: dict[str, int | float | bool] = {}
    try:
        time_series: pl.Series = df[column_name]
        time_series_float = (
            time_series.cast(pl.Int128)
            if time_series.dtype == pl.Date
            else time_series.dt.timestamp("ms")
        )
        # For temporal data, detect outliers using the IQR method
        q1: float = time_series_float.quantile(0.25)
        q3: float = time_series_float.quantile(0.75)
        iqr: float = q3 - q1
        lower_bound: float = floor(q1 - 1.5 * iqr)
        upper_bound: float = ceil(q3 + 1.5 * iqr)
        lower_bound_dt = (
            pl.from_epoch([int(lower_bound)], time_unit="d")[0]
            if time_series.dtype == pl.Date
            else pl.from_epoch([int(lower_bound)], time_unit="ms")[0]
        )
        upper_bound_dt = (
            pl.from_epoch([int(upper_bound)], time_unit="d")[0]
            if time_series.dtype == pl.Date
            else pl.from_epoch([int(upper_bound)], time_unit="ms")[0]
        )
        outliers = df.filter(
            (pl.col(column_name) < lower_bound_dt)
            | (pl.col(column_name) > upper_bound_dt)
        )
        outlier_count = outliers.height
        result["outlier_count"] = outlier_count
        result["outlier_rate"] = outlier_count / df.height if df.height > 0 else 0.0
        # Low variation for temporal: all values same or nearly same
        unique_count = df[column_name].n_unique()
        result["has_low_variation"] = unique_count <= 1
    except Exception:
        logger.exception(f"Error analyzing quality for temporal column {column_name}")
    return result


@mlflow.trace(name="get_data_quality_for_column", span_type="func")
def get_data_quality_for_column(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo | None = None,
    outlier_zscore_threshold: float = 3.0,
    low_variance_threshold: float = 0.01,
) -> dict[str, int | float | bool]:
    """Analyze data quality for a single column.

    Args:
        df: The input dataframe
        column_name: Name of the column to analyze
        column_info: Optional ColumnInfo object
        outlier_zscore_threshold: Z-score threshold for outlier detection
        low_variance_threshold: Variance threshold for low variation

    Returns:
        dict with data quality metrics

    """
    # Determine column type
    if column_info and column_info.type:
        col_type: ColumnType = column_info.type
    else:
        # Crappy fallback (doesn't set the column info...)
        col_type_info = get_type_for_column(df, column_name).get("type")
        if not isinstance(col_type_info, ColumnType):
            msg = f"Invalid column type: {col_type_info}"
            raise ValueError(msg)
        col_type = col_type_info

    def numeric_handler(
        df: pl.DataFrame,
        column_name: str,
        column_info: ColumnInfo | None = None,
    ) -> dict[str, int | float | bool]:
        return get_data_quality_for_column_numeric(
            df,
            column_name,
            column_info=column_info,
            outlier_zscore_threshold=outlier_zscore_threshold,
            low_variance_threshold=low_variance_threshold,
        )

    # Strategy dispatcher
    strategy_map: dict[
        ColumnType,
        Callable[[pl.DataFrame, str, ColumnInfo | None], dict[str, int | float | bool]],
    ] = {
        ColumnType.INTEGER: numeric_handler,
        ColumnType.FLOAT: numeric_handler,
        ColumnType.CATEGORICAL: get_data_quality_for_column_categorical,
        ColumnType.TEXT: get_data_quality_for_column_categorical,
        ColumnType.DATETIME: get_data_quality_for_column_temporal,
        ColumnType.DATE: get_data_quality_for_column_temporal,
        ColumnType.TIME: get_data_quality_for_column_temporal,
    }

    handler = strategy_map.get(col_type)
    if handler is None:
        logger.warning(f"Unknown column type for {column_name}: {col_type}")
        return {}
    return handler(df, column_name, column_info)
