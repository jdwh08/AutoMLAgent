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

from math import ceil, floor
from typing import TYPE_CHECKING, TypeAlias

import mlflow
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.column_type import ColumnType
from automlagent.logger.mlflow_logger import logger

#####################################################
### SETTINGS
LOW_VARIATION_THRESHOLD = 0.05  # x% not majority class to be low variation

ColumnDataQualityMetrics: TypeAlias = dict[str, int | float | bool]


#####################################################
### CODE
def column_data_quality_missing_check(
    df: pl.DataFrame,
    column_name: str,
) -> ColumnDataQualityMetrics:
    """Analyze data quality for a column with missing values.

    Args:
        df: The input dataframe
        column_name: Name of the column to analyze

    Returns:
        ColumnDataQualityMetrics: dict with data quality metrics

    """
    if column_name not in df.columns:
        msg = f"Column '{column_name}' not found in DataFrame."
        raise KeyError(msg)

    # Check if all nulls
    result: ColumnDataQualityMetrics = {}
    if df[column_name].is_null().all():
        result["outlier_count"] = 0
        result["outlier_rate"] = 0.0
        result["has_low_variation"] = True
    return result


@mlflow.trace(name="column_data_quality_numeric", span_type="func")
def column_data_quality_numeric(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo | None = None,
    outlier_zscore_threshold: float = 3.0,
    low_variance_threshold: float = 0.01,
) -> ColumnDataQualityMetrics:
    """Analyze data quality for a numeric column (INTEGER, FLOAT).

    Args:
        df: The input dataframe
        column_name: Name of the column to analyze
        column_info: Optional ColumnInfo object
        outlier_zscore_threshold: Z-score threshold for outlier detection
        low_variance_threshold: Variance threshold for low variation

    Returns:
        ColumnDataQualityMetrics: dict with data quality metrics

    """
    result: ColumnDataQualityMetrics = column_data_quality_missing_check(
        df, column_name
    )
    if result != {}:
        return result

    def _get_mean_and_std(
        col: pl.Series, column_info: ColumnInfo | None = None
    ) -> tuple[float, float]:
        if column_info is not None:
            raw_mean = column_info.mean if column_info.mean is not None else col.mean()
            raw_std = column_info.std if column_info.std is not None else col.std()
        else:
            raw_mean = col.mean()
            raw_std = col.std()

        if raw_mean is None or raw_std is None:
            msg = "Mean or standard deviation is None; cannot compute z-scores."
            raise ValueError(msg)

        try:
            mean = float(raw_mean)
            std = float(raw_std)
        except (TypeError, ValueError) as e:
            msg = (
                "Mean or std is not convertible to float: "
                f"mean={raw_mean!s}, std={raw_std!s}"
            )
            raise ValueError(msg) from e

        return mean, std

    try:
        col = df[column_name]
        mean, std = _get_mean_and_std(col, column_info)

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


@mlflow.trace(name="column_data_quality_categorical", span_type="func")
def column_data_quality_categorical_handler(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo | None = None,
) -> ColumnDataQualityMetrics:
    """Analyze data quality for a categorical column (CATEGORICAL, TEXT).

    Args:
        df: The input dataframe
        column_name: Name of the column to analyze
        column_info: Optional ColumnInfo object

    Returns:
        ColumnDataQualityMetrics: dict with data quality metrics

    """
    result: ColumnDataQualityMetrics = column_data_quality_missing_check(
        df, column_name
    )
    if result != {}:
        return result

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


@mlflow.trace(name="column_data_quality_temporal", span_type="func")
def column_data_quality_temporal_handler(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo | None = None,  # noqa: ARG001
) -> ColumnDataQualityMetrics:
    """Analyze data quality for a temporal column (DATETIME, DATE, TIME).

    Args:
        df: The input dataframe
        column_name: Name of the column to analyze
        column_info: Optional ColumnInfo object

    Returns:
        ColumnDataQualityMetrics: dict with data quality metrics

    """
    result: ColumnDataQualityMetrics = column_data_quality_missing_check(
        df, column_name
    )
    if result != {}:
        return result

    # NOTE(jdwh08): A response to TRY301 -- feels silly?
    def _raise_conversion_error(msg: str) -> None:
        raise ValueError(msg)

    try:
        time_series: pl.Series = df[column_name]
        time_series_float = (
            time_series.cast(pl.Int128)
            if time_series.dtype == pl.Date
            else time_series.dt.timestamp("ms")
        )
        if time_series_float is None:
            _raise_conversion_error("Failed to convert time series to float.")

        # For temporal data, detect outliers using the IQR method
        q1 = time_series_float.quantile(0.25)
        q3 = time_series_float.quantile(0.75)
        if q1 is None or q3 is None:
            _raise_conversion_error("Failed to calculate quantiles for temporal data.")

        iqr: float = q3 - q1  # type: ignore[operator] <handled w/ raise error>
        lower_bound: float = floor(q1 - 1.5 * iqr)  # type: ignore[operator]
        upper_bound: float = ceil(q3 + 1.5 * iqr)  # type: ignore[operator]
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
) -> ColumnDataQualityMetrics:
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
        msg = f"Column type not in ColumnInfo {column_name}"
        raise ValueError(msg)

    def column_data_quality_numeric_handler(
        df: pl.DataFrame,
        column_name: str,
        column_info: ColumnInfo | None = None,
    ) -> ColumnDataQualityMetrics:
        """Handle numeric with outlier and variance args."""
        return column_data_quality_numeric(
            df,
            column_name,
            column_info=column_info,
            outlier_zscore_threshold=outlier_zscore_threshold,
            low_variance_threshold=low_variance_threshold,
        )

    # Strategy dispatcher
    strategy_map: dict[
        ColumnType,
        Callable[[pl.DataFrame, str, ColumnInfo | None], ColumnDataQualityMetrics],
    ] = {
        ColumnType.INTEGER: column_data_quality_numeric_handler,
        ColumnType.FLOAT: column_data_quality_numeric_handler,
        ColumnType.CATEGORICAL: column_data_quality_categorical_handler,
        ColumnType.TEXT: column_data_quality_categorical_handler,
        ColumnType.DATETIME: column_data_quality_temporal_handler,
        ColumnType.DATE: column_data_quality_temporal_handler,
        ColumnType.TIME: column_data_quality_temporal_handler,
    }

    handler = strategy_map.get(col_type)
    if handler is None:
        logger.warning(f"Unknown column type for {column_name}: {col_type}")
        return {}
    return handler(df, column_name, column_info)
