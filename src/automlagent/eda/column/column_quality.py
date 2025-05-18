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
import polars.selectors as cs

if TYPE_CHECKING:
    from collections.abc import Callable

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.column_type import ColumnType
from automlagent.eda.column.column_utils import column_filter_out_missing
from automlagent.logger.mlflow_logger import get_mlflow_logger

#####################################################
### SETTINGS
LOW_VARIATION_THRESHOLD = 0.2  # x% not majority class to be low variation

ColumnDataQualityMetrics: TypeAlias = dict[str, int | float | bool]


#####################################################
### CODE
@mlflow.trace(name="column_data_quality_missing", span_type="func")
def column_data_quality_missing_inf(
    df: pl.DataFrame,
    column_name: str,
) -> ColumnDataQualityMetrics:
    """Analyze data quality for a column with missing values.

    Args:
        df: The input dataframe
        column_name: Name of the column to analyze

    Returns:
        ColumnDataQualityMetrics: If all nulls, populate.
        Empty dict otherwise.

    """
    column = pl.col(column_name)

    df_no_missing = column_filter_out_missing(df, column_name)
    num_missing = df.height - df_no_missing.height

    inf_count = 0
    # NOTE(jdwh08): Only numeric columns can have infinite values
    if column_name in df.select(cs.numeric()).columns:
        inf_count = df.select(
            column.filter(column.is_infinite()).alias(column_name)
        ).height

    output = {
        "missing_count": num_missing,
        "missing_rate": num_missing / df.height,
        "inf_count": inf_count,
    }
    return output


def column_data_quality_missing_check(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo,
) -> ColumnDataQualityMetrics:
    """Analyze data quality for a column with missing values.

    Args:
        df: The input dataframe
        column_name: Name of the column to analyze
        column_info: Optional ColumnInfo object

    Returns:
        ColumnDataQualityMetrics: If all nulls, populate.
        Empty dict otherwise.

    """
    if column_name not in df.columns:
        msg = f"Column '{column_name}' not found in DataFrame."
        raise KeyError(msg)

    # Check if all nulls
    result: ColumnDataQualityMetrics = {}

    if column_info.missing_rate is None:
        column_info = column_info.model_copy(
            update=column_data_quality_missing_inf(df, column_name)
        )
        if column_info.missing_rate is None:
            msg = f"Failed to calculate missing rate for column {column_name}"
            raise ValueError(msg)

    # If all nulls, return this default result
    if column_info.missing_rate >= 1.0:
        result["outlier_count"] = 0
        result["outlier_rate"] = 0.0
        result["has_low_variation"] = True
    return result


@mlflow.trace(name="column_data_quality_boolean", span_type="func")
def column_data_quality_boolean_handler(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo,
) -> ColumnDataQualityMetrics:
    """Analyze data quality for a boolean column.

    Args:
        df: The input dataframe
        column_name: Name of the column to analyze
        column_info: Optional ColumnInfo object

    Returns:
        ColumnDataQualityMetrics: dict with data quality metrics

    """
    result: ColumnDataQualityMetrics = column_data_quality_missing_check(
        df, column_name, column_info
    )
    if result != {}:
        return result

    try:
        # For boolean columns, we only need to check for low variation
        # A boolean column has low variation if one value dominates
        value_counts = df.select(
            pl.col(column_name).value_counts(sort=True, name="counts")
        ).unnest(column_name)

        if value_counts.height > 0:
            max_count = value_counts.select(pl.col("counts").max()).item()
            has_low_variation = (max_count / df.height) > (1 - LOW_VARIATION_THRESHOLD)
            result["has_low_variation"] = has_low_variation

        # Boolean columns don't have outliers in the traditional sense
        result["outlier_count"] = 0
        result["outlier_rate"] = 0.0
    except Exception:
        logger = get_mlflow_logger()
        logger.exception(f"Error analyzing quality for boolean column {column_name}")
    return result


@mlflow.trace(name="column_data_quality_numeric", span_type="func")
def column_data_quality_numeric(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo,
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
        df, column_name, column_info
    )
    if result != {}:
        return result

    def _get_mean_and_std(
        col: pl.Series, column_info: ColumnInfo
    ) -> tuple[float, float]:
        raw_mean = (
            column_info.mean
            if column_info.mean is not None
            else col.drop_nans().drop_nulls().mean()
        )
        raw_std = (
            column_info.std
            if column_info.std is not None
            else col.drop_nans().drop_nulls().std()
        )

        if raw_mean is None or raw_std is None:
            # NOTE(jdwh08): mean of not-computable column is none.
            # NOTE(jdwh08): polars mean of column ignores missing (nanmean).
            msg = "Column mean or standard deviation is None or not computable."
            raise ValueError(msg)

        try:
            mean = float(raw_mean)  # type: ignore[arg-type] # NOTE(jdwh08): polars sucks for typing
            std = float(raw_std)  # type: ignore[arg-type]
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
        logger = get_mlflow_logger()
        logger.exception(f"Error analyzing quality for numeric column {column_name}")
    return result


@mlflow.trace(name="column_data_quality_categorical", span_type="func")
def column_data_quality_categorical_handler(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo,
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
        df, column_name, column_info
    )
    if result != {}:
        return result

    try:
        top_count = None
        # Get categorical histogram data
        categorical_histogram = {
            k: v for k, v in column_info.histogram.items() if isinstance(k, str)
        }

        if categorical_histogram:
            top_count = max(categorical_histogram.values())
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
        logger = get_mlflow_logger()
        logger.exception(
            f"Error analyzing quality for categorical column {column_name}"
        )
    return result


@mlflow.trace(name="column_data_quality_temporal", span_type="func")
def column_data_quality_temporal_handler(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo,
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
        df, column_name, column_info
    )
    if result != {}:
        return result

    # NOTE(jdwh08): A response to TRY301 -- feels silly?
    def _raise_conversion_error(msg: str) -> None:
        raise ValueError(msg)

    time_series: pl.Series = df[column_name]

    try:
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

        # NOTE(jdwh08): type ignore <handled w/ raise error>
        iqr: float = q3 - q1  # type: ignore[operator]
        lower_bound: float = floor(q1 - 1.5 * iqr)  # type: ignore[operator]
        upper_bound: float = ceil(q3 + 1.5 * iqr)  # type: ignore[operator]

        lower_bound_dt = (
            pl.from_epoch([int(lower_bound)], time_unit="d")
            if time_series.dtype == pl.Date
            else pl.from_epoch([int(lower_bound)], time_unit="ms")
        )
        if lower_bound_dt.dtype == pl.Datetime:
            lower_bound_dt = lower_bound_dt.dt.replace_time_zone(column_info.timezone)
        lower_bound_dt = lower_bound_dt[0]

        upper_bound_dt = (
            pl.from_epoch([int(upper_bound)], time_unit="d")
            if time_series.dtype == pl.Date
            else pl.from_epoch([int(upper_bound)], time_unit="ms")
        )
        if upper_bound_dt.dtype == pl.Datetime:
            upper_bound_dt = upper_bound_dt.dt.replace_time_zone(column_info.timezone)
        upper_bound_dt = upper_bound_dt[0]

        outliers = df.filter(
            (pl.col(column_name) < lower_bound_dt)
            | (pl.col(column_name) > upper_bound_dt)
        )
        outlier_count = outliers.height
        result["outlier_count"] = outlier_count
        result["outlier_rate"] = outlier_count / df.height if df.height > 0 else 0.0
        # TODO(jdwh08): Improve "low variation" method for temporal
        # Low variation for temporal: all values same or nearly same
        unique_count = df[column_name].n_unique()
        result["has_low_variation"] = unique_count <= 1
    except Exception:
        logger = get_mlflow_logger()
        logger.exception(f"Error analyzing quality for temporal column {column_name}")
    return result


@mlflow.trace(name="get_data_quality_for_column", span_type="func")
def get_data_quality_for_column(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo,
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
        ColumnType.INT: column_data_quality_numeric_handler,
        ColumnType.FLOAT: column_data_quality_numeric_handler,
        ColumnType.CATEGORICAL: column_data_quality_categorical_handler,
        ColumnType.TEXT: column_data_quality_categorical_handler,
        ColumnType.DATETIME: column_data_quality_temporal_handler,
        ColumnType.DATE: column_data_quality_temporal_handler,
        ColumnType.TIME: column_data_quality_temporal_handler,
        ColumnType.BOOLEAN: column_data_quality_boolean_handler,
    }

    handler = strategy_map.get(col_type)
    if handler is None:
        logger = get_mlflow_logger()
        logger.warning(f"Unknown column type for {column_name}: {col_type}")
        return {}
    return handler(df, column_name, column_info)
