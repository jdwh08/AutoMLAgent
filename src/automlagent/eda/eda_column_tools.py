#####################################################
# AutoMLAgent [EDA COLUMN TOOLS]
# ####################################################
# Jonathan Wang <jdwh08>

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""EDA Column Tools."""

#####################################################
### BOARD

#####################################################
### IMPORTS

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mlflow
import polars as pl

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.column_type import ColumnType
from automlagent.dataclass.df_info import DataFrameInfo
from automlagent.eda.column.column_info_string import generate_info_string_for_column
from automlagent.eda.column.column_quality import get_data_quality_for_column
from automlagent.eda.column.column_stats import (
    get_category_levels_for_column,
    get_histogram_bins_for_column,
    get_numerical_stats_for_column,
    get_temporal_stats_for_column,
)
from automlagent.eda.column.column_types import get_type_for_column
from automlagent.logger.mlflow_logger import get_mlflow_logger

#####################################################
### SETTINGS


#####################################################
### DATACLASSES
# NOTE(jdwh08): Default settings for column analysis
@dataclass
class ColumnAnalysisSettings:
    outlier_zscore_threshold: float = 3.0
    low_variance_threshold: float = 0.01


#####################################################
### CODE


@mlflow.trace(name="analyze_column_unknown_type", span_type="func")
def analyze_column_unknown_type_handler(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo,
) -> ColumnInfo:
    type_data = get_type_for_column(df, column_name, column_info=column_info)
    column_info = column_info.model_copy(update=type_data)
    return column_info


@mlflow.trace(name="analyze_column_missing_value", span_type="func")
def analyze_column_missing_value_handler(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo,
) -> ColumnInfo:
    total_rows = len(df)
    missing_count = df[column_name].null_count()
    column_info = column_info.model_copy(
        update={
            "missing_count": missing_count,
            "missing_rate": missing_count / total_rows if total_rows > 0 else 0.0,
        }
    )
    return column_info


@mlflow.trace(name="analyze_column_numerical_type", span_type="func")
def analyze_column_numerical_type_handler(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo,
) -> ColumnInfo:
    # Get numerical stats
    try:
        numeric_stats = get_numerical_stats_for_column(df, column_name)
        column_info = column_info.model_copy(update=numeric_stats)
    except Exception:
        logger = get_mlflow_logger()
        logger.exception(f"Failed to calculate numeric stats for {column_name}")

    # Get histogram data
    try:
        histogram_data = get_histogram_bins_for_column(df, column_name)
        if histogram_data:
            column_info = column_info.model_copy(update=histogram_data)
    except Exception:
        logger = get_mlflow_logger()
        logger.exception(f"Failed to create histogram for {column_name}")
    return column_info


@mlflow.trace(name="analyze_column_categorial_type", span_type="func")
def analyze_column_categorial_type_handler(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo,
) -> ColumnInfo:
    # Get category levels and counts
    try:
        category_data = get_category_levels_for_column(df, column_name)
        column_info = column_info.model_copy(update=category_data)
    except Exception:
        logger = get_mlflow_logger()
        logger.exception(f"Failed to get category levels for {column_name}")
    return column_info


@mlflow.trace(name="analyze_column_temporal_type", span_type="func")
def analyze_column_temporal_type_handler(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo,
) -> ColumnInfo:
    # Get temporal stats
    try:
        temporal_stats = get_temporal_stats_for_column(df, column_name)
        column_info = column_info.model_copy(update=temporal_stats)
    except Exception:
        logger = get_mlflow_logger()
        logger.exception(f"Failed to calculate temporal stats for {column_name}")
    return column_info


@mlflow.trace(name="analyze_column_data_quality", span_type="func")
def analyze_column_data_quality(
    df: pl.DataFrame,
    column_name: str,
    column_info: ColumnInfo,
    outlier_zscore_threshold: float = 3.0,
    low_variance_threshold: float = 0.01,
) -> ColumnInfo:
    try:
        quality_metrics = get_data_quality_for_column(
            df,
            column_name,
            column_info=column_info,
            outlier_zscore_threshold=outlier_zscore_threshold,
            low_variance_threshold=low_variance_threshold,
        )
        column_info = column_info.model_copy(update=quality_metrics)
    except Exception:
        logger = get_mlflow_logger()
        logger.exception(f"Failed to calculate data quality metrics for {column_name}")
    return column_info


def create_analyze_column_handlers(
    column_info: ColumnInfo,
    outlier_zscore_threshold: float = 3.0,
    low_variance_threshold: float = 0.01,
) -> Sequence[
    tuple[
        Callable[[ColumnInfo], bool],
        Callable[[pl.DataFrame, str, ColumnInfo], ColumnInfo],
    ]
]:
    """Create a sequence of handlers for analyzing columns.

    Args:
        column_info: ColumnInfo object to update
        outlier_zscore_threshold: Z-score threshold for outlier detection
        low_variance_threshold: Variance threshold for low variation

    Returns:
        Sequence of handlers for analyzing columns

    """

    def analyze_column_data_quality_handler(
        df: pl.DataFrame,
        column_name: str,
        column_info: ColumnInfo,
    ) -> ColumnInfo:
        return analyze_column_data_quality(
            df,
            column_name,
            column_info,
            outlier_zscore_threshold,
            low_variance_threshold,
        )

    return [
        (
            lambda col_info: col_info.type == ColumnType.UNKNOWN,
            analyze_column_unknown_type_handler,
        ),
        (
            lambda col_info: col_info.missing_count is None,
            analyze_column_missing_value_handler,
        ),
        (
            lambda col_info: col_info.is_numeric,
            analyze_column_numerical_type_handler,
        ),
        (
            lambda col_info: col_info.is_categorial,
            analyze_column_categorial_type_handler,
        ),
        (
            lambda col_info: col_info.is_temporal,
            analyze_column_temporal_type_handler,
        ),
        (
            lambda col_info: col_info.outlier_count is None,
            analyze_column_data_quality_handler,
        ),
    ]


@mlflow.trace(name="analyze_column", span_type="func")
def analyze_column(
    df: pl.DataFrame,
    column_name: str,
    *,
    column_info: ColumnInfo | None = None,
    df_info: DataFrameInfo | None = None,
    analysis_settings: ColumnAnalysisSettings | None = None,
) -> ColumnInfo:
    """Analyze a single variable column in a dataframe.

    This function composes multiple specialized analysis functions to provide
    a comprehensive analysis of a column based on its type.

    Args:
        df: Dataframe containing the variable
        column_name: Name of the column to analyze
        column_info: Pre-existing ColumnInfo to update (optional)
        df_info: Pre-existing DataFrameInfo from which to extract column_info (optional)
        analysis_settings: Settings for column analysis

    Returns:
        Updated ColumnInfo with comprehensive analysis

    """
    # Validate inputs
    logger = get_mlflow_logger()
    if column_name not in df.columns:
        msg = f"Column {column_name} not found in dataframe"
        logger.exception(msg)
        raise ValueError(msg)

    # Get or create column_info
    if column_info is None and df_info is not None:
        column_info = next(
            (c for c in df_info.column_info if c.name == column_name), None
        )

    if column_info is None:
        msg = f"Column {column_name} not found in column_info or df_info"
        logger.exception(msg)
        raise ValueError(msg)

    # Generate warning if column is analyzed already.
    if column_info.is_analyzed:
        msg = f"Column {column_name} has already been analyzed"
        logger.warning(msg)
        return column_info

    # Generate settings if not provided
    if analysis_settings is None:
        analysis_settings = ColumnAnalysisSettings()

    try:
        handlers = create_analyze_column_handlers(
            column_info,
            analysis_settings.outlier_zscore_threshold,
            analysis_settings.low_variance_threshold,
        )

        for condition, handler in handlers:
            if condition(column_info):
                column_info = handler(df, column_name, column_info)

        # Generate info summary string based on column type
        column_info = column_info.model_copy(update={"is_analyzed": True})
        column_info_string = generate_info_string_for_column(column_info)
        column_info = column_info.model_copy(update=column_info_string)

    except Exception:
        logger.exception(f"Failed to analyze column {column_name}")
        column_info = column_info.model_copy(update={"is_analyzed": False})

    return column_info
