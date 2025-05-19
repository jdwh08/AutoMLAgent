#####################################################
# AutoMLAgent [EDA COLUMN TOOLS]
# ####################################################
# Jonathan Wang <jdwh08>

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""EDA Column Tool Orchestration."""

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
from automlagent.eda.column.column_histogram import get_histogram_for_column
from automlagent.eda.column.column_info_string import generate_info_string_for_column
from automlagent.eda.column.column_quality import (
    column_data_quality_missing_inf,
    get_data_quality_for_column,
)
from automlagent.eda.column.column_stats import (
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
    """Settings for column analysis."""

    outlier_zscore_threshold: float = 3.0
    low_variance_threshold: float = 0.01
    default_is_feature_var: bool = (
        True  # NOTE(jdwh08): assume is feature var as opposed to target
    )


#####################################################
### HANDLER FUNCTIONS


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
    missing_inf_data = column_data_quality_missing_inf(df, column_name)
    column_info = column_info.model_copy(update=missing_inf_data)
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
        histogram_data = get_histogram_for_column(
            df, column_name, column_info=column_info
        )
        if histogram_data:
            # Update the histogram field with the numerical histogram data
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
        category_data = get_histogram_for_column(
            df, column_name, column_info=column_info
        )
        # Update the histogram field with the categorical data
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
        quality_data = get_data_quality_for_column(
            df,
            column_name,
            column_info=column_info,
            outlier_zscore_threshold=outlier_zscore_threshold,
            low_variance_threshold=low_variance_threshold,
        )
        column_info = column_info.model_copy(update=quality_data)
    except Exception:
        logger = get_mlflow_logger()
        logger.exception(f"Failed to calculate data quality for {column_name}")
    return column_info


#####################################################
### PREDICATE FUNCTIONS


def unknown_type_predicate(column_info: ColumnInfo) -> bool:
    return getattr(column_info, "type", None) == ColumnType.UNKNOWN


def missing_value_predicate(column_info: ColumnInfo) -> bool:
    return getattr(column_info, "missing_count", None) is None


def numerical_type_predicate(column_info: ColumnInfo) -> bool:
    return getattr(column_info, "is_numeric", False)


def categorial_type_predicate(column_info: ColumnInfo) -> bool:
    return getattr(column_info, "is_categorical", False)


def temporal_type_predicate(column_info: ColumnInfo) -> bool:
    return getattr(column_info, "is_temporal", False)


def data_quality_predicate(column_info: ColumnInfo) -> bool:
    return getattr(column_info, "outlier_count", None) is None


#####################################################
### HANDLER SEQUENCE FACTORY
def create_analyze_column_handlers(
    column_info: ColumnInfo,  # noqa: ARG001
    outlier_zscore_threshold: float = 3.0,
    low_variance_threshold: float = 0.01,
) -> Sequence[
    tuple[
        Callable[[ColumnInfo], bool],
        Callable[[pl.DataFrame, str, ColumnInfo], ColumnInfo],
    ]
]:
    """Create a sequence of (predicate, handler) pairs for analyzing columns.

    Args:
        column_info: ColumnInfo object to update
        outlier_zscore_threshold: Z-score threshold for outlier detection
        low_variance_threshold: Variance threshold for low variation

    Returns:
        Sequence of (predicate, handler) pairs for analyzing columns

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
        (unknown_type_predicate, analyze_column_unknown_type_handler),
        (missing_value_predicate, analyze_column_missing_value_handler),
        (numerical_type_predicate, analyze_column_numerical_type_handler),
        (categorial_type_predicate, analyze_column_categorial_type_handler),
        (temporal_type_predicate, analyze_column_temporal_type_handler),
        (data_quality_predicate, analyze_column_data_quality_handler),
    ]


#####################################################
### MAIN ORCHESTRATION FUNCTION
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
    logger = get_mlflow_logger()

    # Validate inputs
    if column_info is None and df_info is None:
        msg = "Either column_info or df_info must be provided"
        logger.exception(msg)
        raise ValueError(msg)

    if column_info is not None and df_info is not None:
        msg = "Cannot provide both column_info and df_info"
        logger.exception(msg)
        raise ValueError(msg)

    if column_name not in df.columns:
        msg = f"Column {column_name} not found in dataframe"
        logger.exception(msg)
        raise ValueError(msg)

    # Get column_info from df_info
    if column_info is None and df_info is not None:
        column_info = next(
            (c for c in df_info.column_info if c.name == column_name), None
        )

    if column_info is None:  # NOTE(jdwh08): this is here insetad of nested for typing
        msg = f"Column info for {column_name} not found in df_info."
        logger.exception(msg)
        raise ValueError(msg)

    # Generate settings if not provided
    analysis_settings = analysis_settings or ColumnAnalysisSettings()

    # Generate warning if column is analyzed already.
    if column_info.is_analyzed:
        msg = f"Column {column_name} has already been analyzed"
        logger.warning(msg)
        return column_info

    try:
        handlers = create_analyze_column_handlers(
            column_info,
            analysis_settings.outlier_zscore_threshold,
            analysis_settings.low_variance_threshold,
        )

        for predicate, handler in handlers:
            if predicate(column_info):
                column_info = handler(df, column_name, column_info)

        # Generate info summary string based on column type
        column_info = column_info.model_copy(update={"is_analyzed": True})
        column_info_string = generate_info_string_for_column(column_info)
        column_info = column_info.model_copy(update=column_info_string)

    except Exception:
        logger.exception(f"Failed to analyze column {column_name}")
        column_info = column_info.model_copy(update={"is_analyzed": False})

    return column_info
