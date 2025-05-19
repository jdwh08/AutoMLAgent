#####################################################
# AutoMLAgent [EDA DATAFRAME TOOLS]
#####################################################
# Jonathan Wang <jdwh08>

# ABOUT:
# Orchestrates DataFrame-level EDA using modular df functions.

"""EDA DataFrame Tool Orchestration."""

#####################################################
### IMPORTS

from collections.abc import Callable, Sequence
from typing import TypeAlias

import polars as pl
from pydantic import BaseModel, Field

### OWN MODULES
from automlagent.dataclass.df_info import DataFrameInfo
from automlagent.eda.df.df_corr import get_pearson_correlation_for_df
from automlagent.eda.df.df_histogram import get_histogram_for_df
from automlagent.eda.df.df_quality import (
    get_data_quality_for_df,
    get_missing_values_for_df,
)
from automlagent.eda.df.df_stats import (
    get_numerical_stats_for_df,
    get_string_stats_for_df,
    get_temporal_stats_for_df,
)
from automlagent.eda.df.df_types import get_column_types
from automlagent.logger.mlflow_logger import get_mlflow_logger

#####################################################
### SETTINGS CLASSES


class QualityAnalysisSettings(BaseModel):
    outlier_zscore_threshold: float = Field(default=3.0, ge=0.0)
    low_variance_threshold: float = Field(default=0.01, ge=0.0, le=1.0)


class DataFrameAnalysisSettings(BaseModel):
    quality: QualityAnalysisSettings = Field(default_factory=QualityAnalysisSettings)


#####################################################
### TYPE ALIASES

PredicateFunc: TypeAlias = Callable[[pl.DataFrame, DataFrameInfo], bool]
HandlerFunc: TypeAlias = Callable[
    [pl.DataFrame, DataFrameInfo, DataFrameAnalysisSettings], DataFrameInfo
]
HandlerPair: TypeAlias = tuple[PredicateFunc, HandlerFunc]

#####################################################
### HANDLER FUNCTIONS


def analyze_types_handler(
    df: pl.DataFrame,
    df_info: DataFrameInfo,
    settings: DataFrameAnalysisSettings,  # noqa: ARG001
) -> DataFrameInfo:
    logger = get_mlflow_logger()
    try:
        return get_column_types(df, df_info)
    except Exception:
        logger.exception("Failed to infer column types.")
        return df_info


def analyze_quality_handler(
    df: pl.DataFrame, df_info: DataFrameInfo, settings: DataFrameAnalysisSettings
) -> DataFrameInfo:
    logger = get_mlflow_logger()
    try:
        return get_data_quality_for_df(
            df,
            df_info,
            outlier_zscore_threshold=settings.quality.outlier_zscore_threshold,
            low_variance_threshold=settings.quality.low_variance_threshold,
        )
    except Exception:
        logger.exception("Failed to compute data quality.")
        return df_info


def analyze_missing_handler(
    df: pl.DataFrame,
    df_info: DataFrameInfo,
    settings: DataFrameAnalysisSettings,  # noqa: ARG001
) -> DataFrameInfo:
    logger = get_mlflow_logger()
    try:
        return get_missing_values_for_df(df, df_info)
    except Exception:
        logger.exception("Failed to compute missing values.")
        return df_info


def analyze_numerical_stats_handler(
    df: pl.DataFrame,
    df_info: DataFrameInfo,
    settings: DataFrameAnalysisSettings,  # noqa: ARG001
) -> DataFrameInfo:
    logger = get_mlflow_logger()
    try:
        return get_numerical_stats_for_df(df, df_info)
    except Exception:
        logger.exception("Failed to compute numerical stats.")
        return df_info


def analyze_string_stats_handler(
    df: pl.DataFrame,
    df_info: DataFrameInfo,
    settings: DataFrameAnalysisSettings,  # noqa: ARG001
) -> DataFrameInfo:
    logger = get_mlflow_logger()
    try:
        return get_string_stats_for_df(df, df_info)
    except Exception:
        logger.exception("Failed to compute string stats.")
        return df_info


def analyze_histogram_bins_handler(
    df: pl.DataFrame,
    df_info: DataFrameInfo,
    settings: DataFrameAnalysisSettings,  # noqa: ARG001
) -> DataFrameInfo:
    logger = get_mlflow_logger()
    try:
        return get_histogram_for_df(df, df_info)
    except Exception:
        logger.exception("Failed to compute histogram bins.")
        return df_info


def analyze_temporal_stats_handler(
    df: pl.DataFrame,
    df_info: DataFrameInfo,
    settings: DataFrameAnalysisSettings,  # noqa: ARG001
) -> DataFrameInfo:
    logger = get_mlflow_logger()
    try:
        return get_temporal_stats_for_df(df, df_info)
    except Exception:
        logger.exception("Failed to compute temporal stats.")
        return df_info


def analyze_correlation_handler(
    df: pl.DataFrame,
    df_info: DataFrameInfo,
    settings: DataFrameAnalysisSettings,  # noqa: ARG001
) -> DataFrameInfo:
    logger = get_mlflow_logger()
    try:
        return get_pearson_correlation_for_df(df, df_info)
    except Exception:
        logger.exception("Failed to compute Pearson correlation.")
        return df_info


#####################################################
### PREDICATE FUNCTIONS


def types_missing_predicate(df: pl.DataFrame, df_info: DataFrameInfo) -> bool:  # noqa: ARG001
    return any(
        getattr(col, "type", None) is None or getattr(col, "type", None) == "UNKNOWN"
        for col in getattr(df_info, "column_info", [])
    )


def quality_missing_predicate(df: pl.DataFrame, df_info: DataFrameInfo) -> bool:  # noqa: ARG001
    return True  # Always run quality analysis (safe default)


def missing_values_predicate(df: pl.DataFrame, df_info: DataFrameInfo) -> bool:  # noqa: ARG001
    return True  # Always run missing values analysis (safe default)


def numerical_stats_predicate(df: pl.DataFrame, df_info: DataFrameInfo) -> bool:  # noqa: ARG001
    return any(
        getattr(col, "is_numeric", False) for col in getattr(df_info, "column_info", [])
    )


def string_stats_predicate(df: pl.DataFrame, df_info: DataFrameInfo) -> bool:  # noqa: ARG001
    return any(
        getattr(col, "type", None) == "TEXT"
        for col in getattr(df_info, "column_info", [])
    )


def histogram_bins_predicate(df: pl.DataFrame, df_info: DataFrameInfo) -> bool:  # noqa: ARG001
    return any(
        getattr(col, "is_numeric", False) or getattr(col, "is_categorical", False)
        for col in getattr(df_info, "column_info", [])
    )


def temporal_stats_predicate(df: pl.DataFrame, df_info: DataFrameInfo) -> bool:  # noqa: ARG001
    return any(
        getattr(col, "is_temporal", False)
        for col in getattr(df_info, "column_info", [])
    )


def correlation_predicate(df: pl.DataFrame, df_info: DataFrameInfo) -> bool:  # noqa: ARG001
    numeric_cols = [
        col
        for col in getattr(df_info, "column_info", [])
        if getattr(col, "is_numeric", False)
    ]
    return len(numeric_cols) > 1


#####################################################
### HANDLER SEQUENCE FACTORY


def create_analyze_dataframe_handlers() -> Sequence[HandlerPair]:
    """Create (predicate, handler) pairs for DataFrame EDA."""
    return [
        (types_missing_predicate, analyze_types_handler),
        (quality_missing_predicate, analyze_quality_handler),
        (missing_values_predicate, analyze_missing_handler),
        (numerical_stats_predicate, analyze_numerical_stats_handler),
        (string_stats_predicate, analyze_string_stats_handler),
        (histogram_bins_predicate, analyze_histogram_bins_handler),
        (temporal_stats_predicate, analyze_temporal_stats_handler),
        (correlation_predicate, analyze_correlation_handler),
    ]


#####################################################
### MAIN ORCHESTRATION FUNCTION
def analyze_dataframe(
    df: pl.DataFrame,
    *,
    df_info: DataFrameInfo,
    analysis_settings: DataFrameAnalysisSettings | None = None,
) -> DataFrameInfo:
    """Perform comprehensive EDA on a DataFrame using predicate-handler orchestration.

    Args:
        df (pl.DataFrame): Input dataframe.
        df_info (DataFrameInfo): DataFrame metadata and column info.
        analysis_settings (DataFrameAnalysisSettings, optional):
            Settings for outlier/variance detection.

    Returns:
        DataFrameInfo: Updated DataFrameInfo with EDA results.

    """
    if analysis_settings is None:
        analysis_settings = DataFrameAnalysisSettings()
    for predicate, handler in create_analyze_dataframe_handlers():
        if predicate(df, df_info):
            df_info = handler(df, df_info, analysis_settings)
    return df_info
