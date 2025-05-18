"""Tests for eda_column_tool.py."""

import polars as pl
import pytest

from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.column_type import ColumnType
from automlagent.dataclass.df_info import DataFrameInfo
from automlagent.eda.column.column_types import get_type_for_column
from automlagent.eda.eda_column_tool import (
    ColumnAnalysisSettings,
    analyze_column,
    analyze_column_categorial_type_handler,
    analyze_column_data_quality,
    analyze_column_missing_value_handler,
    analyze_column_numerical_type_handler,
    analyze_column_temporal_type_handler,
    analyze_column_unknown_type_handler,
    categorial_type_predicate,
    data_quality_predicate,
    missing_value_predicate,
    numerical_type_predicate,
    temporal_type_predicate,
    unknown_type_predicate,
)


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a sample dataframe for testing."""
    return pl.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5],
            "categorical": ["a", "b", "a", "c", "b"],
            "temporal": pl.Series(
                [
                    "2023-01-01 00:00:00",
                    "2023-01-02 00:00:00",
                    "2023-01-03 00:00:00",
                    "2023-01-04 00:00:00",
                    "2023-01-05 00:00:00",
                ],
                dtype=pl.Datetime,
            ),
            "missing": [1, None, 3, None, 5],
        }
    )


@pytest.fixture
def sample_column_info() -> ColumnInfo:
    """Create a sample ColumnInfo for testing."""
    return ColumnInfo(
        name="test_column",
        type=ColumnType.UNKNOWN,
        is_numeric=False,
        is_categorical=False,
        is_temporal=False,
    )


def test_unknown_type_predicate(sample_column_info: ColumnInfo) -> None:
    """Test the unknown_type_predicate function."""
    assert unknown_type_predicate(sample_column_info)
    sample_column_info.type = ColumnType.FLOAT
    assert not unknown_type_predicate(sample_column_info)


def test_missing_value_predicate(sample_column_info: ColumnInfo) -> None:
    """Test the missing_value_predicate function."""
    assert missing_value_predicate(sample_column_info)
    sample_column_info.missing_count = 0
    assert not missing_value_predicate(sample_column_info)


def test_numerical_type_predicate(sample_column_info: ColumnInfo) -> None:
    """Test the numerical_type_predicate function."""
    assert not numerical_type_predicate(sample_column_info)
    sample_column_info.is_numeric = True
    assert numerical_type_predicate(sample_column_info)


def test_categorial_type_predicate(sample_column_info: ColumnInfo) -> None:
    """Test the categorial_type_predicate function."""
    assert not categorial_type_predicate(sample_column_info)
    sample_column_info.is_categorical = True
    assert categorial_type_predicate(sample_column_info)


def test_temporal_type_predicate(sample_column_info: ColumnInfo) -> None:
    """Test the temporal_type_predicate function."""
    assert not temporal_type_predicate(sample_column_info)
    sample_column_info.is_temporal = True
    assert temporal_type_predicate(sample_column_info)


def test_data_quality_predicate(sample_column_info: ColumnInfo) -> None:
    """Test the data_quality_predicate function."""
    assert data_quality_predicate(sample_column_info)
    sample_column_info.outlier_count = 0
    assert not data_quality_predicate(sample_column_info)


def test_analyze_column_unknown_type_handler(
    sample_df: pl.DataFrame, sample_column_info: ColumnInfo
) -> None:
    """Test the analyze_column_unknown_type_handler function."""
    result = analyze_column_unknown_type_handler(
        sample_df, "numeric", sample_column_info
    )
    assert result.type != ColumnType.UNKNOWN


def test_analyze_column_missing_value_handler(
    sample_df: pl.DataFrame, sample_column_info: ColumnInfo
) -> None:
    """Test the analyze_column_missing_value_handler function."""
    result = analyze_column_missing_value_handler(
        sample_df, "missing", sample_column_info
    )
    assert result.missing_count is not None


def test_analyze_column_numerical_type_handler(
    sample_df: pl.DataFrame, sample_column_info: ColumnInfo
) -> None:
    """Test the analyze_column_numerical_type_handler function."""
    sample_column_info.is_numeric = True
    result = analyze_column_numerical_type_handler(
        sample_df, "numeric", sample_column_info
    )
    assert hasattr(result, "mean")
    assert hasattr(result, "std")


def test_analyze_column_categorial_type_handler(
    sample_df: pl.DataFrame, sample_column_info: ColumnInfo
) -> None:
    """Test the analyze_column_categorial_type_handler function."""
    sample_column_info.is_categorical = True
    result = analyze_column_categorial_type_handler(
        sample_df, "categorical", sample_column_info
    )
    assert hasattr(result, "histogram")


def test_analyze_column_temporal_type_handler(
    sample_df: pl.DataFrame, sample_column_info: ColumnInfo
) -> None:
    """Test the analyze_column_temporal_type_handler function."""
    sample_column_info.is_temporal = True
    result = analyze_column_temporal_type_handler(
        sample_df, "temporal", sample_column_info
    )
    assert hasattr(result, "temporal_min")
    assert hasattr(result, "temporal_max")


def test_analyze_column_data_quality(
    sample_df: pl.DataFrame, sample_column_info: ColumnInfo
) -> None:
    """Test the analyze_column_data_quality function."""
    sample_column_info.is_numeric = True
    result = analyze_column_data_quality(
        sample_df,
        "numeric",
        sample_column_info,
        outlier_zscore_threshold=3.0,
        low_variance_threshold=0.01,
    )
    assert hasattr(result, "outlier_count")
    assert hasattr(result, "has_low_variation")


def test_analyze_column_with_settings(
    sample_df: pl.DataFrame, sample_column_info: ColumnInfo
) -> None:
    """Test the analyze_column function with custom settings."""
    settings = ColumnAnalysisSettings(
        outlier_zscore_threshold=2.0,
        low_variance_threshold=0.02,
        default_is_feature_var=True,
    )
    ci = ColumnInfo(
        name="numeric",
        type=ColumnType.UNKNOWN,
        is_numeric=True,
        is_categorical=False,
        is_temporal=False,
    )
    ci = ci.model_copy(update=get_type_for_column(sample_df, "numeric"))
    result = analyze_column(
        sample_df,
        "numeric",
        analysis_settings=settings,
        column_info=ci,
    )
    assert result.type == ColumnType.INT
    assert result.is_numeric


def test_analyze_column_with_df_info(
    sample_df: pl.DataFrame,
) -> None:
    """Test the analyze_column function with DataFrameInfo."""
    ci = ColumnInfo(
        name="numeric",
        type=ColumnType.UNKNOWN,
        is_numeric=True,
        is_categorical=False,
        is_temporal=False,
    )
    ci = ci.model_copy(update=get_type_for_column(sample_df, "numeric"))
    df_info = DataFrameInfo(
        num_rows=5,
        num_cols=4,
        column_info=[ci],
    )
    result = analyze_column(
        sample_df,
        "numeric",
        df_info=df_info,
    )
    assert result.type == ColumnType.INT
    assert result.is_numeric
