"""Unit tests for column histogram functionality."""

import polars as pl
import pytest

from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.column_type import ColumnType
from automlagent.eda.column.column_histogram import get_histogram_for_column


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a sample dataframe for testing."""
    return pl.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
            "categorical": ["a", "b", "a", "c", "b"],
            "empty": [None, None, None, None, None],
        }
    )


@pytest.fixture
def sample_column_info() -> ColumnInfo:
    """Create a sample column info for testing."""
    return ColumnInfo(
        name="numeric",
        type=ColumnType.FLOAT,
        is_numeric=True,
        is_categorical=False,
        is_temporal=False,
    )


class TestColumnHistogram:
    def test_get_histogram_for_column_numerical(self, sample_df: pl.DataFrame) -> None:
        """Test histogram generation for numerical column."""
        result = get_histogram_for_column(sample_df, "numeric")
        assert result is not None
        assert "histogram" in result
        histogram = result["histogram"]
        assert all(k.startswith("[") and k.endswith("]") for k in histogram)
        assert all(isinstance(v, int) for v in histogram.values())

    def test_get_histogram_for_column_categorical(
        self, sample_df: pl.DataFrame
    ) -> None:
        """Test histogram generation for categorical column."""
        result = get_histogram_for_column(sample_df, "categorical")
        assert result is not None
        assert "histogram" in result
        histogram = result["histogram"]
        assert all(isinstance(k, str) for k in histogram)
        assert all(isinstance(v, int) for v in histogram.values())

    def test_get_histogram_for_column_empty(self, sample_df: pl.DataFrame) -> None:
        """Test histogram generation for empty column."""
        result = get_histogram_for_column(sample_df, "empty")
        assert result is not None
        assert "histogram" in result
        assert len(result["histogram"]) == 1

    def test_get_histogram_for_column_with_column_info(
        self,
        sample_df: pl.DataFrame,
        sample_column_info: ColumnInfo,
    ) -> None:
        """Test histogram generation with provided ColumnInfo."""
        result = get_histogram_for_column(
            sample_df,
            "numeric",
            column_info=sample_column_info,
        )
        assert result is not None
        assert "histogram" in result
        histogram = result["histogram"]
        assert all(k.startswith("[") and k.endswith("]") for k in histogram)
        assert all(isinstance(v, int) for v in histogram.values())

    def test_get_histogram_for_column_invalid_column(
        self, sample_df: pl.DataFrame
    ) -> None:
        """Test histogram generation with invalid column name."""
        with pytest.raises(KeyError):
            get_histogram_for_column(sample_df, "nonexistent")

    def test_get_histogram_for_column_null_values(self) -> None:
        """Test histogram generation with null values."""
        df = pl.DataFrame(
            {
                "with_nulls": [1.0, None, 3.0, None, 5.0],
            }
        )
        result = get_histogram_for_column(df, "with_nulls")
        assert result is not None
        assert "histogram" in result
        histogram = result["histogram"]
        assert all(k.startswith("[") and k.endswith("]") for k in histogram)
        assert all(isinstance(v, int) for v in histogram.values())

    def test_get_histogram_for_column_edge_bins(self, sample_df: pl.DataFrame) -> None:
        """Test that edge bins are properly created."""
        result = get_histogram_for_column(sample_df, "numeric")
        assert result is not None
        histogram = result["histogram"]

        # Check for -inf bin
        inf_bins = [k for k in histogram if k.startswith("[-inf")]
        assert len(inf_bins) == 1

        # Check for +inf bin
        inf_bins = [k for k in histogram if k.endswith(",inf]")]
        assert len(inf_bins) == 1
