"""Unit tests for histogram utilities."""

import polars as pl
import pytest

from automlagent.utils.histogram_utils import (
    create_histogram,
    freedman_diaconis_bins,
    sturges_bins,
)


@pytest.fixture
def sample_numerical_df() -> pl.DataFrame:
    """Create a sample numerical DataFrame."""
    return pl.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
            "categorical": ["a", "b", "a", "c", "b"],
        }
    )


@pytest.fixture
def sample_numerical_series() -> pl.Series:
    """Create a sample numerical Series."""
    return pl.Series("numeric", [1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_categorical_series() -> pl.Series:
    """Create a sample categorical Series."""
    return pl.Series("categorical", ["a", "b", "a", "c", "b"])


class TestHistogramUtils:
    def test_freedman_diaconis_bins(self, sample_numerical_series: pl.Series) -> None:
        """Test Freedman-Diaconis bin calculation."""
        num_bins = freedman_diaconis_bins(sample_numerical_series)
        assert isinstance(num_bins, int)
        assert num_bins > 0

    def test_sturges_bins(self, sample_numerical_series: pl.Series) -> None:
        """Test Sturges bin calculation."""
        num_bins = sturges_bins(sample_numerical_series)
        assert isinstance(num_bins, int)
        assert num_bins > 0

    def test_create_histogram_numerical_df(
        self, sample_numerical_df: pl.DataFrame
    ) -> None:
        """Test histogram creation from numerical DataFrame."""
        histogram = create_histogram(sample_numerical_df, "numeric")
        assert histogram is not None
        assert all(isinstance(k, tuple) for k in histogram)
        assert all(isinstance(v, int) for v in histogram.values())

    def test_create_histogram_numerical_series(
        self,
        sample_numerical_series: pl.Series,
    ) -> None:
        """Test histogram creation from numerical Series."""
        histogram = create_histogram(sample_numerical_series)
        assert histogram is not None
        assert all(isinstance(k, tuple) for k in histogram)
        assert all(isinstance(v, int) for v in histogram.values())

    def test_create_histogram_categorical_df(
        self, sample_numerical_df: pl.DataFrame
    ) -> None:
        """Test histogram creation from categorical DataFrame."""
        histogram = create_histogram(sample_numerical_df, "categorical")
        assert histogram is not None
        assert all(isinstance(k, str) for k in histogram)
        assert all(isinstance(v, int) for v in histogram.values())

    def test_create_histogram_categorical_series(
        self,
        sample_categorical_series: pl.Series,
    ) -> None:
        """Test histogram creation from categorical Series."""
        histogram = create_histogram(sample_categorical_series)
        assert histogram is not None
        assert all(isinstance(k, str) for k in histogram)
        assert all(isinstance(v, int) for v in histogram.values())

    def test_create_histogram_custom_bin_calculator(
        self,
        sample_numerical_series: pl.Series,
    ) -> None:
        """Test histogram creation with custom bin calculator."""
        num_custom_bins = 10

        def custom_bins(data: pl.Series) -> int:
            return num_custom_bins

        histogram = create_histogram(
            sample_numerical_series,
            bin_calculator=custom_bins,
        )
        assert histogram is not None
        assert len(histogram) == num_custom_bins + 2  # 10 bins + 2 edge bins

    def test_create_histogram_empty_series(self) -> None:
        """Test histogram creation with empty Series."""
        empty_series = pl.Series("empty", [])
        histogram = create_histogram(empty_series)
        assert histogram is not None
        assert len(histogram) == 0

    def test_create_histogram_null_values(self) -> None:
        """Test histogram creation with null values."""
        series_with_nulls = pl.Series("with_nulls", [1.0, None, 3.0, None, 5.0])
        histogram = create_histogram(series_with_nulls)
        assert histogram is not None
        assert all(isinstance(k, tuple) for k in histogram)
        assert all(isinstance(v, int) for v in histogram.values())

    def test_create_histogram_invalid_column(self) -> None:
        """Test histogram creation with invalid column."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        with pytest.raises(KeyError):
            create_histogram(df, "nonexistent")

    def test_create_histogram_missing_column_name(self) -> None:
        """Test histogram creation with missing column name for DataFrame."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        with pytest.raises(ValueError, match="column_name"):
            create_histogram(df)

    def test_create_histogram_edge_bins(
        self, sample_numerical_series: pl.Series
    ) -> None:
        """Test that edge bins are properly created."""
        histogram = create_histogram(sample_numerical_series)
        assert histogram is not None

        # Check for -inf bin
        inf_bins = [k for k in histogram if k[0] == float("-inf")]
        assert len(inf_bins) == 1

        # Check for +inf bin
        inf_bins = [k for k in histogram if k[1] == float("inf")]
        assert len(inf_bins) == 1
