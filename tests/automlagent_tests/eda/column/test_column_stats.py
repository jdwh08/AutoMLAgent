#####################################################
# AutoMLAgent [EDA COLUMN STATS TESTS]
# ####################################################
# Jonathan Wang

# ABOUT:
# Unit tests for column-level EDA statistics functions in column_stats.py

"""Unit tests for column stats."""

#####################################################
### IMPORTS

import datetime
from collections.abc import Callable

import numpy as np  # NOTE(jdwh08): used as double check for polars operations
import polars as pl
import pytest
from scipy.stats import kurtosis, skew  # type: ignore[import-untyped]

### OWN MODULES
from automlagent.eda.column.column_stats import (
    get_numerical_stats_for_column,
    get_string_stats_for_column,
    get_temporal_stats_for_column,
)

#####################################################
### TESTS


class TestColumnStats:
    def test_get_numerical_stats_for_column_expected(self) -> None:
        """Test numerical stats for a normal numeric column."""
        arr = [1, 2, 3, 4, 5]
        df = pl.DataFrame({"num": arr}, schema=[("num", pl.Int64)])
        result = get_numerical_stats_for_column(df, "num")
        assert result["min"] == np.min(arr)
        assert result["max"] == np.max(arr)
        assert result["mean"] == np.mean(arr)
        assert result["median"] == np.median(arr)
        assert result["std"] == np.std(arr, ddof=1)
        assert result["p5"] == np.quantile(arr, 0.05)
        assert result["q1"] == np.quantile(arr, 0.25)
        assert result["q3"] == np.quantile(arr, 0.75)
        assert result["p95"] == np.quantile(arr, 0.95)
        assert result["skewness"] == skew(arr)
        assert result["kurtosis"] == kurtosis(arr)

    def test_get_numerical_stats_for_column_with_nan(self) -> None:
        """Test numerical stats when column contains NaN values."""
        # NOTE(jdwh08): We have ignore nan like polars.
        df = pl.DataFrame({"num": [1.0, float("nan"), 3.0]})
        result = get_numerical_stats_for_column(df, "num")
        assert result["mean"] == pytest.approx(2.0, nan_ok=False)

    def test_get_string_stats_for_column_expected(self) -> None:
        """Test string stats for a column of string lengths."""
        arr = [1, 2, 3, 4]
        arr_chars = ["a", "b", "c", "d"]
        col = [char * length for char, length in zip(arr_chars, arr, strict=True)]
        df = pl.DataFrame({"slen": col})
        result = get_string_stats_for_column(df, "slen")
        assert result["char_length_mean"] == np.mean(arr)
        assert result["char_length_min"] == np.min(arr)
        assert result["char_length_max"] == np.max(arr)
        assert result["char_length_std"] == np.std(arr, ddof=1)

    def test_get_temporal_stats_for_column_expected(self) -> None:
        """Test temporal stats for a column of datetimes."""
        dates = [datetime.date(2023, 1, i + 1) for i in range(5)]
        df = pl.DataFrame({"dt": dates})
        result = get_temporal_stats_for_column(df, "dt")
        assert result["temporal_min"] == dates[0]
        assert result["temporal_max"] == dates[-1]
        assert result["temporal_diff"] == datetime.timedelta(days=4)

    def test_get_temporal_stats_for_column_min_equals_max(self) -> None:
        """Test temporal stats when all values are the same."""
        date = datetime.date(2023, 1, 1)
        df = pl.DataFrame({"dt": [date] * 3})
        result = get_temporal_stats_for_column(df, "dt")
        assert result["temporal_min"] == date
        assert result["temporal_max"] == date
        assert result["temporal_diff"] == datetime.timedelta(days=0)

    @pytest.mark.parametrize(
        ("func", "df", "column", "exc"),
        [
            (get_numerical_stats_for_column, pl.DataFrame({"a": []}), "a", Exception),
            (get_string_stats_for_column, pl.DataFrame({"a": []}), "a", Exception),
            (get_temporal_stats_for_column, pl.DataFrame({"a": []}), "a", Exception),
        ],
    )
    def test_stats_functions_empty_column_raises(
        self,
        func: Callable[[pl.DataFrame, str], dict[str, float]],
        df: pl.DataFrame,
        column: str,
        exc: type[Exception],
    ) -> None:
        """Test that stats functions raise or error on empty columns."""
        with pytest.raises(exc):
            func(df, column)
