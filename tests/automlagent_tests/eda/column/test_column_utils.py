#####################################################
# AutoMLAgent [EDA COLUMN UTILS TESTS]
# ####################################################
# Jonathan Wang

# ABOUT:
# Unit tests for column utility functions in column_utils.py

"""Unit tests for column_utils.py."""

#####################################################
### IMPORTS

import polars as pl
import polars.testing as pt
import pytest

### OWN MODULES
from automlagent.eda.column.column_utils import column_filter_out_missing

#####################################################
### TESTS


class TestColumnUtils:
    def test_filter_numeric_removes_none_and_nan(self) -> None:
        """Test that None and NaN are removed from numeric columns."""
        df = pl.DataFrame({"a": [1.0, None, 2.0, float("nan"), 3.0]})
        result = column_filter_out_missing(df, "a")
        expected = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        pt.assert_frame_equal(result, expected)

    def test_column_not_found_raises(self) -> None:
        """Test that KeyError is raised if column does not exist."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(KeyError):
            column_filter_out_missing(df, "b")

    def test_no_missing_values_returns_identical(self) -> None:
        """Test that DataFrame is unchanged if no missing values present."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = column_filter_out_missing(df, "a")
        pt.assert_frame_equal(result, df)

    def test_all_missing_returns_empty(self) -> None:
        """Test that all missing values returns empty DataFrame."""
        df = pl.DataFrame({"a": [None, None, None]})
        result = column_filter_out_missing(df, "a")
        expected = pl.DataFrame({"a": []}, schema=[("a", pl.Null)])
        assert result.shape == (0, 1)
        assert result.schema == expected.schema

    @pytest.mark.parametrize(
        ("data", "col", "expected"),
        [
            ([None, None], "both_none", 0),
            ([1, 2, 3], "no_none", 3),
            (["1", None, "3"], "some_none", 2),
            ([1.0, float("nan"), 3.0], "some_nan", 2),
            ([float("nan"), float("nan"), None], "all_none_or_nan", 0),
        ],
    )
    def test_parametrized_missing_cases(
        self, data: list[int], col: str, expected: int
    ) -> None:
        """Parametrized: various missing value scenarios."""
        df = pl.DataFrame({col: data})
        result = column_filter_out_missing(df, col)
        assert result.shape[0] == expected
