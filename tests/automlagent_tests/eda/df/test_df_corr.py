#####################################################
# AutoMLAgent [EDA DF CORR TESTS]
#####################################################
# Jonathan Wang

# ABOUT:
# Unit tests for dataframe-level correlation functions in df_corr.py

"""Unit tests for df_corr.py."""

#####################################################
### IMPORTS

import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.df_info import DataFrameInfo
from automlagent.eda.df.df_corr import (
    get_correlation_for_df,
    get_pearson_correlation_for_df,
)


#####################################################
### FIXTURES
def make_df_and_info_numeric() -> tuple[pl.DataFrame, DataFrameInfo]:
    df = pl.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0],
            "a": [2.0, 4.0, 6.0, 8.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        }
    )
    cols = [
        ColumnInfo(
            name="target", is_numeric=True, is_target_var=True, is_feature_var=False
        ),
        ColumnInfo(name="a", is_numeric=True),
        ColumnInfo(name="b", is_numeric=True),
    ]
    df_info = DataFrameInfo(
        num_rows=4,
        num_cols=3,
        column_info=cols,
        correlation_pearson=None,
    )
    return df, df_info


#####################################################
### TESTS


class TestDfCorr:
    def test_get_pearson_correlation_expected(self) -> None:
        """Test Pearson correlation matrix for all numeric columns."""
        df, df_info = make_df_and_info_numeric()
        result = get_pearson_correlation_for_df(df, df_info)
        corr = result.correlation_pearson
        assert corr is not None
        # Should contain only numeric columns (no target logic)
        expected_cols = {"index", "target", "a", "b"}
        assert set(corr.columns) == expected_cols
        # Diagonal should be 1.0
        for col in ["target", "a", "b"]:
            row = corr.filter(pl.col("index") == col)
            assert pytest.approx(row[0, col], abs=1e-6) == 1.0
        # Off-diagonal: check correlation is correct for perfectly correlated columns
        row = corr.filter(pl.col("index") == "target")
        assert pytest.approx(row[0, "a"], abs=1e-6) == 1.0

    def test_get_pearson_correlation_no_numeric(self) -> None:
        """Test warning and unchanged DataFrameInfo if no numeric features."""
        df = pl.DataFrame({"cat": ["x", "y", "z"]})
        cols = [ColumnInfo(name="cat", is_numeric=False)]
        df_info = DataFrameInfo(
            num_rows=3,
            num_cols=1,
            column_info=cols,
            correlation_pearson=None,
        )
        with warnings.catch_warnings(record=True) as w:
            result = get_pearson_correlation_for_df(df, df_info)
            assert result.correlation_pearson is None
            assert any("No numeric features" in str(warn.message) for warn in w)

    def test_get_pearson_correlation_all_non_numeric(self) -> None:
        """Test that correlation is None if all columns are non-numeric."""
        df = pl.DataFrame({"cat": ["a", "b", "c"]})
        cols = [ColumnInfo(name="cat", is_numeric=False)]
        df_info = DataFrameInfo(
            num_rows=3,
            num_cols=1,
            column_info=cols,
            correlation_pearson=None,
        )
        result = get_pearson_correlation_for_df(df, df_info)
        assert result.correlation_pearson is None

    def test_get_pearson_correlation_all_constant(self) -> None:
        """Test that correlation matrix is nan off-diagonal for constant columns."""
        df = pl.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2], "c": [3, 3, 3]})
        cols = [
            ColumnInfo(name="a", is_numeric=True),
            ColumnInfo(name="b", is_numeric=True),
            ColumnInfo(name="c", is_numeric=True),
        ]
        df_info = DataFrameInfo(
            num_rows=3,
            num_cols=3,
            column_info=cols,
            correlation_pearson=None,
        )
        result = get_pearson_correlation_for_df(df, df_info)
        corr = result.correlation_pearson

        # NOTE(jdwh08): EVEN THE DIAGONAL VALUES ARE NaN!
        assert corr is not None
        assert_frame_equal(
            corr,
            pl.DataFrame(
                {
                    "index": ["a", "b", "c"],
                    "a": [float("nan"), float("nan"), float("nan")],
                    "b": [float("nan"), float("nan"), float("nan")],
                    "c": [float("nan"), float("nan"), float("nan")],
                }
            ),
        )

    def test_get_pearson_correlation_empty_df(self) -> None:
        """Test empty DataFrame returns None correlation."""
        df = pl.DataFrame({})
        df_info = DataFrameInfo(
            num_rows=0,
            num_cols=0,
            column_info=[],
            correlation_pearson=None,
        )
        result = get_pearson_correlation_for_df(df, df_info)
        assert result.correlation_pearson is None

    def test_get_correlation_expected(self) -> None:
        """Test correlation value between two columns."""
        df, df_info = make_df_and_info_numeric()
        df_info = get_pearson_correlation_for_df(df, df_info)
        corr = get_correlation_for_df("target", "a", df_info)
        assert pytest.approx(corr, abs=1e-6) == 1.0

    def test_get_correlation_missing_matrix(self) -> None:
        """Test warning and None if correlation matrix not computed."""
        _, df_info = make_df_and_info_numeric()
        with warnings.catch_warnings(record=True) as w:
            result = get_correlation_for_df("target", "a", df_info)
            assert result is None
            assert any("not computed" in str(warn.message) for warn in w)

    def test_get_correlation_missing_column(self) -> None:
        """Test warning and None if column is missing from correlation matrix."""
        df, df_info = make_df_and_info_numeric()
        df_info = get_pearson_correlation_for_df(df, df_info)
        with warnings.catch_warnings(record=True) as w:
            result = get_correlation_for_df("target", "not_a_col", df_info)
            assert result is None
            assert any("not found" in str(warn.message) for warn in w)

    def test_get_pearson_correlation_for_df_expected(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_info = DataFrameInfo(
            num_rows=3,
            num_cols=2,
            column_info=[
                ColumnInfo(name="a", is_numeric=True),
                ColumnInfo(name="b", is_numeric=True),
            ],
            correlation_pearson=None,
        )
        result = get_pearson_correlation_for_df(df, df_info)
        corr = result.correlation_pearson
        assert corr is not None
        # Should contain only numeric columns (no target logic)
        expected_cols = {"index", "a", "b"}
        assert set(corr.columns) == expected_cols
        # Diagonal should be 1.0
        for col in ["a", "b"]:
            row = corr.filter(pl.col("index") == col)
            assert pytest.approx(row[0, col], abs=1e-6) == 1.0
        # Off-diagonal: check correlation is correct for perfectly correlated columns
        row = corr.filter(pl.col("index") == "a")
        assert pytest.approx(row[0, "b"], abs=1e-6) == 1.0

    def test_get_pearson_correlation_for_df_no_numeric(self) -> None:
        df = pl.DataFrame({"a": ["x", "y", "z"], "b": ["1", "2", "3"]})
        df_info = DataFrameInfo(
            num_rows=3,
            num_cols=2,
            column_info=[ColumnInfo(name="a"), ColumnInfo(name="b")],
            correlation_pearson=None,
        )
        result = get_pearson_correlation_for_df(df, df_info)
        assert result.correlation_pearson is None

    def test_get_pearson_correlation_for_df_all_non_numeric(self) -> None:
        df = pl.DataFrame({"a": ["x", "y", "z"], "b": ["1", "2", "3"]})
        df_info = DataFrameInfo(
            num_rows=3,
            num_cols=2,
            column_info=[ColumnInfo(name="a"), ColumnInfo(name="b")],
            correlation_pearson=None,
        )
        result = get_pearson_correlation_for_df(df, df_info)
        assert result.correlation_pearson is None

    def test_get_pearson_correlation_for_df_all_constant(self) -> None:
        df = pl.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
        df_info = DataFrameInfo(
            num_rows=3,
            num_cols=2,
            column_info=[
                ColumnInfo(name="a", is_numeric=True),
                ColumnInfo(name="b", is_numeric=True),
            ],
            correlation_pearson=None,
        )
        result = get_pearson_correlation_for_df(df, df_info)
        corr = result.correlation_pearson
        assert corr is not None
        assert_frame_equal(
            corr,
            pl.DataFrame(
                {
                    "index": ["a", "b"],
                    "a": [float("nan"), float("nan")],
                    "b": [float("nan"), float("nan")],
                }
            ),
        )

    def test_get_pearson_correlation_for_df_empty_df(self) -> None:
        df = pl.DataFrame({"a": [], "b": []})
        df_info = DataFrameInfo(
            num_rows=0,
            num_cols=2,
            column_info=[
                ColumnInfo(name="a", is_numeric=True),
                ColumnInfo(name="b", is_numeric=True),
            ],
            correlation_pearson=None,
        )
        result = get_pearson_correlation_for_df(df, df_info)
        assert_frame_equal(
            result.correlation_pearson,
            pl.DataFrame(
                {
                    "index": ["a", "b"],
                    "a": [float("nan"), float("nan")],
                    "b": [float("nan"), float("nan")],
                }
            ),
        )
