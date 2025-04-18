#####################################################
# AutoMLAgent [EDA COLUMN QUALITY TESTS]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Column-level quality tests."""

#####################################################
### BOARD

# TODO(jdwh08): Add tests for partially None values
# TODO(jdwh08): Add tests for partially NaN values

#####################################################
### IMPORTS

import polars as pl
import pytest

from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.column_type import ColumnType
from automlagent.eda.column.column_quality import (
    column_data_quality_categorical_handler,
    column_data_quality_numeric,
    column_data_quality_temporal_handler,
    get_data_quality_for_column,
)


#####################################################
### UNIT TESTS
class TestUnitColumnQuality:
    @pytest.fixture
    def basic_float_df(self) -> pl.DataFrame:
        return pl.DataFrame({"col": [1.0, 2.0, 3.0, 4.0, 5.0]})

    def test_numeric_no_outliers_no_low_variation(
        self, basic_float_df: pl.DataFrame
    ) -> None:
        res = column_data_quality_numeric(
            basic_float_df,
            "col",
            outlier_zscore_threshold=3.0,
            low_variance_threshold=0.01,
        )
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0
        assert res["has_low_variation"] is False

    def test_numeric_all_nulls_returns_zero_and_low_variation(self) -> None:
        df = pl.DataFrame({"col": [None, None, None]})
        res = column_data_quality_numeric(df, "col")
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0
        assert res["has_low_variation"] is True

    def test_numeric_custom_thresholds_detect_outlier_and_variation(self) -> None:
        df = pl.DataFrame({"col": [1.0, 1.0, 1.0, 10.0]})
        res = column_data_quality_numeric(
            df, "col", outlier_zscore_threshold=1.0, low_variance_threshold=1.0
        )
        # Expect one outlier (10.0)
        assert res["outlier_count"] == 1
        assert pytest.approx(res["outlier_rate"], rel=1e-3) == 1.0 / 4.0
        # Variance ~20.25 > 1.0 => no low variation
        assert res["has_low_variation"] is False

    def test_numeric_with_column_info_overrides(self) -> None:
        ci = ColumnInfo(name="col", missing_count=1, mean=2.5, std=0.5)
        df = pl.DataFrame({"col": [2.0, 3.0]})
        res = column_data_quality_numeric(
            df,
            "col",
            column_info=ci,
            outlier_zscore_threshold=1.0,
            low_variance_threshold=0.1,
        )
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0
        # Variance = 0.25 > 0.1 => False
        assert res["has_low_variation"] is False

    def test_categorical_no_info_low_variation_false(self) -> None:
        df = pl.DataFrame({"col": ["a", "b", "c", "a"]})
        res = column_data_quality_categorical_handler(df, "col")
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0
        assert res["has_low_variation"] is False

    def test_categorical_uniform_true(self) -> None:
        df = pl.DataFrame({"col": ["x"] * 10})
        res = column_data_quality_categorical_handler(df, "col")
        assert res["has_low_variation"] is True
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0

    def test_categorical_with_column_info_override(self) -> None:
        ci = ColumnInfo(name="col")
        ci.category_counts = {"a": 8, "b": 2}
        df = pl.DataFrame({"col": ["a"] * 8 + ["b"] * 2})
        res = column_data_quality_categorical_handler(df, "col", column_info=ci)
        assert res["has_low_variation"] is False
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0

    def test_temporal_outliers_and_variation(self) -> None:
        dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2030-01-01"]
        df = pl.DataFrame({"col": dates})
        df = df.with_columns(pl.col("col").str.to_date().alias("col"))
        res = column_data_quality_temporal_handler(df, "col")
        assert res["outlier_count"] == 1
        assert pytest.approx(res["outlier_rate"], rel=1e-3) == 1.0 / 4.0
        assert res["has_low_variation"] is False

    def test_temporal_uniform_true(self) -> None:
        dates = ["2021-06-01"] * 5
        df = pl.DataFrame({"col": dates})
        df = df.with_columns(pl.col("col").str.to_date().alias("col"))
        res = column_data_quality_temporal_handler(df, "col")
        assert res["outlier_count"] == 0
        assert pytest.approx(res["outlier_rate"], rel=1e-3) == 0.0
        assert res["has_low_variation"] is True

    def test_dispatch_with_column_info_type(self) -> None:
        """Test to ensure multiple types can work with get_data_quality_for_column."""
        df = pl.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "cat": ["a", "a", "b"],
                "dt": [
                    "2020-01-01 14:01:33",
                    "2020-01-02 18:28:28",
                    "2020-01-03 03:59:07",
                ],
            }
        )
        df = df.with_columns(pl.col("dt").str.to_datetime().alias("dt"))
        ci_num = ColumnInfo(name="num", type=ColumnType.INTEGER)
        ci_cat = ColumnInfo(name="cat", type=ColumnType.CATEGORICAL)
        ci_dt = ColumnInfo(name="dt", type=ColumnType.DATETIME)
        out_num = get_data_quality_for_column(df, "num", column_info=ci_num)
        out_cat = get_data_quality_for_column(df, "cat", column_info=ci_cat)
        out_dt = get_data_quality_for_column(df, "dt", column_info=ci_dt)
        assert "outlier_count" in out_num
        assert "has_low_variation" in out_cat
        assert "outlier_rate" in out_dt

    def test_dispatch_unknown_type_returns_empty(self) -> None:
        """Test to ensure unknown type dispatches empty dict."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        ci = ColumnInfo(name="x", type=ColumnType.UNKNOWN)
        assert get_data_quality_for_column(df, "x", column_info=ci) == {}
