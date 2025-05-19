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

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.column_type import ColumnType
from automlagent.eda.column.column_quality import (
    column_data_quality_categorical_handler,
    column_data_quality_missing_check,
    column_data_quality_missing_inf,
    column_data_quality_numeric,
    column_data_quality_temporal_handler,
    get_data_quality_for_column,
)
from automlagent.eda.column.column_types import get_type_for_column


#####################################################
### UNIT TESTS
class TestUnitColumnQuality:
    @pytest.fixture
    def basic_float_df(self) -> pl.DataFrame:
        return pl.DataFrame({"col": [1.0, 2.0, 3.0, 4.0, 5.0]})

    def test_column_name_not_in_df(
        self,
        basic_float_df: pl.DataFrame,
    ) -> None:
        ci = ColumnInfo(name="col2")
        with pytest.raises(KeyError, match="Column 'col2' not found in DataFrame."):
            column_data_quality_missing_check(basic_float_df, "col2", column_info=ci)

    def test_column_all_missing(
        self,
    ) -> None:
        missing_arr = [None] * 3
        df = pl.DataFrame({"col": missing_arr})
        res = column_data_quality_missing_inf(df, "col")
        assert res == {
            "missing_count": len(missing_arr),
            "missing_rate": 1.0,
            "inf_count": 0,
        }

    def test_column_some_missing_inf(
        self,
    ) -> None:
        missing_arr = [None, None, None, 3.1415926, 2.7182818, float("inf")]
        df = pl.DataFrame({"col": missing_arr})
        res = column_data_quality_missing_inf(df, "col")
        assert res == {
            "missing_count": 3,
            "missing_rate": 0.5,
            "inf_count": 1,
        }

    def test_column_all_missing_check(
        self,
    ) -> None:
        missing_arr = [None, None, None] * 3
        df = pl.DataFrame({"col": missing_arr})
        ci = ColumnInfo(name="col")
        res = column_data_quality_missing_check(df, "col", column_info=ci)
        assert res == {
            "outlier_count": 0,
            "outlier_rate": 0.0,
            "has_low_variation": True,
        }

    def test_numeric_non_numeric_column(self) -> None:
        """Covers fallback/exception path at end of column_data_quality_numeric."""
        df = pl.DataFrame({"col": ["a", "b", "c"]})
        ci = ColumnInfo(name="col", type=ColumnType.INT)
        res = column_data_quality_numeric(df, "col", column_info=ci)
        assert res == {}

    def test_numeric_get_mean_and_std_none(self) -> None:
        """Covers exception when mean or std is None in _get_mean_and_std."""
        df = pl.DataFrame({"col": [None, None, None]})
        # Patch ColumnInfo to have None mean/std
        ci = ColumnInfo(name="col", mean=None, std=None)
        res = column_data_quality_numeric(df, "col", column_info=ci)
        assert res == {
            "has_low_variation": True,
            "outlier_count": 0,
            "outlier_rate": 0.0,
        }

    def test_numeric_no_outliers_no_low_variation(
        self, basic_float_df: pl.DataFrame
    ) -> None:
        ci = ColumnInfo(name="col", type=ColumnType.FLOAT)
        res = column_data_quality_numeric(
            basic_float_df,
            "col",
            column_info=ci,
            outlier_zscore_threshold=3.0,
            low_variance_threshold=0.01,
        )
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0
        assert res["has_low_variation"] is False

    def test_numeric_all_nulls_returns_zero_and_low_variation(self) -> None:
        df = pl.DataFrame({"col": [None, None, None]})
        ci = ColumnInfo(name="col", type=ColumnType.FLOAT)
        res = column_data_quality_numeric(df, "col", column_info=ci)
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0
        assert res["has_low_variation"] is True

    def test_numeric_partial_missing_values(self) -> None:
        """Test numeric quality analysis with mixed missing values."""
        df = pl.DataFrame({"col": [1.0, None, 2.0, float("nan"), 3.0, 999.0, None]})
        ci = ColumnInfo(name="col", type=ColumnType.FLOAT)
        res = column_data_quality_numeric(df, "col", column_info=ci)
        # Should detect inf as outlier
        assert res["outlier_count"] == 1
        assert pytest.approx(res["outlier_rate"], rel=1e-3) == 1.0 / 7.0
        # Should not be low variation due to spread of values
        assert res["has_low_variation"] is False

    def test_numeric_custom_thresholds_detect_outlier_and_variation(self) -> None:
        df = pl.DataFrame({"col": [1.0, 1.0, 1.0, 10.0]})
        ci = ColumnInfo(name="col", type=ColumnType.FLOAT)
        res = column_data_quality_numeric(
            df,
            "col",
            column_info=ci,
            outlier_zscore_threshold=1.0,
            low_variance_threshold=1.0,
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
        ci = ColumnInfo(name="col", type=ColumnType.CATEGORICAL)
        res = column_data_quality_categorical_handler(df, "col", column_info=ci)
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0
        assert res["has_low_variation"] is False

    def test_categorical_uniform_true(self) -> None:
        df = pl.DataFrame({"col": ["x"] * 10})
        ci = ColumnInfo(name="col", type=ColumnType.CATEGORICAL)
        res = column_data_quality_categorical_handler(df, "col", column_info=ci)
        assert res["has_low_variation"] is True
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0

    def test_categorical_with_column_info_override(self) -> None:
        ci = ColumnInfo(name="col")
        ci.histogram = {"a": 8, "b": 2}
        df = pl.DataFrame({"col": ["a"] * 8 + ["b"] * 2})
        res = column_data_quality_categorical_handler(df, "col", column_info=ci)
        assert res["has_low_variation"] is False
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0

    def test_categorical_partial_missing_values(self) -> None:
        """Test categorical quality analysis with mixed missing values."""
        col_array = ["a", None, "a", "a", None, "c", "a", "b"] + ["a"] * 50
        df = pl.DataFrame({"col": col_array})
        ci = ColumnInfo(name="col", type=ColumnType.CATEGORICAL)
        res = column_data_quality_categorical_handler(df, "col", column_info=ci)
        # No outliers in categorical data
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0
        # Should be low variation as 'a' is majority
        assert res["has_low_variation"] is True

    def test_temporal_outliers_and_variation(self) -> None:
        dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2030-10-10"]
        df = pl.DataFrame({"col": dates})
        df = df.with_columns(pl.col("col").str.to_date().alias("col"))
        ci = ColumnInfo(name="col")
        res = column_data_quality_temporal_handler(df, "col", column_info=ci)
        assert res["outlier_count"] == 1
        assert pytest.approx(res["outlier_rate"], rel=1e-3) == 1.0 / 4.0
        assert res["has_low_variation"] is False

    def test_temporal_datetime_with_timezone(self) -> None:
        """Test temporal quality analysis with timezone-aware datetimes."""
        dates = [
            "2020-01-01 12:00:00+00:00",
            "2020-01-02 12:00:00+00:00",
            "2020-01-03 12:00:00+00:00",
            "2030-01-01 12:00:00+00:00",  # Outlier
        ]
        df = pl.DataFrame({"col": dates})
        df = df.with_columns(
            pl.col("col")
            .str.to_datetime(time_unit="us", time_zone="Antarctica/Troll")  # :3
            .alias("col")
        )
        ci = ColumnInfo(name="col")
        ci = ci.model_copy(
            update=get_type_for_column(df=df, column_name="col", column_info=ci)
        )
        res = column_data_quality_temporal_handler(df, "col", column_info=ci)
        assert res["outlier_count"] == 1
        assert pytest.approx(res["outlier_rate"], rel=1e-3) == 1.0 / 4.0
        assert res["has_low_variation"] is False

    def test_temporal_uniform_true(self) -> None:
        dates = ["2021-06-01"] * 5
        df = pl.DataFrame({"col": dates})
        df = df.with_columns(pl.col("col").str.to_date().alias("col"))
        ci = ColumnInfo(name="col")
        ci = ci.model_copy(
            update=get_type_for_column(df=df, column_name="col", column_info=ci)
        )
        res = column_data_quality_temporal_handler(df, "col", column_info=ci)
        assert res["outlier_count"] == 0
        assert pytest.approx(res["outlier_rate"], rel=1e-3) == 0.0
        assert res["has_low_variation"] is True

    def test_temporal_partial_missing_values(self) -> None:
        """Test temporal quality analysis with mixed missing values."""
        dates = [
            "2020-01-01",
            None,
            "2020-01-02",
            "2020-01-03",
            None,
            "2030-01-01",  # Outlier
        ]
        df = pl.DataFrame({"col": dates})
        df = df.with_columns(pl.col("col").str.to_date().alias("col"))
        ci = ColumnInfo(name="col", type=ColumnType.DATE)
        ci = ci.model_copy(
            update=get_type_for_column(df=df, column_name="col", column_info=ci)
        )
        res = column_data_quality_temporal_handler(df, "col", column_info=ci)
        # Should detect 2030 date as outlier
        assert res["outlier_count"] == 1
        assert pytest.approx(res["outlier_rate"], rel=1e-3) == 1.0 / 6.0
        # Should not be low variation due to spread of dates
        assert res["has_low_variation"] is False

    def test_temporal_partial_missing_with_timezone(self) -> None:
        """Test temporal quality analysis with mixed missing values and timezone."""
        dates = [
            "2020-01-01 12:00:00+00:00",
            None,
            "2020-01-02 12:00:00+00:00",
            "2020-01-03 12:00:00+00:00",
            None,
            "2030-01-01 12:00:00+00:00",  # Outlier
        ]
        df = pl.DataFrame({"col": dates})
        df = df.with_columns(
            pl.col("col").str.to_datetime(time_unit="us", time_zone="UTC").alias("col")
        )
        ci = ColumnInfo(name="col", type=ColumnType.DATETIME)
        ci = ci.model_copy(
            update=get_type_for_column(df=df, column_name="col", column_info=ci)
        )
        res = column_data_quality_temporal_handler(df, "col", column_info=ci)
        # Should detect 2030 datetime as outlier
        assert res["outlier_count"] == 1
        assert pytest.approx(res["outlier_rate"], rel=1e-3) == 1.0 / 6.0
        # Should not be low variation due to spread of datetimes
        assert res["has_low_variation"] is False

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
        ci_num = ColumnInfo(name="num", type=ColumnType.INT)
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

    def test_categorical_all_nulls(self) -> None:
        """Covers categorical handler with all nulls."""
        df = pl.DataFrame({"col": [None, None, None]})
        ci = ColumnInfo(name="col", type=ColumnType.CATEGORICAL)
        res = column_data_quality_categorical_handler(df, "col", column_info=ci)
        assert res["has_low_variation"] is True
        assert res["outlier_count"] == 0
        assert res["outlier_rate"] == 0.0

    def test_categorical_broken_column_info(self) -> None:
        """Covers categorical handler with malformed ColumnInfo."""
        ci = ColumnInfo(name="col", type=ColumnType.CATEGORICAL)
        ci.histogram = {}  # Empty histogram instead of None
        df = pl.DataFrame({"col": ["a", "b", "c"]})
        res = column_data_quality_categorical_handler(df, "col", column_info=ci)
        assert isinstance(res, dict)

    def test_temporal_non_convertible_column(self) -> None:
        """Covers temporal handler conversion error."""
        df = pl.DataFrame({"col": ["not_date", "also_not_a_date"]})
        ci = ColumnInfo(name="col", type=ColumnType.DATETIME)
        res = column_data_quality_temporal_handler(df, "col", column_info=ci)
        assert isinstance(res, dict)

    def test_temporal_all_nulls(self) -> None:
        """Covers temporal handler with all nulls."""
        df = pl.DataFrame({"col": [None, None, None]})
        ci = ColumnInfo(name="col", type=ColumnType.DATETIME)
        res = column_data_quality_temporal_handler(df, "col", column_info=ci)
        assert isinstance(res, dict)

    def test_temporal_one_unique(self) -> None:
        """Covers temporal handler unique_count <= 1."""
        df = pl.DataFrame({"col": ["2022-01-01"] * 5})
        df = df.with_columns(pl.col("col").str.to_date().alias("col"))
        ci = ColumnInfo(name="col", type=ColumnType.DATE)
        res = column_data_quality_temporal_handler(df, "col", column_info=ci)
        assert res["has_low_variation"] is True

    def test_numeric_handler_missing_column(self) -> None:
        """Covers numeric handler edge case."""
        ci = ColumnInfo(name="not_in_df", type=ColumnType.FLOAT)
        df = pl.DataFrame({"col": [1.0, 2.0]})
        with pytest.raises(KeyError):
            get_data_quality_for_column(df, "not_in_df", column_info=ci)
