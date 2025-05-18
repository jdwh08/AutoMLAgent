#####################################################
# AutoMLAgent [EDA COLUMN TYPES TESTS]
# ####################################################
# Jonathan Wang

# ABOUT:
# Unit tests for column type inference and dtype utilities.

"""Unit tests for column type inference and dtype utilities."""

####################################################
### IMPORTS

import datetime
from typing import Never

import polars as pl
import pytest
import pytz

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.column_type import ColumnType
from automlagent.eda.column.column_types import (
    ColumnTypeDict,
    _get_cardinality,  # type: ignore[reportPrivateUsage]
    create_column_type_handlers,
    get_type_for_column,
    initialize_output_dict,
)


#####################################################
### UNIT TESTS
class TestUnitColumnTypes:
    @pytest.fixture
    def df_all_types(self) -> pl.DataFrame:
        data = {
            "bool_col": [True, False, True, False],
            "int_col": [1, 2, 3, 4],
            "float_col": [1.1, 2.2, 3.3, 4.4],
            "str_col": ["a", "b", "c", "d"],
            "date_col": pl.date_range(
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 4),
                "1d",
                eager=True,
            ).to_list(),  # NOTE(jdwh08): eager returns Series not Expr
            "datetime_col": pl.datetime_range(
                datetime.datetime(2020, 1, 1, 0, 0, tzinfo=None),  # noqa: DTZ001
                datetime.datetime(2020, 1, 4, 0, 0, tzinfo=None),  # noqa: DTZ001
                "1d",
                eager=True,
            ).to_list(),
            "time_col": [
                datetime.time(1, 0, 0),
                datetime.time(2, 23, 0),
                datetime.time(3, 0, 44),
                datetime.time(4, 0, 0),
            ],
            "datetime_col_with_tz": pl.datetime_range(
                datetime.datetime(
                    2020, 1, 1, 0, 0, tzinfo=pytz.timezone("Antarctica/Troll")
                ),
                datetime.datetime(
                    2020, 1, 4, 0, 0, tzinfo=pytz.timezone("Antarctica/Troll")
                ),
                "1d",
                eager=True,
            ).to_list(),
        }
        output = pl.DataFrame(data)
        return output

    def test_initialize_output_dict_basic(self, df_all_types: pl.DataFrame) -> None:
        """Test output dict is initialized with correct keys and types."""
        col = "int_col"
        out = initialize_output_dict(df_all_types, col, None)
        assert set(out.keys()) == {
            "cardinality",
            "unique_rate",
            "type",
            "is_categorical",
            "is_numeric",
            "is_temporal",
        }
        assert isinstance(out["cardinality"], int)
        assert isinstance(out["unique_rate"], float)
        assert out["unique_rate"] == out["cardinality"] / df_all_types.height
        assert out["type"] == ColumnType.UNKNOWN
        assert out["is_categorical"] is False
        assert out["is_numeric"] is False
        assert out["is_temporal"] is False

    def test_initialize_output_dict_missing_column(
        self, df_all_types: pl.DataFrame
    ) -> None:
        """Test initialize_output_dict handles missing column gracefully."""
        out = initialize_output_dict(df_all_types, "not_a_col", None)
        assert out["cardinality"] == df_all_types.height
        assert out["unique_rate"] == 1.0
        assert out["type"] == ColumnType.UNKNOWN

    def test_initialize_output_dict_empty_df(self) -> None:
        """Test initialize_output_dict with empty DataFrame."""
        df = pl.DataFrame({"col": []})
        out = initialize_output_dict(df, "col", None)
        assert out["cardinality"] == 0
        assert out["unique_rate"] == 0.0
        assert out["type"] == ColumnType.UNKNOWN

    @pytest.mark.parametrize(
        ("col_name", "expected_type"),
        [
            ("bool_col", ColumnType.BOOLEAN),
            ("int_col", ColumnType.INT),
            ("float_col", ColumnType.FLOAT),
            ("date_col", ColumnType.DATE),
            ("datetime_col", ColumnType.DATETIME),
            ("datetime_col_with_tz", ColumnType.DATETIME),
            ("time_col", ColumnType.TIME),
            ("str_col", ColumnType.TEXT),
        ],
    )
    def test_create_column_type_handlers_detects_type(
        self, df_all_types: pl.DataFrame, col_name: str, expected_type: ColumnType
    ) -> None:
        """Test create_column_type_handlers returns correct handler for each dtype."""
        handlers = create_column_type_handlers(
            df_all_types, col_name, df_all_types[col_name].n_unique()
        )
        output: ColumnTypeDict = {
            "type": ColumnType.UNKNOWN,
            "is_categorical": False,
            "is_numeric": False,
            "is_temporal": False,
        }
        for predicate, handler in handlers:
            if predicate():
                handler(output)
                break
        assert output["type"] == expected_type

    def test_create_column_type_handlers_high_cardinality(
        self, df_all_types: pl.DataFrame
    ) -> None:
        """Test categorical flag is False for high cardinality columns."""
        col_name = "int_col"
        cardinality = 1000
        handlers = create_column_type_handlers(df_all_types, col_name, cardinality)
        output: ColumnTypeDict = {
            "type": ColumnType.UNKNOWN,
            "is_categorical": False,
            "is_numeric": False,
            "is_temporal": False,
        }
        for predicate, handler in handlers:
            if predicate():
                handler(output)
                break
        assert output["is_categorical"] is False

    @pytest.mark.parametrize(
        ("col_name", "expected_type"),
        [
            ("bool_col", ColumnType.BOOLEAN),
            ("int_col", ColumnType.INT),
            ("float_col", ColumnType.FLOAT),
            ("date_col", ColumnType.DATE),
            ("datetime_col", ColumnType.DATETIME),
            ("time_col", ColumnType.TIME),
            ("str_col", ColumnType.TEXT),
        ],
    )
    def test_get_type_for_column_all_supported_types(
        self, df_all_types: pl.DataFrame, col_name: str, expected_type: ColumnType
    ) -> None:
        """Test get_type_for_column returns correct type for each supported dtype."""
        out: ColumnTypeDict = get_type_for_column(df_all_types, col_name)
        assert out["type"] == expected_type
        assert isinstance(out["cardinality"], int)
        assert out["cardinality"] == df_all_types[col_name].n_unique()
        assert out["unique_rate"] == out["cardinality"] / df_all_types.height

    def test_get_type_for_column_with_column_info(
        self, df_all_types: pl.DataFrame
    ) -> None:
        """Test get_type_for_column uses ColumnInfo if provided."""
        col_name = "int_col"
        ci = ColumnInfo(name=col_name, cardinality=2)
        out: ColumnTypeDict = get_type_for_column(
            df_all_types, col_name, column_info=ci
        )
        assert out["type"] == ColumnType.INT
        assert isinstance(ci.cardinality, int)
        assert out["cardinality"] == ci.cardinality
        assert out["unique_rate"] == ci.cardinality / df_all_types.height

    def test_get_type_for_column_missing_column(
        self, df_all_types: pl.DataFrame
    ) -> None:
        """Test get_type_for_column returns UNKNOWN for missing column."""
        out: ColumnTypeDict = get_type_for_column(df_all_types, "not_a_col")
        assert out["type"] == ColumnType.UNKNOWN
        assert out["cardinality"] == df_all_types.height
        assert out["unique_rate"] == 1.0

    def test_get_type_for_column_all_nulls(self) -> None:
        """Test get_type_for_column handles all-null columns."""
        df = pl.DataFrame({"col": [None, None, None]})
        out: ColumnTypeDict = get_type_for_column(df, "col")
        assert out["type"] == ColumnType.UNKNOWN
        assert out["cardinality"] == 1
        assert out["unique_rate"] == 1 / 3

    def test_get_type_for_column_empty_df(self) -> None:
        """Test get_type_for_column handles empty DataFrame."""
        df = pl.DataFrame({"col": []})
        out: ColumnTypeDict = get_type_for_column(df, "col")
        assert out["type"] == ColumnType.UNKNOWN
        assert out["cardinality"] == 0
        assert out["unique_rate"] == 0

    def test_get_type_for_column_unique_rate_normal(self) -> None:
        """Test get_type_for_column sets correct unique_rate for normal columns."""
        df = pl.DataFrame({"col": [1, 2, 2, 3]})
        out: ColumnTypeDict = get_type_for_column(df, "col")
        assert out["unique_rate"] == 3 / 4

    def test_get_type_for_column_high_cardinality(self) -> None:
        """Test get_type_for_column sets is_categorical False for high cardinality."""
        col_name = "int_col"
        df = pl.DataFrame({col_name: list(range(1000))})
        out = get_type_for_column(df, col_name)
        assert out["is_categorical"] is False

    def test_get_type_for_column_low_cardinality(self) -> None:
        """Test get_type_for_column sets is_categorical True for low cardinality."""
        df = pl.DataFrame({"col": [1, 1, 2, 2]})
        out = get_type_for_column(df, "col")
        assert out["is_categorical"] is True

    def test_get_cardinality_exception_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _get_cardinality fallback logic when n_unique raises exception."""

        class TestingError(Exception):
            pass

        def fail_n_unique() -> Never:
            msg = "FAILED! Dunno why polars allows this for some types but it does."
            raise TestingError(msg)

        df = pl.DataFrame({"col": [1, 2, 3]})
        monkeypatch.setattr(df["col"], "n_unique", fail_n_unique)
        out = _get_cardinality(df, "col", {})
        assert out["cardinality"] == len(df)
        assert out["unique_rate"] == 1.0

    @pytest.fixture
    def df_with_timezones(self) -> pl.DataFrame:
        """Create a DataFrame with datetime columns in different timezones."""
        data = {
            "utc_dt": pl.datetime_range(
                datetime.datetime(2020, 1, 1, 0, 0, tzinfo=datetime.UTC),
                datetime.datetime(2020, 1, 4, 0, 0, tzinfo=datetime.UTC),
                "1d",
                eager=True,
            ).to_list(),
            "antarcticatroll_dt": pl.datetime_range(
                datetime.datetime(
                    2020,
                    1,
                    1,
                    0,
                    0,
                    tzinfo=pytz.timezone("Antarctica/Troll"),
                ),
                datetime.datetime(
                    2020,
                    1,
                    4,
                    0,
                    0,
                    tzinfo=pytz.timezone("Antarctica/Troll"),
                ),
                "1d",
                eager=True,
            )
            .dt.replace_time_zone("Antarctica/Troll")
            .to_list(),
            "naive_dt": pl.datetime_range(
                datetime.datetime(2020, 1, 1, 0, 0),  # noqa: DTZ001
                datetime.datetime(2020, 1, 4, 0, 0),  # noqa: DTZ001
                "1d",
                eager=True,
            ).to_list(),
        }
        return pl.DataFrame(data)

    def test_get_timezone_utc(self, df_with_timezones: pl.DataFrame) -> None:
        """Test timezone detection for UTC datetime column."""
        out = get_type_for_column(df_with_timezones, "utc_dt")
        assert out["type"] == ColumnType.DATETIME
        assert out["is_temporal"] is True
        assert out["timezone"] == "UTC"

    def test_get_timezone_antarcticatroll(
        self, df_with_timezones: pl.DataFrame
    ) -> None:
        """Test timezone detection for Antarctica/Troll datetime column."""
        out = get_type_for_column(df_with_timezones, "antarcticatroll_dt")
        assert out["type"] == ColumnType.DATETIME
        assert out["is_temporal"] is True
        assert out["timezone"] == "Antarctica/Troll"

    def test_get_timezone_naive(self, df_with_timezones: pl.DataFrame) -> None:
        """Test timezone detection for naive datetime column."""
        out = get_type_for_column(df_with_timezones, "naive_dt")
        assert out["type"] == ColumnType.DATETIME
        assert out["is_temporal"] is True
        assert out["timezone"] is None
