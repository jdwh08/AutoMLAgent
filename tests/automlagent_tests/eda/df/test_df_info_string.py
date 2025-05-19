#####################################################
# AutoMLAgent [EDA DF INFO STRING TESTS]
#####################################################
# Jonathan Wang

"""Unit tests for df_info_string."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

#####################################################
### IMPORTS
### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo, create_column_info
from automlagent.dataclass.df_info import DataFrameInfo
from automlagent.eda.df.df_corr import get_pearson_correlation_for_df
from automlagent.eda.df.df_info_string import (
    _info_string_cardinality_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    _info_string_correlation_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    _info_string_duplicates_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    _info_string_missing_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    _info_string_outlier_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    _info_string_shape_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    _info_string_target_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    get_info_string_for_df,
)
from automlagent.eda.eda_column_tool import analyze_column


#####################################################
### HELPERS
@pytest.fixture(scope="module")
def example_polars_df() -> pl.DataFrame:
    seed = 31415926
    generator = np.random.default_rng(seed)
    n = 10
    df = pl.DataFrame(
        {
            "int_col": np.arange(n),
            "float_col": generator.normal(size=n),
            "bool_col": generator.random(size=n) > 0.5,  # noqa: PLR2004 (magic value)
            "str_col": [
                "foo",
                "bar",
                "baz",
                "foo",
                "bar",
                "baz",
                "foo",
                "bar",
                "baz",
                "foo",
            ],
            "cat_col": ["A", "B", "A", "C", "B", "A", "C", "A", "B", "C"],
            "datetime_col": [
                datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)
            ],
            "missing_col": [None, 2, None, 4, 5, None, 7, None, 9, 10],
            "uniform_val_col": [1] * n,
        }
    )

    # Create the data generating process for the target
    df = df.with_columns(
        [
            (
                pl.col("int_col")
                + pl.col("float_col")
                + pl.col("bool_col").mul(generator.beta(0.5, 0.5, size=n))
            ).alias("target_col")
        ]
    )
    df = df.with_columns(
        [
            pl.col("target_col")
            .mul(
                pl.when(pl.col("cat_col") == "B")
                .then(1.2)
                .when(pl.col("cat_col") == "C")
                .then(0.8)
                .otherwise(1.0)
            )
            .alias("target_col")
        ]
    )
    df = df.with_columns(
        [
            pl.col("target_col")
            .mul(pl.when(pl.col("missing_col").is_null()).then(0.01).otherwise(1.0))
            .alias("target_col")
        ]
    )

    return df


@pytest.fixture(scope="module")
def example_column_infos(example_polars_df: pl.DataFrame) -> list[ColumnInfo]:
    infos: list[ColumnInfo] = []
    for col in example_polars_df.columns:
        ci = create_column_info(
            column_name=col,
            is_target_var=(col == "target_col"),
            is_feature_var=(col != "target_col"),
        )
        ci = analyze_column(example_polars_df, column_name=col, column_info=ci)
        infos.append(ci)

    return infos


@pytest.fixture(scope="module")
def example_dataframe_info(
    example_polars_df: pl.DataFrame, example_column_infos: list[ColumnInfo]
) -> DataFrameInfo:
    # Create initial DataFrameInfo
    df_info = DataFrameInfo(
        num_rows=example_polars_df.height,
        num_cols=example_polars_df.width,
        column_info=example_column_infos,
    )
    # Compute correlation using the proper function
    df_info = get_pearson_correlation_for_df(example_polars_df, df_info)
    return df_info


#####################################################
### TESTS


class TestDfInfoString:
    def test_shape_section_typical(self, example_dataframe_info: DataFrameInfo) -> None:
        result = _info_string_shape_section(example_dataframe_info)
        assert (
            f"Rows: {example_dataframe_info.num_rows}, "
            f"Columns: {example_dataframe_info.num_cols}" in result
        )
        # Check for at least one type summary
        assert all(t in result for t in ["int", "float", "bool", "datetime", "text"])

    def test_missing_section(self, example_dataframe_info: DataFrameInfo) -> None:
        result = _info_string_missing_section(example_dataframe_info)
        # Should mention missing_col
        assert "missing_col" in result or "No missing values detected." in result

    def test_missing_section_high_missing(
        self, example_dataframe_info: DataFrameInfo
    ) -> None:
        result = _info_string_missing_section(example_dataframe_info)
        assert "Columns with >30.0% missing" in result

    def test_duplicates_section_always_empty(
        self, example_dataframe_info: DataFrameInfo
    ) -> None:
        result = _info_string_duplicates_section(example_dataframe_info)
        assert result == ""

    def test_target_section_none(
        self, example_column_infos: list[ColumnInfo], example_polars_df: pl.DataFrame
    ) -> None:
        # Remove target flag
        for ci in example_column_infos:
            ci.is_target_var = False
            ci.is_feature_var = True
        info = DataFrameInfo(
            num_rows=example_polars_df.height,
            num_cols=example_polars_df.width,
            column_info=example_column_infos,
        )
        result = _info_string_target_section(info)
        assert result == ""

    def test_target_section_categorical(
        self, example_column_infos: list[ColumnInfo], example_polars_df: pl.DataFrame
    ) -> None:
        # Mark cat_col as target
        for ci in example_column_infos:
            ci.is_target_var = ci.name == "cat_col"
            ci.is_feature_var = not ci.is_target_var
        info = DataFrameInfo(
            num_rows=example_polars_df.height,
            num_cols=example_polars_df.width,
            column_info=example_column_infos,
        )
        result = _info_string_target_section(info)
        assert "categorical" in result or "Target" in result

    def test_target_section_numeric(
        self, example_column_infos: list[ColumnInfo], example_polars_df: pl.DataFrame
    ) -> None:
        # Mark float_col as target
        for ci in example_column_infos:
            ci.is_target_var = ci.name == "float_col"
            ci.is_feature_var = not ci.is_target_var
        info = DataFrameInfo(
            num_rows=example_polars_df.height,
            num_cols=example_polars_df.width,
            column_info=example_column_infos,
        )
        result = _info_string_target_section(info)
        assert "numeric" in result or "Target" in result

    def test_correlation_section_none(
        self, example_column_infos: list[ColumnInfo], example_polars_df: pl.DataFrame
    ) -> None:
        info = DataFrameInfo(
            num_rows=example_polars_df.height,
            num_cols=example_polars_df.width,
            column_info=example_column_infos,
            correlation_pearson=None,
        )
        result = _info_string_correlation_section(info)
        assert result == "No correlation matrix available."

    def test_correlation_section_strong(self, example_polars_df: pl.DataFrame) -> None:
        # Create two perfectly correlated columns
        df = example_polars_df.with_columns([(pl.col("int_col") * 2).alias("int_col2")])
        col_infos = [
            create_column_info(column_name=col, is_feature_var=True)
            for col in df.columns
        ]
        infos = [
            analyze_column(df, column_name=col, column_info=ci)
            for col, ci in zip(df.columns, col_infos, strict=False)
        ]
        # Create initial DataFrameInfo
        df_info = DataFrameInfo(
            num_rows=df.height,
            num_cols=df.width,
            column_info=infos,
        )
        # Compute correlation using the proper function
        df_info = get_pearson_correlation_for_df(df, df_info)
        result = _info_string_correlation_section(df_info)
        assert "Strongly correlated feature pairs" in result
        assert "int_col & int_col2" in result

    def test_cardinality_section(self, example_dataframe_info: DataFrameInfo) -> None:
        result = _info_string_cardinality_section(example_dataframe_info)
        # Should mention high cardinality or constant columns if present
        assert "cardinality" in result or "Constant" in result

    def test_outlier_section_always_empty(
        self, example_dataframe_info: DataFrameInfo
    ) -> None:
        result = _info_string_outlier_section(example_dataframe_info)
        assert result == ""

    def test_get_info_string_for_df_comprehensive(
        self, example_dataframe_info: DataFrameInfo
    ) -> None:
        result = get_info_string_for_df(example_dataframe_info)
        # Check that all sections are present
        assert (
            f"Rows: {example_dataframe_info.num_rows}, "
            f"Columns: {example_dataframe_info.num_cols}" in result
        )
        assert "missing" in result or "No missing values" in result
        assert "Target" in result or "categorical" in result or "numeric" in result
        assert "correlated" in result or "No correlation" in result
        assert "cardinality" in result or "Constant" in result
