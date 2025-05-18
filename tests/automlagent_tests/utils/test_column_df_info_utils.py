"""Unit tests for column and dataframe info utilities."""

import polars as pl
import pytest

from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.column_type import ColumnType
from automlagent.dataclass.df_info import DataFrameInfo
from automlagent.utils.column_df_info_utils import validate_column_info


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a sample DataFrame."""
    return pl.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        }
    )


@pytest.fixture
def sample_series() -> pl.Series:
    """Create a sample Series."""
    return pl.Series("col1", [1, 2, 3])


@pytest.fixture
def sample_column_info() -> ColumnInfo:
    """Create a sample ColumnInfo object."""
    return ColumnInfo(
        name="col1",
        type=ColumnType.FLOAT,
        is_numeric=True,
        is_categorical=False,
        is_temporal=False,
    )


@pytest.fixture
def sample_df_info(sample_column_info: ColumnInfo) -> DataFrameInfo:
    """Create a sample DataFrameInfo object."""
    return DataFrameInfo(
        column_info=[sample_column_info],
        num_rows=3,
        num_cols=1,
    )


class TestColumnDFInfoUtils:
    def test_validate_column_info_with_column_info(
        self,
        sample_df: pl.DataFrame,
        sample_column_info: ColumnInfo,
    ) -> None:
        """Test validation with provided ColumnInfo."""
        result = validate_column_info(
            column_name="col1",
            data=sample_df,
            column_info=sample_column_info,
        )
        assert result == sample_column_info

    def test_validate_column_info_with_df_info(
        self,
        sample_df: pl.DataFrame,
        sample_df_info: DataFrameInfo,
    ) -> None:
        """Test validation with provided DataFrameInfo."""
        result = validate_column_info(
            column_name="col1",
            data=sample_df,
            df_info=sample_df_info,
        )
        assert result == sample_df_info.column_info[0]

    def test_validate_column_info_with_series(
        self,
        sample_series: pl.Series,
        sample_column_info: ColumnInfo,
    ) -> None:
        """Test validation with Series data."""
        result = validate_column_info(
            column_name="col1",
            data=sample_series,
            column_info=sample_column_info,
        )
        assert result == sample_column_info

    def test_validate_column_info_missing_inputs(self) -> None:
        """Test validation with missing inputs."""
        with pytest.raises(ValueError):
            validate_column_info(column_name="col1")

    def test_validate_column_info_both_inputs(
        self,
        sample_column_info: ColumnInfo,
        sample_df_info: DataFrameInfo,
    ) -> None:
        """Test validation with both ColumnInfo and DataFrameInfo."""
        with pytest.raises(ValueError):
            validate_column_info(
                column_name="col1",
                column_info=sample_column_info,
                df_info=sample_df_info,
            )

    def test_validate_column_info_invalid_column(
        self,
        sample_df: pl.DataFrame,
        sample_column_info: ColumnInfo,
    ) -> None:
        """Test validation with invalid column name."""
        with pytest.raises(ValueError):
            validate_column_info(
                column_name="nonexistent",
                data=sample_df,
                column_info=sample_column_info,
            )

    def test_validate_column_info_series_name_mismatch(
        self,
        sample_series: pl.Series,
        sample_column_info: ColumnInfo,
    ) -> None:
        """Test validation with mismatched Series name."""
        with pytest.raises(ValueError):
            validate_column_info(
                column_name="wrong_name",
                data=sample_series,
                column_info=sample_column_info,
            )

    def test_validate_column_info_not_in_df_info(
        self,
        sample_df: pl.DataFrame,
        sample_df_info: DataFrameInfo,
    ) -> None:
        """Test validation with column not in DataFrameInfo."""
        with pytest.raises(ValueError):
            validate_column_info(
                column_name="nonexistent",
                data=sample_df,
                df_info=sample_df_info,
            )

    def test_validate_column_info_name_mismatch(
        self,
        sample_df: pl.DataFrame,
        sample_column_info: ColumnInfo,
    ) -> None:
        """Test validation with mismatched ColumnInfo name."""
        wrong_info = ColumnInfo(
            name="wrong_name",
            type=ColumnType.FLOAT,
            is_numeric=True,
            is_categorical=False,
            is_temporal=False,
        )
        with pytest.raises(ValueError):
            validate_column_info(
                column_name="col1",
                data=sample_df,
                column_info=wrong_info,
            )
