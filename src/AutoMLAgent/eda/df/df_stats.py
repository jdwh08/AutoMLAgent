#####################################################
# AutoMLAgent [EDA COLUMN QUALITY]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Column-level stats across dataframe for EDA."""

#####################################################
### BOARD

#####################################################
### IMPORTS

import mlflow
import polars as pl

from AutoMLAgent.dataclass.column_info import ColumnInfo
from AutoMLAgent.dataclass.df_info import DataFrameInfo
from AutoMLAgent.eda.column.column_stats import (
    get_category_levels_for_column,
    get_histogram_bins_for_column,
    get_numerical_stats_for_column,
    get_string_stats_for_column,
    get_temporal_stats_for_column,
)


#####################################################
### CODE
@mlflow.trace(name="get_category_levels", span_type="func")
def get_category_levels(df: pl.DataFrame, df_info: DataFrameInfo) -> DataFrameInfo:
    """Generate category levels and their counts for categorical variables.

    Args:
        df: DataFrame containing the data
        df_info: DataFrame type information

    Returns:
        DataFrameInfo: Comprehensive type information for all columns

    """
    column_infos: list[ColumnInfo] = df_info.column_info
    column_infos = [
        column_info.model_copy(
            update=get_category_levels_for_column(df, column_info.name)
        )
        for column_info in column_infos
        if column_info.is_categorial
    ]
    df_info.column_info = column_infos
    return df_info


@mlflow.trace(name="get_histogram_bins", span_type="func")
def get_histogram_bins(df: pl.DataFrame, df_info: DataFrameInfo) -> DataFrameInfo:
    """Generate a markdown table of histogram bins and counts for numeric variables.

    Args:
        df: DataFrame containing the data
        df_info: DataFrame type information

    Returns:
        DataFrameInfo: Comprehensive type information for all columns

    """
    column_infos: list[ColumnInfo] = df_info.column_info
    column_infos = [
        column_info.model_copy(
            update=get_histogram_bins_for_column(df, column_info.name)
        )
        for column_info in column_infos
        if column_info.is_numeric
    ]
    df_info.column_info = column_infos
    return df_info


@mlflow.trace(name="get_numerical_stats", span_type="func")
def get_numerical_stats(df: pl.DataFrame, df_info: DataFrameInfo) -> DataFrameInfo:
    """Generate numerical statistics for all numeric columns in a dataframe.

    Args:
        df (pl.DataFrame): DataFrame containing the data
        df_info (DataFrameInfo): DataFrame type information

    Returns:
        DataFrameInfo: DataFrame type information with numerical statistics

    """
    column_infos: list[ColumnInfo] = df_info.column_info
    column_infos = [
        column_info.model_copy(
            update=get_numerical_stats_for_column(df, column_info.name)
        )
        for column_info in column_infos
        if column_info.is_numeric
    ]
    df_info.column_info = column_infos
    return df_info


@mlflow.trace(name="get_string_stats", span_type="func")
def get_string_stats(df: pl.DataFrame, df_info: DataFrameInfo) -> DataFrameInfo:
    """Generate string statistics for all string columns in a dataframe.

    Args:
        df (pl.DataFrame): DataFrame containing the data
        df_info (DataFrameInfo): DataFrame type information

    Returns:
        DataFrameInfo: Comprehensive type information for all columns

    """
    column_infos: list[ColumnInfo] = df_info.column_info
    column_infos = [
        column_info.model_copy(update=get_string_stats_for_column(df, column_info.name))
        for column_info in column_infos
        if column_info.is_string
    ]
    df_info.column_info = column_infos
    return df_info


@mlflow.trace(name="get_temporal_stats", span_type="func")
def get_temporal_stats(df: pl.DataFrame, df_info: DataFrameInfo) -> DataFrameInfo:
    """Generate temporal statistics for all temporal columns in a dataframe.

    Args:
        df (pl.DataFrame): DataFrame containing the data
        df_info (DataFrameInfo): DataFrame type information

    Returns:
        DataFrameInfo: Comprehensive type information for all columns

    """
    column_infos: list[ColumnInfo] = df_info.column_info
    column_infos = [
        column_info.model_copy(
            update=get_temporal_stats_for_column(df, column_info.name)
        )
        for column_info in column_infos
        if column_info.is_temporal
    ]
    df_info.column_info = column_infos
    return df_info
