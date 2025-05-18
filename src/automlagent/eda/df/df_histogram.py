#####################################################
# AutoMLAgent [EDA DATAFRAME HISTOGRAM]
#####################################################
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

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.df_info import DataFrameInfo
from automlagent.eda.column.column_histogram import get_histogram_for_column

#####################################################
### CODE


@mlflow.trace(name="get_histogram_for_df", span_type="func")
def get_histogram_for_df(df: pl.DataFrame, df_info: DataFrameInfo) -> DataFrameInfo:
    """Generate histogram for all relevant columns in a dataframe.

    Args:
        df (pl.DataFrame): DataFrame containing the data
        df_info (DataFrameInfo): DataFrame type information

    Returns:
        DataFrameInfo: DataFrame type information with histogram

    """
    column_infos: list[ColumnInfo] = df_info.column_info
    column_infos = [
        column_info.model_copy(
            update=get_histogram_for_column(
                df, column_info.name, column_info=column_info
            )
        )
        for column_info in column_infos
        if column_info.is_numeric or column_info.is_categorical
    ]
    df_info.column_info = column_infos
    return df_info
