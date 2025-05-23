#####################################################
# AutoMLAgent [EDA DATAFRAME QUALITY]
#####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Column-level quality across dataframe for EDA."""

#####################################################
### BOARD

#####################################################
### IMPORTS

import mlflow
import polars as pl

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.df_info import DataFrameInfo
from automlagent.eda.column.column_quality import (
    column_data_quality_missing_inf,
    get_data_quality_for_column,
)


#####################################################
### CODE
@mlflow.trace(name="get_data_quality", span_type="func")
def get_data_quality_for_df(
    df: pl.DataFrame,
    df_info: DataFrameInfo,
    *,
    outlier_zscore_threshold: float = 3.0,
    low_variance_threshold: float = 0.01,
) -> DataFrameInfo:
    """Analyze data quality for all columns in the dataframe.

    This function applies quality analysis to each column and updates
    the DataFrameInfo object with the results.

    Args:
        df: The input dataframe
        df_info: DataFrameInfo containing column information
        outlier_zscore_threshold: Z-score threshold for outlier detection
        low_variance_threshold: Variance threshold below which a column
            is considered to have low variation

    Returns:
        Updated DataFrameInfo with data quality metrics

    """
    column_infos: list[ColumnInfo] = df_info.column_info

    if len(column_infos) <= 0 and df.shape[1] > 0:
        msg = "DataFrameInfo does not have any column infos. Please create them first."
        raise ValueError(msg)

    column_infos = [
        column_info.model_copy(
            update=get_data_quality_for_column(
                df,
                column_info.name,
                column_info=column_info,
                outlier_zscore_threshold=outlier_zscore_threshold,
                low_variance_threshold=low_variance_threshold,
            )
        )
        for column_info in column_infos
    ]
    df_info = df_info.model_copy(update={"column_info": column_infos})
    return df_info


@mlflow.trace(name="get_missing_values_for_df", span_type="func")
def get_missing_values_for_df(
    df: pl.DataFrame, df_info: DataFrameInfo
) -> DataFrameInfo:
    """Analyze missing values for all columns in the dataframe.

    This function applies missing value analysis to each column and updates
    the DataFrameInfo object with the results.

    Args:
        df: DataFrame containing the data
        df_info: DataFrame type information

    Returns:
        DataFrameInfo: Comprehensive type information for all columns

    """
    column_infos: list[ColumnInfo] = df_info.column_info

    if len(column_infos) <= 0 and df.shape[1] > 0:
        msg = "DataFrameInfo does not have any column infos. Please create them first."
        raise ValueError(msg)

    column_infos = [
        column_info.model_copy(
            update=column_data_quality_missing_inf(
                df,
                column_info.name,
            )
        )
        for column_info in column_infos
    ]
    df_info = df_info.model_copy(update={"column_info": column_infos})
    return df_info


def analyze_missing_values(df_info: DataFrameInfo) -> str:
    """Generate a markdown table summarizing missing value rates for all columns.

    Args:
        df_info: DataFrameInfo containing column information

    Returns:
        str: Markdown table summarizing missing value rates

    """
    md_table: str = "| Column | Missing Rate |\n"
    md_table += "| ------ | ----------- |\n"
    md_table_rows = "\n".join(
        [
            f"| {column_info.name} | {(column_info.missing_rate or 0) * 100:.4f}% |"
            for column_info in df_info.column_info
            if column_info.missing_rate is not None and column_info.missing_rate > 0
        ]
    )
    md_table += md_table_rows if md_table_rows else "| No missing values found | 0% |"
    return md_table
