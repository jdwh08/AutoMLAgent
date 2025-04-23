#####################################################
# AutoMLAgent [EDA DATAFRAME TYPES]
#####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Column-level type inference tools across dataframe for EDA."""

#####################################################
### BOARD

#####################################################
### IMPORTS
import mlflow
import polars as pl

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.df_info import DataFrameInfo
from automlagent.eda.column.column_types import get_type_for_column


#####################################################
### CODE
@mlflow.trace(name="get_column_types", span_type="func")
def get_column_types(df: pl.DataFrame, df_info: DataFrameInfo) -> DataFrameInfo:
    """Extract column type information for columns in a dataframe.

    Args:
        df: The dataframe to get the column type from.
        df_info: A DataFrameInfo object with at least the name populated.
            Other fields may be pre-populated and will be preserved unless overridden.

    Returns:
        DataFrameInfo: updated DataFrameInfo object with column type info.

    """
    column_infos: list[ColumnInfo] = df_info.column_info
    column_infos = [
        column_info.model_copy(
            update=get_type_for_column(
                df, column_name=column_info.name, column_info=column_info
            )
        )
        for column_info in column_infos
    ]
    df_info.column_info = column_infos
    return df_info
