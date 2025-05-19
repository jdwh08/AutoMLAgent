#####################################################
# AutoMLAgent [COLUMN AND DF INFO UTILS]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Utility functions for column and dataframe information."""

#####################################################
### BOARD

#####################################################
### IMPORTS

import polars as pl

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.df_info import DataFrameInfo
from automlagent.logger.mlflow_logger import get_mlflow_logger

#####################################################
### CODE


def validate_column_info(
    column_name: str,
    *,
    data: pl.DataFrame | pl.Series | None = None,
    column_info: ColumnInfo | None = None,
    df_info: DataFrameInfo | None = None,
) -> ColumnInfo:
    """Validate column_info from data and/or df_info.

    Args:
        data: Dataframe or series of data
        column_name: Name of the column to validate
        column_info: ColumnInfo object
        df_info: DataFrameInfo object

    Returns:
        ColumnInfo object

    """
    logger = get_mlflow_logger()

    # Validate inputs
    if column_info is None and df_info is None:
        msg = "Either column_info or df_info must be provided"
        logger.exception(msg)
        raise ValueError(msg)

    if column_info is not None and df_info is not None:
        msg = "Cannot provide both column_info and df_info"
        logger.exception(msg)
        raise ValueError(msg)

    # Validate column_name is in data
    if data is not None:
        if isinstance(data, pl.DataFrame) and column_name not in data.columns:
            msg = f"Column {column_name} not found in dataframe"
            logger.exception(msg)
            raise ValueError(msg)
        if isinstance(data, pl.Series) and data.name != column_name:
            msg = f"Series name {data.name} does not match column_name {column_name}"
            logger.exception(msg)
            raise ValueError(msg)

    # Get column_info from df_info
    if column_info is None and df_info is not None:
        column_info = next(
            (c for c in df_info.column_info if c.name == column_name), None
        )

    if column_info is None:  # NOTE(jdwh08): this is here insetad of nested for typing
        msg = f"Column info for {column_name} not found in df_info."
        logger.exception(msg)
        raise ValueError(msg)

    if column_info.name != column_name:
        msg = (
            f"Column info name {column_info.name}"
            f" does not match column_name {column_name}"
        )
        logger.exception(msg)
        raise ValueError(msg)

    return column_info
