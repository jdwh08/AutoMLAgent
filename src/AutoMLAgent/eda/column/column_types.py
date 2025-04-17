#####################################################
# AutoMLAgent [EDA COLUMN TYPE]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Type inference and dtype utilities for EDA columns."""

#####################################################
### BOARD

#####################################################
### IMPORTS

from __future__ import annotations

import mlflow
import polars as pl

from AutoMLAgent.dataclass.column_info import ColumnInfo
from AutoMLAgent.dataclass.column_type import ColumnType
from AutoMLAgent.eda.column.column_utils import (
    DEFAULT_CATEGORICAL_THRESHOLD,
    MAX_CATEGORICAL_RATIO,
)
from AutoMLAgent.logger.mlflow_logger import logger


#####################################################
### CODE
@mlflow.trace(name="get_type_for_column", span_type="func")
def get_type_for_column(
    df: pl.DataFrame,
    column_name: str,
    *,
    column_info: ColumnInfo | None = None,
) -> dict[str, ColumnType | bool | int]:
    """Get the type of a column in a dataframe.

    Args:
        df: The dataframe to get the column type from.
        column_name: The name of the column to get the type for.
        column_info: The column info for the column, if known.

    Returns:
        dict[str, ColumnType | bool | int]: A dictionary containing the attributes for column type.

    """
    output: dict[str, ColumnType | bool | int] = {}

    # Calculate cardinality if not already known
    cardinality: int = 99999
    if column_info is None or column_info.cardinality is None:
        try:
            cardinality = df[column_name].n_unique()
            output["cardinality"] = cardinality
        except Exception:
            logger.exception(f"Failed to calculate cardinality for {column_name}")
            cardinality = len(df)  # assume worse case
            output["cardinality"] = cardinality

    # Calculate cardinality threshold based on dataset size
    cardinality_threshold = min(
        int(len(df) * MAX_CATEGORICAL_RATIO), DEFAULT_CATEGORICAL_THRESHOLD
    )
    cardinality_threshold = max(cardinality_threshold, DEFAULT_CATEGORICAL_THRESHOLD)

    # Initialize type attributes
    output["type"] = ColumnType.UNKNOWN
    output["is_categorial"] = False
    output["is_numeric"] = False
    output["is_temporal"] = False

    try:
        # Check if column exists
        if column_name not in df.columns:
            return output

        # Create a single-column dataframe for selector operations
        single_col_df = df.select(column_name)

        # Boolean columns are always categorical
        if column_name in single_col_df.select(pl.selectors.boolean()).columns:
            output["type"] = ColumnType.BOOLEAN
            output["is_categorial"] = True
            return output

        # Numeric columns (but with low cardinality are categorical)
        if column_name in single_col_df.select(pl.selectors.integer()).columns:
            output["type"] = ColumnType.INTEGER
            output["is_numeric"] = True
            output["is_categorial"] = cardinality <= cardinality_threshold
            return output

        if column_name in single_col_df.select(pl.selectors.float()).columns:
            output["type"] = ColumnType.FLOAT
            output["is_numeric"] = True
            output["is_categorial"] = cardinality <= cardinality_threshold
            return output

        # Date and time columns
        if column_name in single_col_df.select(pl.selectors.datetime()).columns:
            output["type"] = ColumnType.DATETIME
            output["is_temporal"] = True
            return output

        if column_name in single_col_df.select(pl.selectors.date()).columns:
            output["type"] = ColumnType.DATE
            output["is_temporal"] = True
            return output

        if column_name in single_col_df.select(pl.selectors.time()).columns:
            output["type"] = ColumnType.TIME
            output["is_temporal"] = True
            return output

        # String columns
        if column_name in single_col_df.select(pl.selectors.string()).columns:
            output["type"] = ColumnType.TEXT
            output["is_categorial"] = cardinality <= cardinality_threshold
            return output

    except Exception:
        logger.exception(f"Error inferring column type for {column_name}")
        return output
    else:
        logger.warning(f"Failed to infer column type for {column_name}")
        return output
