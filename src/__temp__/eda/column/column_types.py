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

from typing import TYPE_CHECKING, TypeAlias

import mlflow
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.column_type import ColumnType
from automlagent.eda.column.column_utils import (
    DEFAULT_CATEGORICAL_THRESHOLD,
    MAX_CATEGORICAL_RATIO,
)
from automlagent.logger.mlflow_logger import get_mlflow_logger

#####################################################
### SETTINGS

ColumnTypeDict: TypeAlias = dict[str, ColumnType | bool | int]


#####################################################
### CODE
def initialize_output_dict(
    df: pl.DataFrame, column_name: str, column_info: ColumnInfo | None
) -> ColumnTypeDict:
    """Initialize the output dictionary with default values."""
    output: ColumnTypeDict = {}

    # Calculate cardinality if not already known
    cardinality: int = 99999  # NOTE(jdwh08): default very high (non-categorical)
    if column_info is None or column_info.cardinality is None:
        try:
            cardinality = df[column_name].n_unique()
            output["cardinality"] = cardinality
        except Exception:
            logger = get_mlflow_logger()
            logger.exception(f"Failed to calculate cardinality for {column_name}")
            cardinality = len(df)  # assume worse case
            output["cardinality"] = cardinality

    # Initialize type attributes
    output["type"] = ColumnType.UNKNOWN
    output["is_categorial"] = False
    output["is_numeric"] = False
    output["is_temporal"] = False

    return output


def create_column_type_handlers(
    df: pl.DataFrame,
    column_name: str,
    cardinality: int,
) -> Sequence[tuple[Callable[[], bool], Callable[[ColumnTypeDict], None]]]:
    """Create type handlers based on cardinality and dataset size.

    Args:
        df: The dataframe to get the column type from.
        column_name: The name of the column to get the type for.
        cardinality: The cardinality of the column.
            NOTE: yes we could caclulate it here, but I don't want duplicate logic

    Returns:
        list[tuple[Callable[[], bool], Callable[[ColumnTypeDict], None]]]:
            A list of (predicate, handler) pairs.

    """
    # Calculate cardinality threshold based on dataset size
    cardinality_threshold = min(
        int(len(df) * MAX_CATEGORICAL_RATIO), DEFAULT_CATEGORICAL_THRESHOLD
    )
    cardinality_threshold = max(cardinality_threshold, DEFAULT_CATEGORICAL_THRESHOLD)

    # Define handlers for each type
    def handle_boolean(output: ColumnTypeDict) -> None:
        output["type"] = ColumnType.BOOLEAN
        output["is_categorial"] = True

    def handle_integer(output: ColumnTypeDict) -> None:
        output["type"] = ColumnType.INTEGER
        output["is_numeric"] = True
        output["is_categorial"] = cardinality <= cardinality_threshold

    def handle_float(output: ColumnTypeDict) -> None:
        output["type"] = ColumnType.FLOAT
        output["is_numeric"] = True
        output["is_categorial"] = cardinality <= cardinality_threshold

    def handle_datetime(output: ColumnTypeDict) -> None:
        output["type"] = ColumnType.DATETIME
        output["is_temporal"] = True

    def handle_date(output: ColumnTypeDict) -> None:
        output["type"] = ColumnType.DATE
        output["is_temporal"] = True

    def handle_time(output: ColumnTypeDict) -> None:
        output["type"] = ColumnType.TIME
        output["is_temporal"] = True

    def handle_string(output: ColumnTypeDict) -> None:
        output["type"] = ColumnType.TEXT
        output["is_categorial"] = cardinality <= cardinality_threshold

    # Return ordered list of (predicate, handler) pairs
    # NOTE(jdwh08): predicate checks if handler applies, apply first handler that does
    output = [
        (
            lambda: column_name in df.select(pl.selectors.boolean()).columns,
            handle_boolean,
        ),
        (
            lambda: column_name in df.select(pl.selectors.integer()).columns,
            handle_integer,
        ),
        (
            lambda: column_name in df.select(pl.selectors.float()).columns,
            handle_float,
        ),
        (
            lambda: column_name in df.select(pl.selectors.datetime()).columns,
            handle_datetime,
        ),
        (
            lambda: column_name in df.select(pl.selectors.date()).columns,
            handle_date,
        ),
        (
            lambda: column_name in df.select(pl.selectors.time()).columns,
            handle_time,
        ),
        (
            lambda: column_name in df.select(pl.selectors.string()).columns,
            handle_string,
        ),
    ]
    return output


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
        dict[str, ColumnType | bool | int]: A dictionary containing
            the attributes for column type.

    """
    logger = get_mlflow_logger()
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

        # Get handlers
        type_handlers = create_column_type_handlers(
            df=df, column_name=column_name, cardinality=cardinality
        )
        # Apply handlers
        for predicate, handler in type_handlers:
            if predicate():
                handler(output)
                return output

    except Exception:
        logger.exception(f"Error inferring column type for {column_name}")
        return output
    else:
        logger.warning(f"Failed to infer column type for {column_name}")
        return output
