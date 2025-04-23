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

# TODO(jdwh08): Better categorical column detection
# parameter-specified values?
# LLM-judged based on column description if exists?

#####################################################
### IMPORTS

from __future__ import annotations

import math
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

ColumnTypeDict: TypeAlias = dict[str, ColumnType | bool | int | float]


#####################################################
### CODE
def _get_cardinality(
    df: pl.DataFrame, column_name: str, input_dict: ColumnTypeDict
) -> ColumnTypeDict:
    """Get the cardinality of a column in a dataframe.

    Args:
        df (pl.DataFrame): The dataframe to get the cardinality from.
        column_name (str): The name of the column to get the cardinality for.
        input_dict (dict[str, object]): The input dictionary containing column info.

    Returns:
        dict[str, object]: We add cardinality info to the input dictionary.

    """
    cardinality: int = 99999  # NOTE(jdwh08): default very high (non-categorical)
    output = input_dict.copy()
    try:
        cardinality = df[column_name].n_unique()
    except Exception:
        logger = get_mlflow_logger()
        logger.exception(f"Failed to calculate cardinality for {column_name}")
        cardinality = len(df)  # assume worse case
    output["cardinality"] = cardinality
    output["unique_rate"] = cardinality / len(df) if len(df) > 0 else 0
    return output


def initialize_output_dict(
    df: pl.DataFrame, column_name: str, column_info: ColumnInfo | None
) -> ColumnTypeDict:
    """Initialize the output dictionary with default values."""
    output: ColumnTypeDict = {}

    # Calculate cardinality if not already known
    if column_info is None or column_info.cardinality is None:
        output = _get_cardinality(df, column_name, output)

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
) -> ColumnTypeDict:
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
    output: ColumnTypeDict = {}

    # Calculate cardinality if not already known
    if column_info is None or column_info.cardinality is None:
        output = _get_cardinality(df, column_name, output)
    else:
        output["cardinality"] = column_info.cardinality
        output["unique_rate"] = column_info.cardinality / df.height
    cardinality = output["cardinality"]

    if not isinstance(cardinality, int):  # pragma: no coverage <for type checker>
        msg = f"Expected integer cardinality, got {type(cardinality)}"
        logger.error(msg)
        raise TypeError(msg)

    # Calculate cardinality threshold based on dataset size
    cardinality_threshold = min(
        math.ceil(df.shape[0] * MAX_CATEGORICAL_RATIO), DEFAULT_CATEGORICAL_THRESHOLD
    )
    cardinality_threshold = max(cardinality_threshold, DEFAULT_CATEGORICAL_THRESHOLD)

    # Initialize type attributes
    output["type"] = ColumnType.UNKNOWN
    output["is_categorial"] = False
    output["is_numeric"] = False
    output["is_temporal"] = False

    # Check if column exists
    if column_name not in df.columns:
        return output

    try:
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
