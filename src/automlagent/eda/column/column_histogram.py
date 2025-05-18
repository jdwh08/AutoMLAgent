#####################################################
# AutoMLAgent [EDA HISTOGRAM UTILS]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Column-level histogram generation and analysis."""

#####################################################
### BOARD
# TODO(jdwh08): Apply Dependency Injection to allow multiple histogram bin options

#####################################################
### IMPORTS

import mlflow
import polars as pl

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.logger.mlflow_logger import get_mlflow_logger
from automlagent.utils.histogram_utils import (
    HistogramKey,
    create_histogram,
)

#####################################################
### SETTINGS


#####################################################
### CODE


@mlflow.trace(name="get_histogram_for_column", span_type="func")
def get_histogram_for_column(
    df: pl.DataFrame,
    column_name: str,
    *,
    column_info: ColumnInfo | None = None,
) -> dict[str, dict[HistogramKey, int]] | None:
    """Create a histogram for a column in a dataframe.

    Args:
        df (pl.DataFrame): The dataframe to analyze.
        column_name (str): The column to analyze.
        column_info (ColumnInfo, optional): Column information.
            Defaults to None.

    Returns:
        dict[str, dict[HistogramKey, int]]:
            Histogram of data to update column_info with.
            For numerical columns: dict from bin ranges (start, end) to counts.
            For categorical columns: dict from category values to counts.
            If failed to create, dict is None.

    """
    logger = get_mlflow_logger()

    if column_name not in df.columns:
        msg = f"Column '{column_name}' not found in DataFrame."
        raise KeyError(msg)
    if df.select(pl.col(column_name)).is_empty():
        return {}

    try:
        histogram = create_histogram(
            data=df, column_name=column_name, column_info=column_info
        )
    except Exception:
        logger.exception(f"Failed to create histogram for {column_name}")
        return {}

    return {"histogram": histogram}
