#####################################################
# AutoMLAgent [EDA HISTOGRAM UTILS]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Utility functions for histogram generation and analysis."""

#####################################################
### BOARD

#####################################################
### IMPORTS

import math
from typing import Protocol, cast, runtime_checkable

import mlflow
import polars as pl

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.df_info import DataFrameInfo
from automlagent.logger.mlflow_logger import get_mlflow_logger
from automlagent.types.core import HistogramKey
from automlagent.utils.column_df_info_utils import validate_column_info

#####################################################
### SETTINGS


# Protocol for histogram bin calculation functions
@runtime_checkable
class HistogramBinCalculator(Protocol):
    """Protocol for calculating the number of bins for a histogram.

    This protocol defines the interface for functions that calculate the optimal
    number of bins for a histogram based on statistical properties of the data.
    """

    def __call__(self, data: pl.Series) -> int:
        """Calculate the optimal number of bins for a histogram.

        Args:
            data (pl.Series): The data to analyze.

        Returns:
            int: The optimal number of bins to use for the histogram.

        """
        ...


#####################################################
### CODE


def freedman_diaconis_bins(data: pl.Series) -> int:
    """Calculate the optimal number of bins using the Freedman-Diaconis rule.

    The Freedman-Diaconis rule is a method for determining the optimal bin width
    for a histogram. It is based on the interquartile range (IQR) and the sample size.

    Args:
        data (pl.Series): The data to analyze.

    Returns:
        int: The optimal number of bins.

    """
    # Filter out missing values
    data_filtered = data.filter(data.is_not_null())

    # Calculate IQR
    q1 = data_filtered.quantile(0.25)
    q3 = data_filtered.quantile(0.75)

    if q1 is None:
        msg = "Failed to calculate Freedman-Diaconis bins, q1 is None"
        raise ValueError(msg)
    if q3 is None:
        msg = "Failed to calculate Freedman-Diaconis bins, q3 is None"
        raise ValueError(msg)

    iqr = q3 - q1

    # Get sample size
    n = len(data_filtered)

    # Calculate bin width using Freedman-Diaconis rule
    bin_width = 2 * iqr / (n ** (1 / 3))

    # Calculate range
    data_min = data_filtered.min()
    data_max = data_filtered.max()

    if type(data_min) is not type(data_max):
        msg = (
            "Failed to calculate Freedman-Diaconis bins, "
            "data_min and data_max are not of the same type"
        )
        raise ValueError(msg)

    data_range = data_max - data_min  # type: ignore[operator]

    # Calculate number of bins
    num_bins = max(1, math.ceil(data_range / bin_width))
    return num_bins


def sturges_bins(data: pl.Series) -> int:
    """Calculate the number of bins using Sturges' rule.

    Sturges' rule is a simple method that suggests using log2(n) + 1 bins,
    where n is the number of observations.

    Args:
        data (pl.Series): The data to analyze.

    Returns:
        int: The number of bins to use.

    """
    n = len(data.filter(data.is_not_null()))
    return max(1, math.ceil(math.log2(n) + 1))


def _create_categorical_histogram(data: pl.Series) -> dict[str, int]:
    """Create a histogram for categorical data.

    Args:
        data (pl.Series): The data to analyze.

    Returns:
        dict[str, int]: Dictionary mapping category values to counts.

    """
    value_counts = data.value_counts(sort=True, name="counts", parallel=True)

    if value_counts.is_empty():
        return {}

    # Convert to dict
    histogram_data: dict[str, int] = {}
    for row in value_counts.rows():
        val, count = row
        histogram_data[str(val)] = count

    return histogram_data


def _create_numerical_histogram(
    data: pl.Series,
    bin_calculator: HistogramBinCalculator | None = None,
) -> dict[tuple[float, float], int]:
    """Create a histogram for numerical data.

    Args:
        data (pl.Series): The data to analyze.
        bin_calculator (HistogramBinCalculator, optional): Function calculating # bins.
            Defaults to None, for Polars default.

    Returns:
        dict[tuple[float, float], int]: Dictionary mapping bin ranges to counts.

    """
    logger = get_mlflow_logger()

    # Calculate number of bins
    num_bins = bin_calculator(data) if bin_calculator is not None else None

    # Get data range for edge bins
    data_min = float(data.min())  # type: ignore[arg-type]
    data_max = float(data.max())  # type: ignore[arg-type]

    # Create histogram with specified number of bins
    histogram = data.hist(bin_count=num_bins, include_category=True).sort("category")

    if histogram is None or len(histogram) <= 0:
        return {}

    # Process histogram data
    histogram_data: dict[tuple[float, float], int] = {}

    # Add bin for values below minimum
    histogram_data[(float("-inf"), data_min)] = 0

    for row in histogram.rows():
        _, category, count = row

        if not category or not isinstance(category, str):
            continue

        # Extract numeric values from interval notation
        bounds = category.strip("()[]").split(",")
        if len(bounds) != 2:
            continue

        try:
            lower = float(bounds[0].strip())
            upper = float(bounds[1].strip())
            histogram_data[(lower, upper)] = count
        except ValueError:
            logger.exception(f"Failed to convert bounds to float: {bounds}")
            continue

    # Add bin for values above maximum
    histogram_data[(data_max, float("inf"))] = 0

    return histogram_data


@mlflow.trace(name="create_histogram", span_type="func")
def create_histogram(
    data: pl.DataFrame | pl.Series,
    column_name: str | None = None,
    *,
    bin_calculator: HistogramBinCalculator | None = None,
    column_info: ColumnInfo | None = None,
    df_info: DataFrameInfo | None = None,
) -> dict[HistogramKey, int] | None:
    """Create a histogram for data.

    Args:
        data (pl.DataFrame | pl.Series): The data to analyze.
            If DataFrame, column_name must be provided.
        column_name (str | None, optional): Column to analyze if data is a DataFrame.
            Defaults to None.
        bin_calculator (HistogramBinCalculator, optional): Function calculating # bins.
            Only used for numerical data. Defaults to None, for Polars default.
        column_info (ColumnInfo, optional): Column information, e.g., column type.
            Defaults to None.
        df_info (DataFrameInfo, optional): DataFrame information.
            Defaults to None.

    Returns:
        dict[HistogramKey, int]: Histogram of data.
            For numerical data: dict from bin ranges (start, end) to counts.
            For categorical data: dict from category values to counts.
            If failed to create, returns None.

    """
    logger = get_mlflow_logger()

    # Convert dataframe/column_name to series
    if isinstance(data, pl.DataFrame):
        if column_name is None:
            msg = "column_name must be provided when data is a DataFrame"
            raise ValueError(msg)

        if column_name not in data.columns:
            msg = f"Column '{column_name}' not found in DataFrame"
            raise KeyError(msg)

        data = data[column_name]
    elif isinstance(data, pl.Series) and column_name is None:
        column_name = data.name

    output: dict[HistogramKey, int] | None

    # Get data dtype from column_info
    is_categorical: bool = False
    try:
        column_info = validate_column_info(
            column_name=column_name,
            data=data,
            column_info=column_info,
            df_info=df_info,
        )
        is_categorical = column_info.is_categorical
    except Exception:
        logger.exception("Failed to get column_info, estimating with data dtype.")
        is_categorical = not data.dtype.is_numeric()

    try:
        if is_categorical:
            output = cast(
                "dict[HistogramKey, int]",
                _create_categorical_histogram(data),
            )
        else:
            output = cast(
                "dict[HistogramKey, int]",
                _create_numerical_histogram(data, bin_calculator),
            )
    except Exception:
        logger.exception("Failed to create histogram")
        return None
    else:
        return output
