#####################################################
# AutoMLAgent [EDA DATAFRAME CORRELATION]
#####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Correlation analysis tools across dataframe for EDA."""

#####################################################
### BOARD

# TODO(jdwh08): Cramer's V for categorical columns?

#####################################################
### IMPORTS
import warnings

import mlflow
import polars as pl

### OWN MODULES
from automlagent.dataclass.df_info import DataFrameInfo
from automlagent.logger.mlflow_logger import get_mlflow_logger


#####################################################
### CODE
@mlflow.trace(name="get_pearson_correlation_for_df", span_type="func")
def get_pearson_correlation_for_df(
    df: pl.DataFrame, df_info: DataFrameInfo
) -> DataFrameInfo:
    """Compute Pearson correlation for numeric columns.

    Args:
        df (pl.DataFrame): The input dataframe.
        df_info (DataFrameInfo): DataFrame metadata including target column info.

    Returns:
        DataFrameInfo: DataFrame metadata with correlation matrix added.

    Raises:
        ValueError: If the target column is not numeric or not found in the dataframe.

    """
    logger = get_mlflow_logger()

    # Extract numeric column names from DataFrameInfo
    numeric_cols = [
        col.name
        for col in df_info.column_info
        if col.is_numeric and col.name in df.columns
    ]
    if not numeric_cols:
        msg = "No numeric features found in the dataframe."
        logger.warning(msg)
        warnings.warn(msg, stacklevel=2)
        return df_info

    # Select only numeric columns in the correct order
    df_numeric = df.select(numeric_cols)

    # Compute correlation matrix
    # NOTE(jdwh08): Watch out for large matricies!
    # We might needs something better than np.corrcoef on the entire matrix
    corr_matrix = df_numeric.corr()

    # Ensure the order of columns and rows matches numeric_cols
    corr_matrix = corr_matrix.select(numeric_cols)
    corr_matrix = corr_matrix.with_columns([pl.Series("index", numeric_cols)])
    corr_matrix = corr_matrix.select(["index", *numeric_cols])

    # Update df_info with correlation matrix
    df_info = df_info.model_copy(
        update={
            "correlation_pearson": corr_matrix,
        }
    )
    return df_info


@mlflow.trace(name="get_correlation_for_df", span_type="func")
def get_correlation_for_df(
    col_1: str, col_2: str, df_info: DataFrameInfo
) -> float | None:
    """Get the correlation between two columns in the dataframe."""
    logger = get_mlflow_logger()
    if df_info.correlation_pearson is None:
        msg = "Correlation matrix not computed yet!"
        logger.warning(msg)
        warnings.warn(msg, stacklevel=2)
        return None

    # Check if columns are in the correlation matrix
    if col_1 not in df_info.correlation_pearson.columns:
        msg = f"Column '{col_1}' not found in correlation matrix."
        logger.warning(msg)
        warnings.warn(msg, stacklevel=2)
        return None
    if col_2 not in df_info.correlation_pearson.columns:
        msg = f"Column '{col_2}' not found in correlation matrix."
        logger.warning(msg)
        warnings.warn(msg, stacklevel=2)
        return None

    # Get correlation
    row_match = df_info.correlation_pearson.filter(pl.col("index") == col_1)
    output = row_match[0, col_2]
    return float(output)
