#####################################################
# AutoMLAgent [EDA COLUMN UTILITIES]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Shared constants and utilities for EDA column tools."""

#####################################################
### BOARD

#####################################################
### IMPORTS

import polars as pl
import polars.selectors as cs

#####################################################
### SETTINGS

# Constants for categorical analysis
DEFAULT_CATEGORICAL_THRESHOLD = 10  # Default threshold for numeric columns
MAX_CATEGORICAL_RATIO = 0.05  # Max % of rows for categorical consideration
MAX_CATEGORIES_FOR_LEVEL = (
    20  # Max number of categories to show for categorical columns
)
LONG_TEXT_THRESHOLD = 1024  # Number of characters to be considered long text


#####################################################
### CODE
def column_filter_out_missing(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """Filter out missing values from the column (None and NaN)."""
    if column_name not in df.columns:
        msg = f"Column '{column_name}' not found in DataFrame."
        raise KeyError(msg)

    if column_name in df.select(cs.numeric()).columns:
        df_filtered = df.filter(
            pl.col(column_name).is_not_null() & pl.col(column_name).is_not_nan()
        )
        return df_filtered

    # NOTE(jdwh08): is_not_nan not valid for string columns
    df_filtered = df.filter(pl.col(column_name).is_not_null())
    return df_filtered
