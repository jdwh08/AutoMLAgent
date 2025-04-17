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

#####################################################
### SETTINGS

# Constants for categorical analysis
DEFAULT_CATEGORICAL_THRESHOLD = 20  # Default threshold for numeric columns
MAX_CATEGORICAL_RATIO = 0.05  # Max % of rows for categorical consideration
MAX_CATEGORIES_FOR_LEVEL = (
    20  # Max number of categories to show for categorical columns
)
LONG_TEXT_THRESHOLD = 1024  # Number of characters to be considered long text

# Polars temporal types
polars_temporal = (
    pl.datatypes.Datetime
    | pl.datatypes.Date
    | pl.datatypes.Duration
    | pl.datatypes.Time
)
