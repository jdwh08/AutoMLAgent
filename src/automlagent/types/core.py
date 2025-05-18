"""Core type definitions for the AutoMLAgent package."""

from typing import TypeAlias

import polars as pl

# Type for histogram keys (string for categorical, tuple for numerical)
HistogramKey: TypeAlias = str | tuple[float, float]

# Polars temporal types
PolarsTemporal: TypeAlias = (
    pl.datatypes.Datetime
    | pl.datatypes.Date
    | pl.datatypes.Duration
    | pl.datatypes.Time
)
