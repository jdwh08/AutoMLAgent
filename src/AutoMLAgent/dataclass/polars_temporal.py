from typing import TypeAlias

import polars as pl

# Polars temporal types
PolarsTemporal: TypeAlias = (
    pl.datatypes.Datetime
    | pl.datatypes.Date
    | pl.datatypes.Duration
    | pl.datatypes.Time
)
