from typing import TypeAlias  # pragma: no cover  # this is a type definition...

import polars as pl  # pragma: no cover  # this is a type definition...

# Polars temporal types
PolarsTemporal: TypeAlias = (  # pragma: no cover  # this is a type definition...
    pl.datatypes.Datetime
    | pl.datatypes.Date
    | pl.datatypes.Duration
    | pl.datatypes.Time
)
