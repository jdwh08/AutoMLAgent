#####################################################
# AutoMLAgent [Column Info]
# ####################################################
# Jonathan Wang <jdwh08>

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Column Info class for EDA."""

#####################################################
### BOARD

#####################################################
### IMPORTS

import os

import mlflow
import polars as pl
from dotenv.main import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

### OWN MODULES
from AutoMLAgent.dataclass.column_type import ColumnType

#####################################################
### SETTINGS

load_dotenv()

polars_temporal = (
    pl.datatypes.Datetime
    | pl.datatypes.Date
    | pl.datatypes.Duration
    | pl.datatypes.Time
)

# MLFlow
# mlflow.openai.autolog()
mlflow.gemini.autolog()  # NOTE(jdwh08): doesn't really work
mlflow.autolog()  # for modelling tasks
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME"))
mlflow.config.enable_async_logging()


#####################################################
### DATACLASSES


class ColumnInfo(BaseModel):
    """Information about a single column."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # needed for datetime

    # Basics
    name: str
    is_target_var: bool = False
    is_feature_var: bool = True

    # Type
    type: ColumnType = ColumnType.UNKNOWN
    is_categorial: bool = False
    is_numeric: bool = False
    is_temporal: bool = False

    # Data Quality
    missing_count: int | None = None
    missing_rate: float | None = None
    outlier_count: int | None = None
    outlier_rate: float | None = None
    has_low_variation: bool = False

    # Info string
    info: str = Field(default="")

    # Sample values
    sample_values: list[str] | None = None

    # Categorical analysis fields
    cardinality: int | None = None
    category_counts: dict[str, int] = Field(default_factory=dict)

    # Numeric analysis fields
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    std: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    histogram_bins: list[float] = Field(default_factory=list)
    histogram_counts: list[int] = Field(default_factory=list)

    # String analysis fields
    char_length_min: int | None = None
    char_length_max: int | None = None
    char_length_mean: float | None = None
    char_length_std: float | None = None

    # Temporal analysis fields
    temporal_min: polars_temporal | None = None
    temporal_max: polars_temporal | None = None
    temporal_diff: polars_temporal | None = None

    # Flag to track if detailed analysis has been performed
    is_analyzed: bool = False


@mlflow.trace(name="create_column_info", span_type="func")
def create_column_info(
    column_name: str,
    *,
    is_target_var: bool = False,
    is_feature_var: bool = True,
) -> ColumnInfo:
    """Create a ColumnInfo object from a dataframe column.

    Args:
        column_name: The name of the column to create the ColumnInfo object from.
        is_target_var: Whether the column is the target variable.
        is_feature_var: Whether the column is a feature variable.

    Returns:
        ColumnInfo: A blank ColumnInfo object for the column.

    """
    return ColumnInfo(
        name=column_name, is_target_var=is_target_var, is_feature_var=is_feature_var
    )
