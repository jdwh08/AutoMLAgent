#####################################################
# AutoMLAgent [DataFrame Info]
# ####################################################
# Jonathan Wang <jdwh08>

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""DataFrame Info class for EDA."""

#####################################################
### BOARD

#####################################################
### IMPORTS

import mlflow
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, computed_field

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo, create_column_info
from automlagent.dataclass.column_type import ColumnType

#####################################################
### SETTINGS

polars_temporal = (
    pl.datatypes.Datetime
    | pl.datatypes.Date
    | pl.datatypes.Duration
    | pl.datatypes.Time
)


#####################################################
### DATACLASSES
class DataFrameInfo(BaseModel):
    """Type information for all columns in a dataframe."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # needed for correlation matrix

    # Shape
    num_rows: int
    num_cols: int

    # Data Quality
    has_duplicate_rows: bool = False

    @computed_field
    @property
    def missing_value_cols(self) -> list[str]:
        return [
            col.name
            for col in self.column_info
            if col.missing_count is not None and col.missing_count > 0
        ]

    # Per-column information
    column_info: list[ColumnInfo] = Field(default_factory=list)

    # Correlation matrix
    correlation_matrix: pl.DataFrame | None = None

    # Helper properties for convenience
    @computed_field
    @property
    def target_var(self) -> ColumnInfo | None:
        return next(
            (col for col in self.column_info if col.is_target_var),
            None,
        )

    @computed_field
    @property
    def feature_vars(self) -> list[str]:
        return [col.name for col in self.column_info if col.is_feature_var]

    @computed_field
    @property
    def cardinality(self) -> dict[str, int]:
        return {col.name: (col.cardinality or -1) for col in self.column_info}

    @computed_field
    @property
    def int_cols(self) -> list[str]:
        return [col.name for col in self.column_info if col.type == ColumnType.INTEGER]

    @computed_field
    @property
    def float_cols(self) -> list[str]:
        return [col.name for col in self.column_info if col.type == ColumnType.FLOAT]

    @computed_field
    @property
    def bool_cols(self) -> list[str]:
        return [col.name for col in self.column_info if col.type == ColumnType.BOOLEAN]

    @computed_field
    @property
    def datetime_cols(self) -> list[str]:
        return [col.name for col in self.column_info if col.type == ColumnType.DATETIME]

    @computed_field
    @property
    def date_cols(self) -> list[str]:
        return [col.name for col in self.column_info if col.type == ColumnType.DATE]

    @computed_field
    @property
    def time_cols(self) -> list[str]:
        return [col.name for col in self.column_info if col.type == ColumnType.TIME]

    @computed_field
    @property
    def text_cols(self) -> list[str]:
        return [col.name for col in self.column_info if col.type == ColumnType.TEXT]

    @computed_field
    @property
    def categorical_cols(self) -> list[str]:
        return [
            col.name for col in self.column_info if col.type == ColumnType.CATEGORICAL
        ]

    @computed_field
    @property
    def columns_with_missing_values(self) -> list[str]:
        return [
            col.name
            for col in self.column_info
            if col.missing_count is not None and col.missing_count > 0
        ]


@mlflow.trace(name="create_data_frame_info", span_type="func")
def create_data_frame_info(
    df: pl.DataFrame,
    target_var: str,
    feature_vars: list[str] | None = None,
) -> DataFrameInfo:
    """Create a DataFrameInfo object from a dataframe.

    Args:
        df: The dataframe to create the DataFrameInfo object from.
        target_var: The target variable for the dataset.
        feature_vars: The feature variables for the dataset.

    Returns:
        DataFrameInfo: A DataFrameInfo object containing type information for all columns.

    """
    # Create feature_vars from target var if none.
    if feature_vars is None:
        feature_vars = [col for col in df.columns if col != target_var]

    # Create target var column info
    target_var_info: ColumnInfo = create_column_info(target_var, is_target_var=True)

    # Create feature var column info
    feature_var_info: list[ColumnInfo] = [
        create_column_info(col, is_feature_var=True) for col in feature_vars
    ]

    # Create column info
    column_info: list[ColumnInfo] = [target_var_info, *feature_var_info]

    # Create DataFrameInfo
    return DataFrameInfo(
        num_rows=df.shape[0],
        num_cols=df.shape[1],
        column_info=column_info,
    )
