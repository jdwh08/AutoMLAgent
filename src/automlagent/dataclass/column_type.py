from enum import Enum


class ColumnType(str, Enum):
    """Enumeration of column data types."""

    INT = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    TEXT = "text"
    UNKNOWN = "unknown"
