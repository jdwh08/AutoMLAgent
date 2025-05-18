#####################################################
# AutoMLAgent [EDA DATAFRAME INFO STRING]
#####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Dataframe Info String for EDA."""

#####################################################
### BOARD

#####################################################
### IMPORTS
from typing import Final

import polars as pl

### OWN MODULES
from automlagent.dataclass.df_info import DataFrameInfo

#####################################################
### SETTINGS
HIGH_MISSING_RATE_COL: Final[float] = 0.3  # missing rate to have "high missing"
HIGH_CORR_THRESHOLD: Final[float] = 0.8  # threshold to have "high correlation"
HIGH_CARDINALITY: Final[int] = 50  # threshold to have "high cardinality"

#####################################################
### CODE


def _info_string_shape_section(df_info: DataFrameInfo) -> str:
    """Summarize DataFrame shape and column types.

    Args:
        df_info (DataFrameInfo): The DataFrameInfo object.

    Returns:
        str: Shape and type summary.

    """
    type_counts = {
        "int": len(df_info.int_cols),
        "float": len(df_info.float_cols),
        "bool": len(df_info.bool_cols),
        "datetime": len(df_info.datetime_cols),
        "date": len(df_info.date_cols),
        "time": len(df_info.time_cols),
        "text": len(df_info.text_cols),
        "categorical": len(df_info.categorical_cols),
    }
    type_summary = ", ".join([f"{v} {k}" for k, v in type_counts.items() if v > 0])
    output = (
        f"## DataFrame Shape\n"
        f"Rows: {df_info.num_rows}, Columns: {df_info.num_cols} "
        f"({type_summary})"
    )
    return output


def _info_string_missing_section(df_info: DataFrameInfo) -> str:
    """Summarize missing data patterns.

    Args:
        df_info (DataFrameInfo): The DataFrameInfo object.

    Returns:
        str: Missing value summary.

    """
    cols_with_missing = df_info.columns_with_missing_values
    if not cols_with_missing:
        return "No missing values detected."
    lines = [
        f"{len(cols_with_missing)} columns have missing values: "
        f"{', '.join(cols_with_missing)}."
    ]
    # Highlight columns with high missing rate
    high_missing = [
        col.name
        for col in df_info.column_info
        if col.missing_rate and col.missing_rate > HIGH_MISSING_RATE_COL
    ]
    if high_missing:
        lines.append(
            f"Columns with >{HIGH_MISSING_RATE_COL * 100:.1f}% missing:"
            f"{', '.join(high_missing)}."
        )
    return "\n".join(lines)


def _info_string_duplicates_section(df_info: DataFrameInfo) -> str:
    """Summarize duplicate rows (stub; not available in DataFrameInfo).

    Args:
        df_info (DataFrameInfo): The DataFrameInfo object.

    Returns:
        str: Duplicate row summary (empty if not available).

    """
    # No explicit duplicate info in DataFrameInfo; stub for future extension
    return ""  # TODO(jdwh08): Add duplicate row summary if available


def _info_string_target_section(df_info: DataFrameInfo) -> str:
    """Summarize target variable if present.

    Args:
        df_info (DataFrameInfo): The DataFrameInfo object.

    Returns:
        str: Target variable summary.

    """
    target = df_info.target_var
    if not target:
        return ""
    if getattr(target, "is_categorical", False):
        n_classes = target.cardinality or 0
        return f"Target '{target.name}' is categorical with {n_classes} classes."
    if getattr(target, "is_numeric", False):
        return (
            f"Target '{target.name}' is numeric. "
            f"Mean: {target.mean}, Std: {target.std}, "
            f"Min: {target.min}, Max: {target.max}."
        )
    return f"Target '{target.name}' type: {target.type}."


def _info_string_correlation_section(df_info: DataFrameInfo) -> str:
    """Summarize inter-column correlations.

    Args:
        df_info (DataFrameInfo): The DataFrameInfo object.

    Returns:
        str: Correlation summary.

    """
    corr = getattr(df_info, "correlation_pearson", None)
    if corr is None or getattr(corr, "height", 0) == 0:
        return "No correlation matrix available."

    # Get column names excluding the index column
    columns = [col for col in corr.columns if col != "index"]

    # Create a list to store strong correlations
    strong_corrs = []

    # Iterate through upper triangle of correlation matrix
    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1 :]:
            corr_value = corr.filter(pl.col("index") == col1).select(col2).item()
            if abs(corr_value) >= HIGH_CORR_THRESHOLD:
                strong_corrs.append((col1, col2, corr_value))

    if not strong_corrs:
        return f"No strongly correlated feature pairs (|r| >= {HIGH_CORR_THRESHOLD})."

    # Format the results
    lines = [f"Strongly correlated feature pairs (|r| >= {HIGH_CORR_THRESHOLD}):"]
    for col1, col2, val in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
        lines.append(f"- {col1} & {col2}: r = {val:.2f}")

    return "\n".join(lines)


def _info_string_cardinality_section(df_info: DataFrameInfo) -> str:
    """Highlight high-cardinality and constant columns.

    Args:
        df_info (DataFrameInfo): The DataFrameInfo object.

    Returns:
        str: Cardinality/constant columns summary.

    """
    high_card_cols = [
        col.name
        for col in df_info.column_info
        if col.cardinality and col.cardinality > HIGH_CARDINALITY
    ]
    constant_cols = [col.name for col in df_info.column_info if col.cardinality == 1]
    lines: list[str] = []
    if high_card_cols:
        lines.append(
            f"Columns with high cardinality (>{HIGH_CARDINALITY}): {', '.join(high_card_cols)}."
        )
    if constant_cols:
        lines.append(
            f"Constant columns (single unique value): {', '.join(constant_cols)}."
        )
    return "\n".join(lines)


def _info_string_outlier_section(df_info: DataFrameInfo) -> str:
    """Summarize outlier findings (stub; not available in DataFrameInfo).

    Args:
        df_info (DataFrameInfo): The DataFrameInfo object.

    Returns:
        str: Outlier summary (empty if not available).

    """
    # No DataFrame-level outlier info in DataFrameInfo; stub for future extension
    return ""  # TODO(jdwh08): Add outlier summary if available


def generate_info_string_for_df(df_info: DataFrameInfo) -> str:
    """Generate a comprehensive EDA summary string for the DataFrame.

    Args:
        df_info (DataFrameInfo): The DataFrameInfo object.

    Returns:
        str: A formatted string with key EDA findings.

    """
    info_parts: list[str] = [
        _info_string_shape_section(df_info),
        _info_string_missing_section(df_info),
        _info_string_duplicates_section(df_info),
        _info_string_target_section(df_info),
        _info_string_correlation_section(df_info),
        _info_string_cardinality_section(df_info),
        _info_string_outlier_section(df_info),
    ]
    return "\n\n".join(part for part in info_parts if part)
