#####################################################
# AutoMLAgent [EDA COLUMN INFO STRING]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Column info string for EDA columns."""

#####################################################
### BOARD

#####################################################
### IMPORTS

from __future__ import annotations

from typing import TypeAlias

import mlflow

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.eda.column.column_utils import MAX_CATEGORIES_FOR_LEVEL
from automlagent.logger.mlflow_logger import get_mlflow_logger

#####################################################
### SETTINGS
LOW_VARIATION_THRESHOLD = 0.05  # x% not majority class to be low variation

ColumnDataQualityMetrics: TypeAlias = dict[str, int | float | bool]


#####################################################
### CODE


def _info_string_type_section(column_info: ColumnInfo) -> str:
    """Add type information."""
    type_info = f"**Type**: {column_info.type.value}"
    if column_info.is_target_var:
        type_info += " (Target Variable)"
    elif column_info.is_feature_var:
        type_info += " (Feature)"
    return type_info


def _info_string_missing_values_section(column_info: ColumnInfo) -> str:
    """Add missing values info."""
    if (
        column_info.missing_count is not None
        and column_info.missing_rate is not None
        and column_info.missing_rate > 0
    ):
        return (
            f"Missing values: {column_info.missing_count} "
            f"({column_info.missing_rate * 100:.2f}%)"
        )
    return "No missing values"


def _info_string_categorial_stats_section(column_info: ColumnInfo) -> str:
    """Add categorial stats info."""
    num_categories = (
        len(column_info.category_counts)
        if column_info.category_counts
        else column_info.cardinality
    )
    output: str = ""

    if num_categories is None:
        return ""

    output += f"Unique categories #: {num_categories}\n"

    if num_categories < MAX_CATEGORIES_FOR_LEVEL:
        output += f"Category distribution: {column_info.category_counts}"
    else:
        # Show top categories only
        top_cats = dict(
            list(column_info.category_counts.items())[:MAX_CATEGORIES_FOR_LEVEL]
        )
        output += f"Top categories: {top_cats} ...\n"
        output += (
            f"and {len(column_info.category_counts) - MAX_CATEGORIES_FOR_LEVEL} "
            "more categories"  # NOTE(jdwh08): we use this for categories omitted check
        )

    return output


def _info_string_numerical_stats_section(column_info: ColumnInfo) -> str:
    """Add numerical stats info."""
    info_parts: list[str] = []
    if column_info.min is not None and column_info.max is not None:
        info_parts.append(f"Range: {column_info.min} to {column_info.max}")

    if column_info.mean is not None and column_info.median is not None:
        info_parts.append(
            f"Mean: {column_info.mean:.4f}, Median: {column_info.median:.4f}"
        )

    if column_info.std is not None:
        info_parts.append(f"Std Dev: {column_info.std:.4f}")

    if column_info.q1 is not None and column_info.q3 is not None:
        info_parts.append(f"Q1: {column_info.q1:.4f}, Q3: {column_info.q3:.4f}")

    if column_info.skewness is not None and column_info.kurtosis is not None:
        info_parts.append(
            f"Skewness: {column_info.skewness:.4f}, "
            f"Kurtosis: {column_info.kurtosis:.4f}"
        )

    return "\n".join(info_parts)


def _info_string_numerical_histogram_section(column_info: ColumnInfo) -> str:
    """Add numerical histogram info."""
    # Format the histogram as text
    output_strings: list[str] = []

    # Add each bin and its count
    bins = column_info.histogram_bins
    counts = column_info.histogram_counts

    if not bins or not counts:
        return ""
    if len(bins) <= 1 or len(bins) != len(counts):
        logger = get_mlflow_logger()
        logger.warning("Invalid histogram data")
        return ""

    output_strings.append("\n**Distribution (Histogram):**\n")

    # Add header
    output_strings.append("| Bin Range | Count |")
    output_strings.append("|-----------|-------|")

    histogram_texts: list[str] = [
        (f"< {bins[i]:.2f}" if i == 0 else f"{bins[i - 1]:.2f} to {bins[i]:.2f}")
        for i in range(len(counts))
    ]
    output_strings = [*output_strings, *histogram_texts]

    output = "\n".join(output_strings)
    return output


def _info_string_temporal_stats_section(column_info: ColumnInfo) -> str:
    """Add temporal stats info."""
    info_parts: list[str] = []
    if column_info.temporal_min is not None and column_info.temporal_max is not None:
        info_parts.append(
            f"Range: {column_info.temporal_min} to {column_info.temporal_max}"
        )

    if column_info.temporal_diff is not None:
        days = column_info.temporal_diff.days
        seconds = column_info.temporal_diff.seconds

        num_days_in_year = 365.2425
        if days > num_days_in_year:
            time_diff_str = (
                f"{days / num_days_in_year} years, {days % num_days_in_year} days"
            )
        elif days > 0:
            time_diff_str = f"{days} days, {seconds // 3600} hours"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            time_diff_str = f"{hours} hours, {minutes} minutes, {seconds} seconds"
        info_parts.append(f"Time span: {time_diff_str}")

    output = "\n".join(info_parts)
    return output


@mlflow.trace(name="generate_info_string_for_column", span_type="func")
def generate_info_string_for_column(
    column_info: ColumnInfo,
) -> dict[str, str]:
    """Generate an information string summarizing the column analysis.

    Args:
        column_info: The analyzed column information

    Returns:
        dict[str, str]: A formatted string with key analysis findings

    """
    info_parts: list[str] = []
    info_parts.append(f"## Analysis of column: {column_info.name}")
    info_parts.append(_info_string_type_section(column_info))
    info_parts.append(_info_string_missing_values_section(column_info))

    # Type-specific info
    displayed_all_categories_flag: bool = False
    if column_info.is_categorial:
        info_parts.append(_info_string_categorial_stats_section(column_info))
        displayed_all_categories_flag = info_parts[-1].endswith("more categories")

    if column_info.is_numeric:
        # Add numeric stats
        info_parts.append(_info_string_numerical_stats_section(column_info))

        if (
            len(column_info.histogram_bins) > 0
            and len(column_info.histogram_counts) > 0
            and not displayed_all_categories_flag
        ):
            # NOTE(jdwh08): If categorical, we use the categories.
            info_parts.append(_info_string_numerical_histogram_section(column_info))

    if (
        column_info.is_temporal
        and column_info.temporal_min is not None
        and column_info.temporal_max is not None
    ):
        info_parts.append(_info_string_temporal_stats_section(column_info))

    # Join all parts with newlines
    output_str = "  \n".join(info_parts)  # "  \n" is for markdown formatting
    return {"info": output_str}
