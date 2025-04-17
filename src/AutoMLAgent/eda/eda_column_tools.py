#####################################################
# AutoMLAgent [EDA COLUMN TOOLS]
# ####################################################
# Jonathan Wang <jdwh08>

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""EDA Column Tools."""

#####################################################
### BOARD

#####################################################
### IMPORTS

import mlflow
import polars as pl

### OWN MODULES
from AutoMLAgent.dataclass.column_info import ColumnInfo
from AutoMLAgent.dataclass.column_type import ColumnType
from AutoMLAgent.dataclass.df_info import DataFrameInfo
from AutoMLAgent.eda.column.column_quality import get_data_quality_for_column
from AutoMLAgent.eda.column.column_stats import (
    get_category_levels_for_column,
    get_histogram_bins_for_column,
    get_numerical_stats_for_column,
    get_temporal_stats_for_column,
)
from AutoMLAgent.eda.column.column_types import get_type_for_column
from AutoMLAgent.eda.column.column_utils import MAX_CATEGORIES_FOR_LEVEL
from AutoMLAgent.logger.mlflow_logger import logger

#####################################################
### SETTINGS

#####################################################
### DATACLASSES


#####################################################
### CODE
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

    # Type information
    type_info = f"**Type**: {column_info.type.value}"
    if column_info.is_target_var:
        type_info += " (Target Variable)"
    elif column_info.is_feature_var:
        type_info += " (Feature)"
    info_parts.append(type_info)

    # Add missing values info
    if (
        column_info.missing_count is not None
        and column_info.missing_rate is not None
        and column_info.missing_rate > 0
    ):
        info_parts.append(
            f"Missing values: {column_info.missing_count} ({column_info.missing_rate * 100:.2f}%)"
        )
    else:
        info_parts.append("No missing values")

    # Type-specific info
    displayed_all_categories_flag: bool = False
    if column_info.is_categorial:
        num_categories = (
            len(column_info.category_counts)
            if column_info.category_counts
            else column_info.cardinality
        )
        if num_categories is not None:
            info_parts.append(f"Unique categories #: {num_categories}")

            if column_info.category_counts:
                if num_categories < MAX_CATEGORIES_FOR_LEVEL:
                    displayed_all_categories_flag = True
                    info_parts.append(
                        f"Category distribution: {column_info.category_counts}"
                    )
                else:
                    # Show top categories only
                    top_cats = dict(
                        list(column_info.category_counts.items())[
                            :MAX_CATEGORIES_FOR_LEVEL
                        ]
                    )
                    info_parts.append(f"Top categories: {top_cats} ...")
                    info_parts.append(
                        f"... and {len(column_info.category_counts) - MAX_CATEGORIES_FOR_LEVEL} more categories"
                    )

    if column_info.is_numeric:
        # Add numeric stats
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
                f"Skewness: {column_info.skewness:.4f}, Kurtosis: {column_info.kurtosis:.4f}"
            )

        if (
            len(column_info.histogram_bins) > 0
            and len(column_info.histogram_counts) > 0
            and not displayed_all_categories_flag  # if categorical, we use the categories.
        ):
            info_parts.append("\n**Distribution (Histogram):**")

            # Format the histogram as text
            histogram_text: list[str] = []

            # Add header
            histogram_text.append("| Bin Range | Count |")
            histogram_text.append("|-----------|-------|")

            # Add each bin and its count
            bins = column_info.histogram_bins
            counts = column_info.histogram_counts

            if len(bins) > 1 and len(bins) == len(counts):
                for i in range(len(counts)):
                    if i == 0:
                        # First bin shows the minimum value
                        bin_range = f"< {bins[i]:.2f}"
                    else:
                        bin_range = f"{bins[i - 1]:.2f} to {bins[i]:.2f}"

                    histogram_text.append(f"| {bin_range} | {counts[i]} |")

            info_parts.append("\n".join(histogram_text))

    if (
        column_info.is_temporal
        and column_info.temporal_min is not None
        and column_info.temporal_max is not None
    ):
        # Add temporal stats
        info_parts.append(
            f"{column_info.type} range: {column_info.temporal_min} to {column_info.temporal_max}"
        )

        if not column_info.temporal_diff:
            # Calculate time span for datetimes
            column_info.temporal_diff = (
                column_info.temporal_max - column_info.temporal_min
            )  # type: ignore

        days = column_info.temporal_diff.days  # type: ignore
        seconds = column_info.temporal_diff.seconds  # type: ignore

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

    # Join all parts with newlines
    output_str = "  \n".join(info_parts)  # "  \n" is for markdown formatting
    return {"info": output_str}


@mlflow.trace(name="analyze_column", span_type="func")
def analyze_column(
    df: pl.DataFrame,
    column_name: str,
    *,
    column_info: ColumnInfo | None = None,
    df_info: DataFrameInfo | None = None,
    outlier_zscore_threshold: float = 3.0,
    low_variance_threshold: float = 0.01,
) -> ColumnInfo:
    """Analyze a single variable column in a dataframe.

    This function composes multiple specialized analysis functions to provide
    a comprehensive analysis of a column based on its type.

    Args:
        df: Dataframe containing the variable
        column_name: Name of the column to analyze
        column_info: Pre-existing ColumnInfo to update (optional)
        df_info: Pre-existing DataFrameInfo from which to extract column_info (optional)
        outlier_zscore_threshold: Threshold for outlier detection
        low_variance_threshold: Threshold for low variance detection

    Returns:
        Updated ColumnInfo with comprehensive analysis

    """
    # Validate inputs
    if column_name not in df.columns:
        msg = f"Column {column_name} not found in dataframe"
        logger.exception(msg)
        raise ValueError(msg)

    # Get or create column_info
    if column_info is None and df_info is not None:
        column_info = next(
            (c for c in df_info.column_info if c.name == column_name), None
        )

    if column_info is None:
        msg = f"Column {column_name} not found in column_info or df_info"
        logger.exception(msg)
        raise ValueError(msg)

    # Generate warning if column is analyzed already.
    if column_info.is_analyzed:
        msg = f"Column {column_name} has already been analyzed"
        logger.warning(msg)
        return column_info

    try:
        # Step 1: Get column type if not already determined
        if column_info.type == ColumnType.UNKNOWN:
            type_data = get_type_for_column(df, column_name, column_info=column_info)
            column_info = column_info.model_copy(update=type_data)

        # Step 2: Get missing values
        if column_info.missing_count is None:
            total_rows = len(df)
            missing_count = df[column_name].null_count()
            column_info.missing_count = missing_count
            column_info.missing_rate = (
                missing_count / total_rows if total_rows > 0 else 0.0
            )

        # Step 3: Apply type-specific analyses
        if column_info.is_numeric:
            # Get numerical statistics
            try:
                numeric_stats = get_numerical_stats_for_column(df, column_name)
                column_info = column_info.model_copy(update=numeric_stats)
            except Exception:
                logger.exception(f"Failed to calculate numeric stats for {column_name}")

            # Get histogram data
            try:
                histogram_data = get_histogram_bins_for_column(df, column_name)
                if histogram_data:
                    column_info = column_info.model_copy(update=histogram_data)
            except Exception:
                logger.exception(f"Failed to create histogram for {column_name}")

        if column_info.is_categorial:
            # Get category levels and counts
            try:
                category_data = {
                    "category_counts": get_category_levels_for_column(df, column_name)
                }
                column_info = column_info.model_copy(update=category_data)
            except Exception:
                logger.exception(f"Failed to get category levels for {column_name}")

        if column_info.is_temporal:
            # Get temporal statistics
            try:
                temporal_stats = get_temporal_stats_for_column(df, column_name)
                column_info = column_info.model_copy(update=temporal_stats)
            except Exception:
                logger.exception(
                    f"Failed to calculate temporal stats for {column_name}"
                )

        # Step 4: Add data quality metrics
        if column_info.outlier_count is None:
            try:
                quality_metrics = get_data_quality_for_column(
                    df,
                    column_name,
                    column_info=column_info,
                    outlier_zscore_threshold=outlier_zscore_threshold,
                    low_variance_threshold=low_variance_threshold,
                )
                column_info = column_info.model_copy(update=quality_metrics)
            except Exception:
                logger.exception(
                    f"Failed to calculate data quality metrics for {column_name}"
                )

        # Step 5: Generate info summary string based on column type
        column_info.is_analyzed = True
        column_info_string = generate_info_string_for_column(column_info)
        column_info = column_info.model_copy(update=column_info_string)

    except Exception:
        logger.exception(f"Failed to analyze column {column_name}")
        column_info.is_analyzed = False

    return column_info
