#####################################################
# AutoMLAgent [EDA COLUMN INFO STRING TESTS]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Unit tests for column info string."""

#####################################################
### BOARD

#####################################################
### IMPORTS

import datetime

import pytest

### OWN MODULES
from automlagent.dataclass.column_info import ColumnInfo
from automlagent.dataclass.column_type import ColumnType
from automlagent.eda.column.column_info_string import (
    _info_string_categorial_stats_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    _info_string_missing_values_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    _info_string_numerical_histogram_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    _info_string_numerical_stats_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    _info_string_temporal_stats_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    _info_string_type_section,  # type: ignore[reportPrivateUsage, unused-ignore]
    generate_info_string_for_column,
)
from automlagent.eda.column.column_utils import MAX_CATEGORIES_FOR_LEVEL


#####################################################
### CODE
class TestColumnInfoString:
    def test_type_section_feature(self) -> None:
        ci = ColumnInfo(
            name="col",
            type=ColumnType.STRING,
            is_feature_var=True,
            is_target_var=False,
        )
        result = _info_string_type_section(ci)
        assert result == "**Type**: string (Feature)"

    def test_type_section_target(self) -> None:
        ci = ColumnInfo(
            name="col",
            type=ColumnType.INTEGER,
            is_feature_var=False,
            is_target_var=True,
        )
        result = _info_string_type_section(ci)
        assert result == "**Type**: integer (Target Variable)"

    @pytest.mark.parametrize(
        ("missing_count", "missing_rate", "expected"),
        [
            (None, None, "No missing values"),
            (0, 0.0, "No missing values"),
            (3, 0.25, "Missing values: 3 (25.00%)"),
        ],
    )
    def test_missing_values_section(
        self, missing_count: int, missing_rate: float, expected: str
    ) -> None:
        ci = ColumnInfo(
            name="col",
            missing_count=missing_count,
            missing_rate=missing_rate,
        )
        result = _info_string_missing_values_section(ci)
        assert result == expected

    def test_categorical_section_empty(self) -> None:
        ci = ColumnInfo(name="col", category_counts={}, cardinality=None)
        result = _info_string_categorial_stats_section(ci)
        assert result == ""

    def test_categorical_section_small(self) -> None:
        counts = {"a": 1, "b": 2}
        ci = ColumnInfo(name="col", category_counts=counts, cardinality=None)
        result = _info_string_categorial_stats_section(ci)
        assert "Unique categories #: 2" in result
        assert "Category distribution: {'a': 1, 'b': 2}" in result

    def test_categorical_section_large(self) -> None:
        counts = {f"cat{i}": i for i in range(MAX_CATEGORIES_FOR_LEVEL + 1)}
        ci = ColumnInfo(name="col", category_counts=counts, cardinality=None)
        result = _info_string_categorial_stats_section(ci)
        assert f"Unique categories #: {MAX_CATEGORIES_FOR_LEVEL + 1}" in result
        assert "Top categories" in result
        assert f"and {1} more categories" in result

    def test_numerical_stats_section_full(self) -> None:
        ci = ColumnInfo(
            name="col",
            min=1.0,
            max=5.0,
            mean=3.0,
            median=3.0,
            std=1.0,
            q1=2.0,
            q3=4.0,
            skewness=0.0,
            kurtosis=0.01,
        )
        result = _info_string_numerical_stats_section(ci)
        lines = result.split("\n")
        assert "Range: 1.0 to 5.0" in lines
        assert "Mean: 3.0000, Median: 3.0000" in lines
        assert "Std Dev: 1.0000" in lines
        assert "Q1: 2.0000, Q3: 4.0000" in lines
        assert "Skewness: 0.0000, Kurtosis: 0.0100" in lines

    def test_numerical_stats_section_partial(self) -> None:
        ci = ColumnInfo(name="col", min=0.0, max=10.0)
        result = _info_string_numerical_stats_section(ci)
        assert result == "Range: 0.0 to 10.0"

    def test_numerical_histogram_section_valid(self) -> None:
        bins = [1.0, 2.0, 3.0]
        counts = [5, 10, 15]
        ci = ColumnInfo(
            name="col",
            histogram_bins=bins,
            histogram_counts=counts,
        )
        result = _info_string_numerical_histogram_section(ci)
        assert "Distribution (Histogram)" in result
        assert "| Bin Range | Count |" in result
        assert "< 1.00" in result
        assert "1.00 to 2.00" in result
        assert "2.00 to 3.00" in result

    @pytest.mark.parametrize(
        ("histogram_bins", "histogram_counts"),
        [
            ([1.0], [1]),
            ([1.0, 2.0], [1, 4, 1]),
        ],
    )
    def test_numerical_histogram_section_invalid(
        self,
        *,
        histogram_bins: list[float],
        histogram_counts: list[int],
    ) -> None:
        """Covers exception when histogram data is invalid."""
        ci = ColumnInfo(
            name="col",
            histogram_bins=histogram_bins,
            histogram_counts=histogram_counts,
        )
        result = _info_string_numerical_histogram_section(ci)
        assert result == ""

    def test_temporal_stats_section_range_only(self) -> None:
        date_min = datetime.date(2020, 1, 1)
        date_max = datetime.date(2020, 1, 5)
        ci = ColumnInfo(
            name="col",
            temporal_min=date_min,
            temporal_max=date_max,
        )
        result = _info_string_temporal_stats_section(ci)
        assert result == "Range: 2020-01-01 to 2020-01-05"

    def test_temporal_stats_section_with_diff_days_hours(self) -> None:
        date_min = datetime.date(2020, 1, 1)
        date_max = datetime.date(2020, 1, 3)
        diff = datetime.timedelta(days=2, seconds=3600)
        ci = ColumnInfo(
            name="col",
            temporal_min=date_min,
            temporal_max=date_max,
            temporal_diff=diff,
        )
        result = _info_string_temporal_stats_section(ci)
        lines = result.split("\n")
        assert "Range: 2020-01-01 to 2020-01-03" in lines
        assert "Time span: 2 days, 1 hours" in lines

    def test_temporal_stats_section_with_diff_hours_min_sec(self) -> None:
        diff = datetime.timedelta(seconds=3661)
        ci = ColumnInfo(name="col", temporal_diff=diff)
        result = _info_string_temporal_stats_section(ci)
        assert result == "Time span: 1 hours, 1 minutes, 1 seconds"

    def test_generate_info_string_default(self) -> None:
        """Covers default case."""
        ci = ColumnInfo(name="col")
        result = generate_info_string_for_column(ci)
        info = result["info"]
        assert info.startswith("## Analysis of column: col")
        assert "**Type**: unknown (Feature)" in info
        assert "No missing values" in info
        assert "Unique categories" not in info
        assert "Mean:" not in info
        assert "Time span:" not in info

    @pytest.mark.parametrize(
        ("is_categorial", "is_numeric", "is_temporal", "expected_parts"),
        [
            (True, False, False, ["Unique categories"]),
            (False, True, False, ["Range:", "Mean:"]),
            (False, False, True, ["Range:", "Time span:"]),
        ],
    )
    def test_generate_info_string_for_column(
        self,
        *,
        is_categorial: bool,
        is_numeric: bool,
        is_temporal: bool,
        expected_parts: list[str],
    ) -> None:
        ci = ColumnInfo(
            name="col",
            is_categorial=is_categorial,
            is_numeric=is_numeric,
            is_temporal=is_temporal,
        )
        if is_categorial:
            ci.category_counts = {"a": 1, "b": 2}
        if is_numeric:
            ci.histogram_bins = [0.0, 1.0]
            ci.histogram_counts = [1, 1]
            ci.min = 0.0
            ci.max = 1.0
            ci.mean = 0.5
            ci.median = 0.5
            ci.std = 0.1
            ci.q1 = 0.25
            ci.q3 = 0.75
        if is_temporal:
            ci.temporal_min = datetime.date(2020, 1, 1)
            ci.temporal_max = datetime.date(2020, 1, 2)
            ci.temporal_diff = datetime.timedelta(days=1)
        result = generate_info_string_for_column(ci)
        info = result["info"]
        for part in expected_parts:
            assert part in info
