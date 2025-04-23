# TASKS

## Active Work

### Develop EDA agent for profiling and reporting

- Handle table-level EDA such as correlations, tests, and feature relationships

- Feature Correlation
  - Pearson Correlation (numeric, numeric)
  - Cramer's V (categorical, categorical)
  - ANOVA F statistic? (numeric, categorical)
  - Mutual Information

- DataFrame level EDA
  - VIF? Highly colinear features?
  - Grouped Stats
  - Leakage against target var (if feature correlation is too high)
  - Outlier analysis between columns?
  - Missing value analysis / co-occurance between columns?
  - Duplicate or very similar row count?
  - SVD Rank?

- Data Quality Improvements (DataFrame level)
  - Duplicate Rows
