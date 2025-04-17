# PLANNING

## Project Overview

Develop a modular, agentic AutoML system for tabular datasets, capable of automating the end-to-end machine learning workflow—including data profiling, feature engineering, model selection, training, validation, and reporting. The system should be extensible, production-ready, and competitive for real-world benchmarks (e.g., Kaggle competitions).

## Architecture

- **Agent Framework:** Pydantic AI Agents ([docs](https://ai.pydantic.dev/agents/)), leveraging type-safe, composable agents for each pipeline stage.
- **Core Components:**
  - Data ingestion & validation (Pydantic v2 models)
  - Exploratory Data Analysis (EDA) agent
  - Feature engineering agent
  - Model selection & training agent
  - Hyperparameter optimization agent
  - Evaluation/reporting agent
  - Orchestration agent to coordinate workflow
- **Logging & Experiment Tracking:** MLflow
- **Configuration:** pyproject.toml (Hatchling for builds)
- **Testing:** pytest (unit & integration tests)
- **Linting/Formatting:** Ruff, UV
- **Python Version:** 3.11+
- **Validation:** Pydantic v2 throughout

## Constraints & Best Practices

- Prioritize readability, maintainability, and extensibility
- Avoid overengineering; build minimal, composable agents first
- Adhere strictly to Python 3.11+ idioms and typing
- Use Pydantic v2 models for all config, data, and results
- Ensure all agents/tools are type-safe and statically checkable (mypy/pyright)
- Modularize code for easy testing and replacement of agents

## Tech Stack

- Python 3.11+
- Pydantic v2, Pydantic AI Agents
- MLflow
- pytest
- Ruff, UV
- Hatchling

## Tools & Integrations

- MLflow for experiment tracking
- Pydantic AI Agents for agent orchestration and tool definition
- pyproject.toml for dependency/build management

## Future Considerations

- Time series specific modelling
- Support for additional data modalities (time series, text, images)
- Integration with cloud compute (optional)
- Plugin system for custom agents/tools
