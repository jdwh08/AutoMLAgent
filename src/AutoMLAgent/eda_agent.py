#####################################################
# AutoMLAgent [BASIC EDA AGENT]
# ####################################################
# Jonathan Wang <jdwh08>

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Basic EDA Agent."""

#####################################################
### BOARD

# TODO(jdwh08): update kickoff prompt for full df
# TODO(jdwh08): once EDA works, make this for all other agents
# TODO(jdwh08): update to local / more open llm when money

#####################################################
### IMPORTS
import asyncio
from dataclasses import dataclass
from pathlib import Path

import mlflow
import polars as pl
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

### OWN MODULES
from automlagent.dataclass.column_info import create_column_info
from automlagent.eda.eda_column_tools import analyze_column
from automlagent.logger.mlflow_logger import get_mlflow_logger

#####################################################
### SETTINGS

# ENVIRONMENT SETUP
# Secrets and keys
load_dotenv()

# MLFlow
# mlflow.openai.autolog()  # noqa: ERA001
mlflow.gemini.autolog()  # NOTE(jdwh08): doesn't really work
mlflow.autolog()  # for modelling tasks
mlflow.set_experiment("PYDANTIC AGENTS TEST v1 EDA")
mlflow.config.enable_async_logging()  # type: ignore[syntax, reportPrivateImportUsage] <mlflow docs recommends this>

# TODO(jdwh08): update model to more local / open
model = (
    "gemini-2.0-flash-lite"  # free for now, don't sue me, i only computer a potato :(
)


#####################################################
### DATACLASSES
@dataclass
class DfEdaDependencies:  # EDA DataFrame and variables
    file_path: Path
    df: pl.DataFrame
    target_var: str
    feature_vars: list[str]


#####################################################
### AGENT
eda_agent = Agent(
    name="EDA Agent",
    model=model,
    model_settings={
        # NOTE(jdwh08): temperature > 0 for tool calling to work.
        "max_tokens": 4096,
        "timeout": 600.0,
        "parallel_tool_calls": False,
    },
    system_prompt=(
        "You are a veteran Senior Data Scientist with years of experience conducting "
        "exploratory data analysis on datasets.\n\n"
        "Conduct extensive exploratory data analysis on the user-provided dataset "
        "using the provided tools. "
        "Based on the results, write a detailed report to help further  "
        "feature engineering and building a machine learning model on this dataset."
    ),
    deps_type=DfEdaDependencies,
)


#####################################################
### TOOLS
@eda_agent.tool
@mlflow.trace(name="eda_column_tool", span_type="tool")
async def eda_column_tool(ctx: RunContext[DfEdaDependencies], column_name: str) -> str:
    """Conduct exploratory analysis on a single column.

    Args:
        ctx: Run context with dependencies
        column_name: Name of the column to analyze

    Returns:
        str: Formatted analysis of the specified column

    """
    df: pl.DataFrame = ctx.deps.df

    # Validate column exists
    if column_name not in df.columns:
        return f"Error: Column '{column_name}' not found in the dataset."

    # Determine variable type
    is_target_var = column_name == ctx.deps.target_var
    column_info = create_column_info(
        column_name,
        is_target_var=is_target_var,
        is_feature_var=not is_target_var,
    )

    try:
        # Analyze the column
        column_info = analyze_column(
            df=df, column_name=column_name, column_info=column_info
        )
    except Exception as e:
        logger = get_mlflow_logger()
        logger.exception(f"Failed to analyze column {column_name}")
        return f"Error analyzing column '{column_name}': {e!s}"
    else:
        # Return the results
        return column_info.info


@mlflow.trace(name="main", span_type="entry")
async def main(
    file_path: Path,
    target_var: str,
    feature_vars: list[str] | None = None,
) -> None:
    """Run EDA on a dataset."""
    with mlflow.start_run():
        logger = get_mlflow_logger()

        with mlflow.start_span(name="data_loading", span_type="data") as span:
            span.set_inputs({"file_path": str(file_path)})
            credit_df = pl.read_csv(file_path)
            if feature_vars is None:
                feature_vars = [col for col in credit_df.columns if col != target_var]
            deps = DfEdaDependencies(
                file_path=file_path,
                df=credit_df,
                target_var=target_var,
                feature_vars=feature_vars,
            )
            span.set_outputs(deps)

        # TODO(jdwh08): update kickoff prompt for full df
        kickoff_prompt: str = (
            "Help me conduct extensive exploratory data analysis for the target variable "
            f"of the dataset, called {target_var}."
        )

        result = await eda_agent.run(
            kickoff_prompt,
            deps=deps,
        )
        output = f"Response: {result.data}"
        logger.info(output)


if __name__ == "__main__":
    file_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "credit_risk_dataset"
        / "credit_risk_dataset.csv"
    )
    asyncio.run(main(file_path, target_var="loan_status"))
