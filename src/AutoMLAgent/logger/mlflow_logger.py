#####################################################
# AutoMLAgent [BASIC AGENT]
# ####################################################
# Jonathan Wang <jdwh08>

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Integrating MLflow into Python logging."""

# NOTE: referencing <https://theforce.hashnode.dev/integrating-mlflow-into-python-logging>

#####################################################
### BOARD

#####################################################
### IMPORTS

import logging
from typing import ClassVar

import mlflow

### OWN MODULES
from automlagent.utils.typing_utils import check_kwargs


####################################################
### CLASS
class MLFlowLogger(logging.Logger):
    """Integrating MLflow into Python logging."""

    mlflow_log_level: ClassVar[int] = logging.INFO + 5

    def __init__(
        self,
        name: str = "mlflow",
        run_id: str | None = None,
        level: int = logging.DEBUG,
    ) -> None:
        """Initialize the logger."""
        # Initialize the parent Logger first with only parameters it accepts
        super().__init__(name, level)

        # Get MLFlow run
        if run_id is None:
            active_run = mlflow.active_run()
            if active_run is None:
                msg = "No active MLFlow run. Please start a run before logging."
                raise RuntimeError(msg)
            run_id = str(active_run.info.run_id)  # type: ignore[reportUnknownMemberType, unused-ignore]

        # Add custom logging level for MLflow
        logging.addLevelName(self.mlflow_log_level, "MLFLOW")

        # Log the run ID using the custom level
        self._log_mlflow(f"Run ID: {run_id}")

    def _log_mlflow(self, message: str) -> None:
        """Log messages at the MLflow level."""
        if self.isEnabledFor(self.mlflow_log_level):
            self._log(level=self.mlflow_log_level, msg=message, args=(), stacklevel=2)

    def log_param(self, param: str, value: object) -> None:
        """Log a parameter to MLflow and to the logger."""
        mlflow.log_param(param, value)
        self._log_mlflow(f"PARAM - {param}: {value!s}")

    def log_metric(self, metric: str, value: float, **kwargs: object) -> None:
        """Log a metric to MLflow and to the logger."""
        filtered_kwargs = check_kwargs(kwargs, mlflow.log_metric)
        mlflow.log_metric(metric, value, **filtered_kwargs)  # type: ignore[arg-type, unused-ignore, reportArgumentType]
        self._log_mlflow(f"METRIC - {metric}: {value!s}")


class RainbowFormatter(logging.Formatter):
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[1;32m",
        "INFO": "\033[1;35m",
        "WARNING": "\033[1;33m",
        "ERROR": "\033[1;31m",
        "CRITICAL": "\033[1;41m",
        "MLFLOW": "\033[1;45m",
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        color = self.COLORS.get(record.levelname, "\033[1;0m")
        return f"{color}{record.levelname}\033[1;0m: {msg}"


#####################################################
### SETTINGS
_logger_instance: MLFlowLogger | None = None


#####################################################
### CODE
def get_mlflow_logger(
    name: str = "mlflow",
    run_id: str | None = None,
    level: int = logging.DEBUG,
    *,
    add_console_handler: bool = True,
    formatter: logging.Formatter | None = None,
) -> MLFlowLogger:
    """Create and configure an MLFlowLogger.

    Args:
        name: Logger name
        run_id: Optional MLflow run ID to use
        level: Logging level
        add_console_handler: Whether to add a console handler
        formatter: Optional formatter to use, defaults to RainbowFormatter

    Returns:
        Configured MLFlowLogger instance

    Note:
        Singleton global MLFlowLogger instance.
        Should only create logger if doesn't exist yet.
        Done this way so that we always have a MLflow run to log to.

    """
    # BUG(jdwh08): Singleton global logger, not sure about multi-threading/process!
    global _logger_instance  # noqa: PLW0603

    if _logger_instance is None:
        _logger_instance = MLFlowLogger(name=name, run_id=run_id, level=level)

        if add_console_handler:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # Use RainbowFormatter by default
            if formatter is None:
                formatter = RainbowFormatter()

            console_handler.setFormatter(formatter)
            _logger_instance.addHandler(console_handler)

    return _logger_instance
