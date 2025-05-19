#####################################################
# AutoMLAgent [MLFLOW LOGGER TESTS]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""MLFlow Logger tests."""

#####################################################
### BOARD

# TODO(jdwh08): Add tests for partially None values
# TODO(jdwh08): Add tests for partially NaN values

#####################################################
### IMPORTS
import logging
import os
from unittest import mock

import pytest

### OWN MODULES
from automlagent.logger.mlflow_logger import MLFlowLogger


#####################################################
### CODE
class TestMLFlowLogger:
    def test_caplog_control(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            logging.getLogger("some_test_logger").info("hello world")
        assert any("hello world" in r.message for r in caplog.records)

    def test_init_raises_without_active_run(self) -> None:
        with (
            mock.patch.dict(os.environ, {"DISABLE_MLFLOW_LOGGING_FOR_TESTS": "0"}),
            mock.patch(
                "automlagent.logger.mlflow_logger.mlflow.active_run", return_value=None
            ),
            pytest.raises(RuntimeError, match="No active MLFlow run"),
        ):
            MLFlowLogger()

    def test_init_no_raise_when_disabled(self) -> None:
        with mock.patch.dict(os.environ, {"DISABLE_MLFLOW_LOGGING_FOR_TESTS": "1"}):
            logger = MLFlowLogger()
            assert logger.propagate is True
            assert logger.handlers == []

    def test_log_param_calls_mlflow_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake_run = mock.Mock()
        with (
            mock.patch.dict(os.environ, {"DISABLE_MLFLOW_LOGGING_FOR_TESTS": "0"}),
            mock.patch(
                "automlagent.logger.mlflow_logger.mlflow.log_param"
            ) as mlog_param,
            mock.patch(
                "automlagent.logger.mlflow_logger.mlflow.active_run",
                return_value=fake_run,
            ),
            caplog.at_level(MLFlowLogger.mlflow_log_level),
        ):
            logger = MLFlowLogger()
            logger.addHandler(caplog.handler)
            logger.log_param("foo", 123)
            mlog_param.assert_called_once_with("foo", 123)
            assert any("PARAM - foo: 123" in r.message for r in caplog.records)
        with (
            mock.patch.dict(os.environ, {"DISABLE_MLFLOW_LOGGING_FOR_TESTS": "1"}),
            mock.patch(
                "automlagent.logger.mlflow_logger.mlflow.log_param"
            ) as mlog_param,
        ):
            logger = MLFlowLogger()
            logger.log_param("bar", 456)
            mlog_param.assert_not_called()

    def test_log_metric_calls_mlflow_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            mock.patch.dict(os.environ, {"DISABLE_MLFLOW_LOGGING_FOR_TESTS": "0"}),
            mock.patch(
                "automlagent.logger.mlflow_logger.mlflow.log_metric"
            ) as mlog_metric,
            mock.patch(
                "automlagent.logger.mlflow_logger.mlflow.active_run"
            ) as mactive_run,
            mock.patch(
                "automlagent.logger.mlflow_logger.check_kwargs",
                side_effect=lambda k, f: k,  # type: ignore[reportUnknownLambdaType]  # noqa: ARG005
            ),
        ):
            fake_run = mock.Mock()
            fake_run.info.run_id = "run77"
            mactive_run.return_value = fake_run
            logger = MLFlowLogger()
            logger.addHandler(caplog.handler)
            with caplog.at_level(MLFlowLogger.mlflow_log_level):
                logger.log_metric("acc", 0.99, step=1)
                mlog_metric.assert_called_once_with("acc", 0.99, step=1)
                assert any("METRIC - acc: 0.99" in r.message for r in caplog.records)

    def test_log_metric_skips_mlflow_when_disabled(self) -> None:
        with (
            mock.patch.dict(os.environ, {"DISABLE_MLFLOW_LOGGING_FOR_TESTS": "1"}),
            mock.patch(
                "automlagent.logger.mlflow_logger.mlflow.log_metric"
            ) as mlog_metric,
        ):
            logger = MLFlowLogger()
            logger.log_metric("loss", 0.1, step=2)
            mlog_metric.assert_not_called()

    def test_custom_log_level_registered(self) -> None:
        level_num = MLFlowLogger.mlflow_log_level
        assert logging.getLevelName(level_num) == "MLFLOW"

    def test_log_param_and_metric_log_even_when_disabled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            mock.patch.dict(os.environ, {"DISABLE_MLFLOW_LOGGING_FOR_TESTS": "1"}),
            mock.patch(
                "automlagent.logger.mlflow_logger.mlflow.log_param"
            ) as mlog_param,
            mock.patch(
                "automlagent.logger.mlflow_logger.mlflow.log_metric"
            ) as mlog_metric,
        ):
            logger = MLFlowLogger()
            logger.addHandler(caplog.handler)
            logger.log_param("baz", "val")
            logger.log_metric("score", 1.23)
            mlog_param.assert_not_called()
            mlog_metric.assert_not_called()
            assert any("PARAM - baz: val" in r.message for r in caplog.records)
            assert any("METRIC - score: 1.23" in r.message for r in caplog.records)
