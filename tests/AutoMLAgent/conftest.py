#####################################################
# AutoMLAgent [TEST CONFIGURATION]
# ####################################################
# Jonathan Wang <jdwh08>

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Base test configuration."""

#####################################################
### BOARD

# NOTE(jdwh08): MLFLOW LOGGING DISABLE FOR TESTS HERE
# DISABLE_MLFLOW_LOGGING_FOR_TESTS
# set to "1" to disable

#####################################################
### IMPORTS
import os

#####################################################
### SETTINGS


def pytest_configure() -> None:
    # NOTE(jdwh08): don't want to setup a mlflow.run() for testing...
    # To actually test mlflow logging, monkeypatch environment at beginning:
    # def test_with_real_mlflow(monkeypatch):
    #     with mlflow.run():
    #         monkeypatch.setenv("DISABLE_MLFLOW_LOGGING_FOR_TESTS","0")  # noqa: ERA001
    os.environ["DISABLE_MLFLOW_LOGGING_FOR_TESTS"] = "1"
