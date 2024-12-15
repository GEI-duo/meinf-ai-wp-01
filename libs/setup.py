import sys
import logging
import pytz
import pandas as pd

from joblib import dump
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Literal, Union


@dataclass(frozen=True)
class RunPaths:
    base: Path
    raw_data: Path
    processed_data: Path
    models: Path
    logs: Path

    @classmethod
    def of(
        cls,
        base: Path,
        data: str = "data",
        raw_data: str = "raw",
        processed_data: str = "processed",
        models: str = "models",
        logs: str = "logs",
    ) -> "RunPaths":
        return RunPaths(
            base=base,
            raw_data=Path.joinpath(base, data, raw_data),
            processed_data=Path.joinpath(base, data, processed_data),
            models=Path.joinpath(base, models),
            logs=Path.joinpath(base, logs),
        )


class Run:

    RUN_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
    LOG_TIMESTAMP_FORMAT = "%H:%M:%S"
    LOGS_FILENAME = "training"
    MODELS_OUTPUT_DIR = "models"

    def __init__(
        self,
        base_runs_path: Path,
        timezone: str = "Europe/Madrid",
        run_timestamp_format: str = RUN_TIMESTAMP_FORMAT,
        log_timestamp_format: str = LOG_TIMESTAMP_FORMAT,
        logs_filename: str = LOGS_FILENAME,
        logger: Union[logging.Logger, None] = None,
        log_formatter: Union[logging.Formatter, None] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)

        self._timezone = pytz.timezone(timezone)
        self._base_runs_path = base_runs_path
        self._timestamp_format = run_timestamp_format
        self._log_filename = logs_filename
        self._log_formatter = (
            log_formatter
            if log_formatter
            else logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt=log_timestamp_format,
            )
        )

        self._paths = None

    def _now(self) -> datetime:
        return datetime.now(tz=self._timezone)

    @property
    def paths(self) -> RunPaths:
        if not self._paths:
            raise ValueError("Run has not been initialized yet.")
        return self._paths

    def initialize(
        self,
        verbosity: Literal["quiet", "normal", "verbose"] = "verbose",
    ) -> "Run":
        """Creates the run output directories and sets up the logger.

        Args:
            verbosity: The verbosity level for the logger.

        Returns:
            RunPaths: the paths to the run output directories.
        """

        self._paths = self._setup_run(verbosity)
        return self

    def add_data(self, data: pd.DataFrame, filename: str, path: Path) -> Path:
        """Adds the data to the run directory.

        Args:
            data (pd.DataFrame): The data to save.
            filename (str): The filename to use for the data (without extension).
            path (Path): The path to save the data.

        Returns: The path to the data in the run directory.
        """
        data_path = Path.joinpath(path, f"{filename}.csv")
        data.to_csv(data_path)
        self.logger.info(f"Data saved to {data_path}")
        return data_path

    def add_raw(self, data: pd.DataFrame, filename: str) -> Path:
        """Adds the raw data to the run directory.

        Args:
            data (pd.DataFrame): The raw data to save.
            filename (str): The filename to use for the raw data (without extension).

        Returns: The path to the raw data in the run directory.
        """
        return self.add_data(data, filename, self.paths.raw_data)

    def add_processed(self, data: pd.DataFrame, filename: str) -> Path:
        """Adds the processed data to the run directory.

        Args:
            data (pd.DataFrame): The processed data to save.
            filename (str): The filename to use for the processed data (without extension).

        Returns:
            Path: The path to the processed data in the run directory.
        """
        return self.add_data(data, filename, self.paths.processed_data)

    def add_model(self, model, filename: str) -> Path:
        """Adds the model to the run directory.

        Args:
            model: The model to save.
            filename (str): The filename to use for the model (without extension).

        Returns: The path to the model in the run directory.
        """
        model_path = Path.joinpath(self.paths.models, filename)
        dump(model, model_path, compress=True)
        self.logger.info(f"Model saved to {model_path}")
        return model_path

    def _init_logger(
        self,
        verbosity: Literal["quiet", "normal", "verbose"],
        run_path: Path,
    ) -> None:
        if verbosity == "quiet":
            self.logger.setLevel(logging.ERROR)
        elif verbosity == "normal":
            self.logger.setLevel(logging.INFO)
        elif verbosity == "verbose":
            self.logger.setLevel(logging.DEBUG)

        # Log to console and file and with time format
        file_handler = logging.FileHandler(
            "{0}/{1}.log".format(
                run_path,
                self._log_filename,
            )
        )
        console_handler = logging.StreamHandler(sys.stdout)

        file_handler.setFormatter(self._log_formatter)
        console_handler.setFormatter(self._log_formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _setup_run(self, verbosity: Literal["quiet", "normal", "verbose"]) -> RunPaths:
        run_id = self._now().strftime(self._timestamp_format)

        # Append the run id to the output models path
        run_path = Path.joinpath(self._base_runs_path, run_id)
        run_path.mkdir(parents=True, exist_ok=True)

        self._init_logger(verbosity, run_path)

        run_paths = RunPaths.of(run_path)
        for path in [run_paths.raw_data, run_paths.processed_data, run_paths.models]:
            Path.joinpath(run_path, path).mkdir(parents=True, exist_ok=True)

        return run_paths
