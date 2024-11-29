import os
import sys
import logging
import pandas as pd
import tzdata

from joblib import dump
from pathlib import Path
from datetime import datetime
from typing import Literal, List
from dataclasses import dataclass, Field

from libs.types import StatisticsInfo, TrainingInfo


@dataclass
class Trainee:
    training_info: TrainingInfo
    statistics_info: StatisticsInfo

    directory_timestamp_format: str = "%Y-%m-%d_%H-%M"

    log_filename: str = "training.log"
    _logger: logging.Logger = Field(default_factory=logging.getLogger(__name__))

    def _init_logger(
        self,
        verbosity: Literal["quiet", "normal", "verbose"],
        run_path: Path,
        file_name,
    ) -> None:
        if verbosity == "quiet":
            self._logger.setLevel(logging.ERROR)
        elif verbosity == "normal":
            self._logger.setLevel(logging.INFO)
        elif verbosity == "verbose":
            self._logger.setLevel(logging.DEBUG)

        # Log to console and file and with time format
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(
            "{0}/{1}.log".format(
                Path.joinpath(run_path, "logs"),
                self.log_filename,
            )
        )
        console_handler = logging.StreamHandler(sys.stdout)

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)

    def _setup_run(
        self,
        verbosity: Literal["quiet", "normal", "verbose"],
    ) -> str:
        """Initalizes the needed directories and logger for the training run.

        Args:
            verbosity (Literal[&quot;quiet&quot;, &quot;normal&quot;, &quot;verbose&quot;]): logger verbosity level.

        Returns:
            str: the run path to store the training run.
        """

        run_id = datetime.now().strftime(self.directory_timestamp_format)

        # Append the run id to the output models path
        run_path = Path.joinpath(self.training_info.output_path, run_id)
        run_path.mkdir(parents=True, exist_ok=True)

        self._init_logger(verbosity, file_handler)

        return run_path

    def run(
        self,
        run_data: RunInfo,
        model_info: List[ModelInfo],
        data: pd.Dataframe,
        verbosity: Literal["quiet", "normal", "verbose"] = "verbose",
    ) -> Path:
        """


        Args:
            model_info (List[ModelInfo]): _description_
            data (pd.Dataframe): _description_
            verbosity (Literal[&quot;quiet&quot;, &quot;normal&quot;, &quot;verbose&quot;], optional): _description_. Defaults to "verbose".

        Returns:
            Path: _description_
        """

        run_id = self._setup_run(verbosity)

        if not dev_config.MODELLING_OVERWRITE_MODELS and os.path.exists(
            dev_config.TRAIN_RESULTS_PATH
        ):
            train_results = pd.read_csv(dev_config.TRAIN_RESULTS_PATH, index_col=0)
        else:
            train_results = pd.DataFrame(
                columns=["Model Name", "Training Time", "File Size"]
            )

        # Train models
        for num, model in enumerate(models, start=1):
            model_path = os.path.join(dev_config.MODELS_FOLDER, f"{model}.bin")

            # Train only if it has not been trained before
            if not dev_config.MODELLING_OVERWRITE_MODELS and os.path.exists(model_path):
                models[model].model = load(model_path)
            else:
                # Track duration
                t0 = time.time()
                models[model].model.fit(x_train, y_train)
                t = time.time()
                duration = str(timedelta(seconds=int(t - t0)))

                # plot_learning_curves(models[model].model, models[model].name, x_train, y_train)

                dump(models[model].model, model_path, compress=True)

                # Track file size
                bytes_size = os.path.getsize(model_path)
                size = humanize.naturalsize(bytes_size)

                train_results.loc[model] = [models[model].name, duration, size]
                print(
                    f"({num}/{len(models.keys())}) Finished training {models[model].name} in {duration}"
                )

        # Save train results to file
        if os.path.exists(dev_config.TRAIN_RESULTS_PATH):
            os.remove(dev_config.TRAIN_RESULTS_PATH)
        train_results.to_csv(dev_config.TRAIN_RESULTS_PATH)
