import os
import sys
import logging
import humanize
import pandas as pd
import pytz

from joblib import dump
from pathlib import Path
from datetime import datetime
from typing import Literal, Union

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline

from .ttypes import (
    ModelInfo,
    PreprocessingInfo,
    RunInfo,
    Scores,
    StatisticsInfo,
    TrainingInfo,
    TrainingResults,
)


class Trainee:
    """Trainee class to train multiple models with multiple preprocessing steps."""

    MODELS_OUTPUT_DIR = "models"
    DATA_OUTPUT_DIR = "data"

    def __init__(
        self,
        training_info: TrainingInfo,
        statistics_info: Union[StatisticsInfo, None] = None,
        directory_timestamp_format: str = "%Y-%m-%d_%H-%M",
        datetime_timezone: str = "Europe/Madrid",
        logs_filename: str = "training",
        logger: Union[logging.Logger, None] = None,
        log_formatter: Union[logging.Formatter, None] = None,
    ) -> None:
        self.training_info = training_info
        self.statistics_info = statistics_info if statistics_info else StatisticsInfo()
        self.directory_timestamp_format = directory_timestamp_format
        self.timezone = pytz.timezone(datetime_timezone)
        self.log_filename = logs_filename
        self.logger = logger or logging.getLogger(__name__)
        self.log_formatter = (
            log_formatter
            if log_formatter
            else logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt=self.statistics_info.timestamp_format,
            )
        )

    def _now(self) -> datetime:
        return datetime.now(tz=self.timezone)

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
                self.log_filename,
            )
        )
        console_handler = logging.StreamHandler(sys.stdout)

        file_handler.setFormatter(self.log_formatter)
        console_handler.setFormatter(self.log_formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _setup_run(
        self,
        verbosity: Literal["quiet", "normal", "verbose"],
    ) -> Path:
        """Initalizes the needed directories and logger for the training run.

        Args:
            verbosity (Literal[&quot;quiet&quot;, &quot;normal&quot;, &quot;verbose&quot;]): logger verbosity level.

        Returns:
            Path: the run path to store the training run.
        """

        run_id = self._now().strftime(self.directory_timestamp_format)

        # Append the run id to the output models path
        run_path = Path.joinpath(self.training_info.output_path, run_id)
        run_path.mkdir(parents=True, exist_ok=True)

        for path in [self.DATA_OUTPUT_DIR, self.MODELS_OUTPUT_DIR]:
            Path.joinpath(run_path, path).mkdir(parents=True, exist_ok=True)

        self._init_logger(verbosity, run_path)
        return run_path

    def _preprocess(
        self,
        run_path: Path,
        X: pd.DataFrame,
        y: pd.Series,
        step_idx: int,
        preprocessing: PreprocessingInfo,
    ) -> pd.DataFrame:
        """Preprocess the given data with the given preprocessing step, store a checkpoint and return the preprocessed data.

        Args:
            run_path (Path): current run base output path.
            X (pd.DataFrame): data to preprocess.
            y (pd.Series): target data to preprocess.
            step_idx (int): current preprocessing step index
            preprocessing (PreprocessingInfo): current preprocessing to apply to the data.

        Returns:
            pd.DataFrame: preprocessed data
        """

        self.logger.info(
            "Starting preprocessing step [%s] %s ...", step_idx, preprocessing.name
        )

        self.logger.info("X shape: %s, Y shape: %s", X.shape, y.shape)

        # Preprocess data
        preprocessing_t0 = self._now()

        try:
            df = pd.DataFrame(
                preprocessing.on(X, y, fit=preprocessing.fit), columns=X.columns
            )
        except Exception as e:
            self.logger.error(
                "Error in preprocessing step [%s] %s: %s",
                step_idx,
                preprocessing.name,
                e,
            )
            return None

        preprocessing_t1 = self._now()

        elapsed = preprocessing_t1 - preprocessing_t0
        self.logger.info(
            "Preprocessing step [%s] %s finished in %ss",
            step_idx,
            preprocessing.name,
            elapsed.seconds,
        )

        # Save checkpoint
        df.to_csv(
            Path.joinpath(
                run_path, self.DATA_OUTPUT_DIR, f"{step_idx}-{preprocessing.name}.csv"
            ),
        )
        return df

    def _train_test(
        self,
        preprocessing_path: Path,
        model_idx: int,
        model_data: ModelInfo,
        data: pd.DataFrame,
        cv: int,
    ) -> TrainingResults:
        """Train the given model with the given data and store the results.

        Args:
            run_path (Path): current run base output with preprocessing step dir.
            model_idx (int): current model index.
            model_data (ModelInfo): model information to train.
            X_train (pd.DataFrame): data to train the model with.
            y_train (pd.Series): target data to train the model with.
        Returns:
            TrainingResults: training results for the model
        """

        self.logger.info(
            "Starting training step [%s] %s ...", model_idx, model_data.name
        )

        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=[model_data.target_column]),
            data[model_data.target_column],
            test_size=self.training_info.train_test_split,
            random_state=self.training_info.random_state,
        )

        training_t0 = self._now()

        try:
            trained_model = cross_validate(model_data.model, X_train, y_train, cv=cv)
        except Exception as e:
            self.logger.error("Error training model %s: %s", model_data.name, e)
            return

        training_t1 = self._now()
        elapsed = training_t1 - training_t0

        self.logger.info(
            "Training step [%s] %s finished in %ss",
            model_idx,
            model_data.name,
            elapsed.seconds,
        )

        model_path = Path.joinpath(
            preprocessing_path,
            self.MODELS_OUTPUT_DIR,
            f"{model_data.name}.bin",
        )

        dump(trained_model, model_path, compress=True)

        # Track file size
        bytes_size = os.path.getsize(model_path)

        return TrainingResults(
            model=model_data,
            trained_size=bytes_size,
        )

    def _test(
        self,
        model_idx: int,
        model_data: ModelInfo,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Scores:
        """Test the given model with the given data and store the results.

        Args:
            model_data (ModelInfo): model information to test.
            X_train (pd.DataFrame): data the model trained with.
            y_train (pd.Series): target data the model trained with.
            X_test (pd.DataFrame): data to test the model with.
            y_test (pd.Series): target data to test the model with.

        Returns:
            Scores: train and test score metrics for the model
        """
        self.logger.info(
            "Starting testing step for model [%s] %s ...", model_idx, model_data.name
        )

        try:
            test_score = model_data.model.score(X_test, y_test)
        except Exception as e:
            self.logger.error("Error testing model %s: %s", model_data.name, e)
            return

        return test_score

    def run(
        self,
        run_data: RunInfo,
        verbosity: Literal["quiet", "normal", "verbose"] = "verbose",
    ) -> Path:
        """
        Train each of the given models with the given data and different preprocessing steps.

        Args:
            run_data (RunInfo): What to do this run, including the data, models, and preprocessing steps.
            verbosity (Literal[&quot;quiet&quot;, &quot;normal&quot;, &quot;verbose&quot;], optional): _description_. Defaults to "verbose".

        Returns:
            Path: output path with the results of the training run.
        """

        run_path = self._setup_run(verbosity)

        self.logger.info("Starting training run at %s ...", run_path)

        # TODO: Define needed columns for the results
        results = pd.DataFrame(columns=["Model Name", "Training Time", "File Size"])

        X = run_data.data.drop(columns=[run_data.target_column])
        y = run_data.data[run_data.target_column]

        for preprocessing_idx, preprocessing in enumerate(
            run_data.preprocessing_info,
            start=1,
        ):

            preprocessed_df = self._preprocess(
                run_path,
                X,
                y,
                preprocessing_idx,
                preprocessing,
            )

            # Check if there was an error in the preprocessing step
            if preprocessed_df is None or preprocessed_df.empty:
                continue

            # for model_idx, model_data in enumerate(run_data.models, start=1):

            #     train_results = self._train_test(
            #         run_path,
            #         model_idx,
            #         model_data,
            #         X_train,
            #         y_train,
            #         preprocessed_df,
            #     )

            #     if not train_results:
            #         continue

            #     train_results.preprocessing = preprocessing
            #     train_results.test_scores = self._test()

            #     size = humanize.naturalsize(train_results.trained_size)

        #         train_results.loc[model] = [models[model].name, duration, size]
        #         print(
        #             f"({num}/{len(models.keys())}) Finished training {models[model].name} in {duration}"
        #         )

        # # Save train results to file
        # if os.path.exists(dev_config.TRAIN_RESULTS_PATH):
        #     os.remove(dev_config.TRAIN_RESULTS_PATH)
        # train_results.to_csv(dev_config.TRAIN_RESULTS_PATH)
