from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline


class Metrics(Enum):
    """Registered loss functions for training the models."""

    MSE = "mean_squared_error"
    RMSE = "root_mean_squared_error"
    MAE = "mean_absolute_error"
    HINGE = "hinge"
    CROSS_ENTROPY = "cross_entropy"
    R2 = "r2_score"


@dataclass
class Scores:
    """Scores for a single model"""

    train_scores: dict[Metrics, float]
    test_scores: dict[Metrics, float]


@dataclass(frozen=True)
class ModelInfo:
    """Model specific information."""

    acronym: str
    """Model acronym to display"""

    name: str
    """Model full-name to display"""

    model: Union[BaseEstimator, RegressorMixin, ClassifierMixin]
    """Model object"""


@dataclass(frozen=True)
class TrainingInfo:
    output_path: Path
    """Path to store the resulting data"""

    metrics: List[Metrics]
    """Metrics to use for training and testing"""

    train_test_split: float = 0.2
    """Percentage of data to use for testing"""

    random_state: int = 2024
    """Random state to use for reproducibility"""

    cv = 5
    """Number of cross-validation folds to use for training"""


@dataclass(frozen=True)
class StatisticsInfo:
    timestamp_format: str = "%H:%M:%S"


@dataclass(frozen=True)
class PreprocessingInfo:
    """Dataframe preprocessing information for training the models."""

    acronym: str
    """Acronym of the preprocessing step to display, should be unique and friendly with your OS file system"""

    name: str
    """Name of the preprocessing step to display, should be unique and friendly with your OS file system"""

    pipeline: Pipeline
    """Preprocessing pipeline to apply to the data"""

    fit: bool = False
    """Whether to fit the preprocessing step to the data"""

    def on(self, X: pd.DataFrame, y: pd.Series, fit: bool = False):
        """Apply the preprocessing step to the data"""
        return (
            self.pipeline.fit_transform(X, y) if fit else self.pipeline.transform(X, y)
        )


@dataclass(frozen=True)
class RunInfo:
    """Current run information"""

    data: pd.DataFrame
    """Raw data to use for preprocessing and training models"""

    target_column: str
    """Column to use as the target for training the models"""

    models: List[ModelInfo]
    """Model information to train with"""

    preprocessing_info: List[PreprocessingInfo]
    """Preprocessing data to apply to the data before training the models"""


@dataclass
class TrainingResults:
    """Training results for a single model"""

    model: ModelInfo

    trained_size: int
    """Byte size of the trained model"""

    preprocessing: PreprocessingInfo
    """Preprocessing information used for training"""

    scores: Scores
    """Scores for the model"""
