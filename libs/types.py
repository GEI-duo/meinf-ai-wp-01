from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class LossFunction(Enum):
    """Registered loss functions for training the models."""

    MSE = "mean_squared_error"
    MAE = "mean_absolute_error"
    HINGE = "hinge"
    CROSS_ENTROPY = "cross_entropy"


@dataclass(frozen=True)
class ModelInfo:
    """Model specific information."""

    acronym: str
    """Model acronym to display"""

    name: str
    """Model full-name to display"""

    model: BaseEstimator
    """Model object"""


@dataclass(frozen=True)
class TrainingInfo:
    output_path: Path
    """Path to store the resulting data"""

    loss_function: LossFunction
    """Loss function to use for training"""

    train_test_split: float = 0.2
    """Percentage of data to use for testing"""

    random_state: int = 2024
    """Random state to use for reproducibility"""


@dataclass(frozen=True)
class StatisticsInfo:
    timestamp_format: str = "%H:%M:%S"


@dataclass(frozen=True)
class PreprocessingInfo:
    """Dataframe preprocessing information for training the models."""

    name: str
    """Name of the preprocessing step to display"""

    description: Union[str | None]
    """Full description of the preprocessing step"""

    pipeline: Pipeline
    """Preprocessing pipeline to apply to the data"""


@dataclass(frozen=True)
class RunInfo:
    """Current run information"""

    data: pd.DataFrame
    """Raw data to use for preprocessing and training models"""

    models: List[ModelInfo]
    """Model information to train with"""

    preprocessing_info: List[PreprocessingInfo]
    """Preprocessing data to apply to the data before training the models"""

    training_info: TrainingInfo
    """Training general information"""
