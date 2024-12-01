# Just to run some tests

from pathlib import Path
from sklearn.linear_model import LinearRegression

import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from libs.training import Trainee
from libs.ttransformers import TLabelEncoder
from libs.ttypes import (
    Metrics,
    ModelInfo,
    RunInfo,
    PreprocessingInfo,
    TrainingInfo,
)


def main():

    df = pd.read_csv("./data/gym_members_exercise_tracking.csv")

    models = [ModelInfo("lr", "Linear Regression", LinearRegression())]

    preprocessing_info = [
        PreprocessingInfo(
            acronym="le",
            name="TLabel encoder",
            pipeline=Pipeline([("tencoder", OrdinalEncoder())]),
            fit=True,
        ),
        PreprocessingInfo(
            acronym="sc",
            name="Standard Scaler",
            pipeline=Pipeline([("scaler", StandardScaler())]),
            fit=True,
        ),
    ]

    Trainee(
        training_info=TrainingInfo(
            output_path=Path.joinpath(Path.cwd(), "tests"),
            metrics=[Metrics.MAE, Metrics.MSE],
        ),
    ).run(
        RunInfo(
            data=df,
            target_column="Calories_Burned",
            models=models,
            preprocessing_info=preprocessing_info,
        )
    )


if __name__ == "__main__":
    main()
