import os
import wandb
import pickle
import argparse
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.dummy import DummyClassifier
from app.const import (
    DEFAULT_CSV_TRAIN_PATH,
    FEATURES_LIST,
    TARGET_COLUMN,
    MODELS_FOLDER,
    DEFAULT_MODEL_PATH,
    WANDB_PROJECT,
)


def save_model(model):
    print(f"Saving model...")

    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    with open(DEFAULT_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Model saved.")


def fit_model(model, X, y):
    print("Fitting model...")
    model.fit(X, y)
    print("Model fitted.")
    return model


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    print("Preparing data...")
    print("Extracting features...")
    X = df[FEATURES_LIST].to_numpy()
    print("Features extracted.")
    print("Extracting target...")
    y = df[TARGET_COLUMN].to_numpy()
    print("Target extracted.")
    print("Data prepared.")
    return X, y


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(df.info())
    print("Data loaded.")
    return df


if __name__ == "__main__":
    print("train.py started...")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-ctp", "--csv-train-path", default=DEFAULT_CSV_TRAIN_PATH, type=str
    )

    parser.add_argument("-wgid", "--wandb-group-id", default=None)

    args = parser.parse_args()

    # wandb.init(project=WANDB_PROJECT, group=args.wandb_group_id, job_type="train")

    train_df = load_data(args.csv_train_path)
    X_train, y_train = prepare_data(train_df)

    model = DummyClassifier()
    model = fit_model(model, X_train, y_train)

    save_model(model)

    # TODO: Log the model to W&B (run.log_model(path="<path-to-model>", name="<name>"))
    # TODO: Evaluate the model on the training and test set

    print("train.py finished.")
