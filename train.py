import argparse
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.dummy import DummyClassifier
from app.const import (
    DEFAULT_CSV_TRAIN_PATH,
    DEFAULT_CSV_TEST_PATH,
    FEATURES_LIST,
)


def save_data(df: pd.DataFrame, path: str):
    print(f"Saving data to {path}...")
    df.to_csv(path, index=False)
    print("Data saved.")


def fit_model(model, X_train, y_train):
    print("Fitting model...")
    model.fit(X_train, y_train)
    print("Model fitted.")
    return model


def extract_features_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    print("Extracting features...")
    features = df[FEATURES_LIST]
    print("Features extracted.")
    print("Extracting target...")
    target = df["is_click"]
    print("Target extracted.")
    return features.to_numpy(), target.to_numpy()


def prepare_train_test_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("Preparing training data...")
    X_train, y_train = extract_features_target(train_df)
    print("Data prepared.")
    print("Preparing test data...")
    X_test, y_test = extract_features_target(test_df)
    print("Data prepared.")
    return X_train, X_test, y_train, y_test


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(df.info())
    print("Data loaded.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-ctp", "--csv-train-path", default=DEFAULT_CSV_TRAIN_PATH, type=str
    )
    parser.add_argument(
        "-ctep", "--csv-test-path", default=DEFAULT_CSV_TEST_PATH, type=str
    )

    args = parser.parse_args()

    train_df = load_data(args.csv_train_path)
    test_df = load_data(args.csv_test_path)

    X_train, X_test, y_train, y_test = prepare_train_test_data(train_df, test_df)

    model = DummyClassifier()
    model = fit_model(model, X_train, y_train)

    # TODO: Evaluate the model on the training and test set
    # TODO: Save the model

    print("Training done.")
