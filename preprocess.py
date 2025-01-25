import os
import wandb
import argparse
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from app.const import (
    DATA_FOLDER,
    DEFAULT_CSV_RAW_TRAIN_PATH,
    DEFAULT_REMOVE_DUPLICATES,
    DEFAULT_REMOVE_MISSING_TARGET,
    DEFAULT_TEST_TRAIN_SPLIT,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_CSV_TRAIN_PATH,
    DEFAULT_CSV_TEST_PATH,
    TARGET_COLUMN,
    DEFAULT_CSV_INFERENCE_PATH,
    WANDB_PROJECT,
)


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(df.info())
    print("Data loaded.")
    return df


def save_data(df: pd.DataFrame, path: str):
    print(f"Saving data to {path}...")
    df.to_csv(path, index=False)
    print("Data saved.")


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    print("Removing rows with missing target values...")
    num_rows_before = len(df.index)
    df = df.dropna(subset=[TARGET_COLUMN], axis=0, ignore_index=True)
    num_rows_after = len(df.index)
    print(
        f"Rows with missing target values removed (Number of rows before: {num_rows_before}, after: {num_rows_after}, removed: {num_rows_before - num_rows_after})."
    )
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("Removing duplicates...")
    num_rows_before = len(df.index)
    df = df.drop_duplicates(ignore_index=True)
    num_rows_after = len(df.index)
    print(
        f"Duplicates removed (Number of rows before: {num_rows_before}, after: {num_rows_after}, removed: {num_rows_before - num_rows_after})."
    )
    return df


def split_data(
    df: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Splitting the dataset into training and test sets...")
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    print("Splitting done.")
    return train_df, test_df


def preprocess_raw_inference(df: pd.DataFrame):
    print("Preprocessing raw inference data...")
    # TODO: Imputation? (based on the training set)
    # TODO: Encoding of categorical variables? (based on the training set)
    # TODO: Feature engineering? (based on the training set)
    # TODO: Other preprocessing steps? (based on the training set)
    save_data(df, DEFAULT_CSV_INFERENCE_PATH)
    print("Raw inference data preprocessed.")


def preprocess_raw_train(
    df: pd.DataFrame,
    remove_duplicates: bool,
    remove_missing_target: bool,
    test_train_split: bool,
    test_size: float,
    random_state: int,
):
    print("Preprocessing raw training data...")

    if remove_duplicates:
        df = drop_duplicates(df)
        save_data(df, os.path.join(DATA_FOLDER, "raw_train_no_duplicates.csv"))

    if remove_missing_target:
        df = drop_missing_target(df)
        save_data(df, os.path.join(DATA_FOLDER, "raw_train_no_missing_target.csv"))

    if not test_train_split:
        save_data(df, DEFAULT_CSV_TRAIN_PATH)
    else:
        train_df, test_df = split_data(df, test_size, random_state)
        save_data(train_df, DEFAULT_CSV_TRAIN_PATH)
        save_data(test_df, DEFAULT_CSV_TEST_PATH)

    # TODO: Delete rows with missing values?
    # TODO: Imputation? (based only on the training set and applied also to the test set)
    # TODO: Encoding of categorical variables?
    # TODO: Feature engineering? (from within the data, e.g., date-time features, or from outside the data, e.g., external data)
    # TODO: Other preprocessing steps? (delete outliers?)
    # TODO: Save the preprocessing parameters (e.g., imputation values, encoding values, etc.) for later use
    # TODO: Save the preprocessing steps in the W&B run
    # TODO: Save the preprocessed data in the W&B run
    # TODO: Save the preprocessed data in the data folder

    print("Raw training data preprocessed.")


if __name__ == "__main__":

    print("preprocess.py started...")

    parser = argparse.ArgumentParser()

    parser.add_argument("-crp", "--csv-raw-path", default=DEFAULT_CSV_RAW_TRAIN_PATH)

    parser.add_argument("-infr", "--inference-run", default=False)

    parser.add_argument("-rd", "--remove-duplicates", default=DEFAULT_REMOVE_DUPLICATES)
    parser.add_argument(
        "-rmt", "--remove-missing-target", default=DEFAULT_REMOVE_MISSING_TARGET
    )

    parser.add_argument("-tts", "--test-train-split", default=DEFAULT_TEST_TRAIN_SPLIT)
    parser.add_argument("-ts", "--test-size", default=DEFAULT_TEST_SIZE)
    parser.add_argument("-rs", "--random-state", default=DEFAULT_RANDOM_STATE)

    parser.add_argument("-wgid", "--wandb-group-id", default=None)

    args = parser.parse_args()

    # wandb.init(project=WANDB_PROJECT, group=args.wandb_group_id, job_type="preprocess")

    raw_df = load_data(args.csv_raw_path)

    if args.inference_run:
        preprocess_raw_inference(raw_df)
    else:
        preprocess_raw_train(
            raw_df,
            args.remove_duplicates,
            args.remove_missing_target,
            args.test_train_split,
            args.test_size,
            args.random_state,
        )

    print("preprocess.py finished.")
