import os
import argparse
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from app.const import (
    DATA_FOLDER,
    DEFAULT_CSV_RAW_TRAIN_PATH,
    DEFAULT_REMOVE_DUPLICATES,
    DEFAULT_TEST_TRAIN_SPLIT,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_CSV_TRAIN_PATH,
    DEFAULT_CSV_TEST_PATH,
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


def preprocess(
    df: pd.DataFrame,
    remove_duplicates: bool,
    test_train_split: bool,
    test_size: float,
    random_state: int,
):
    print("Preprocessing data...")

    if remove_duplicates:
        df = drop_duplicates(df)
        save_data(df, os.path.join(DATA_FOLDER, "raw_train_no_duplicates.csv"))

    if not test_train_split:
        save_data(df, DEFAULT_CSV_TRAIN_PATH)
    else:
        train_df, test_df = split_data(df, test_size, random_state)
        save_data(train_df, DEFAULT_CSV_TRAIN_PATH)
        save_data(test_df, DEFAULT_CSV_TEST_PATH)

    # TODO: Imputation? (based only on the training set and applied also to the test set)
    # TODO: Remove rows with missing target values?
    # TODO: Other preprocessing steps?

    print("Data preprocessed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-crtp", "--csv-raw-train-path", default=DEFAULT_CSV_RAW_TRAIN_PATH
    )
    parser.add_argument("-rd", "--remove-duplicates", default=DEFAULT_REMOVE_DUPLICATES)
    parser.add_argument("-tts", "--test-train-split", default=DEFAULT_TEST_TRAIN_SPLIT)
    parser.add_argument("-ts", "--test-size", default=DEFAULT_TEST_SIZE)
    parser.add_argument("-rs", "--random-state", default=DEFAULT_RANDOM_STATE)

    args = parser.parse_args()

    raw_train_df = load_data(args.csv_raw_train_path)

    preprocess(
        raw_train_df,
        args.remove_duplicates,
        args.test_train_split,
        args.test_size,
        args.random_state,
    )

    print("Preprocessing done.")
