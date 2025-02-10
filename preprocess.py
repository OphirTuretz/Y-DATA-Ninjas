<<<<<<< Updated upstream
"""
Module for data loading and preprocessing.

Best practices:
1. Separation of data loading and cleaning steps into distinct functions.
2. Use of logging for traceability.
3. Clear docstrings for each function.
"""

import os
import logging
=======
# File: preprocess.py
import os
import argparse
>>>>>>> Stashed changes
import pandas as pd
from sklearn.model_selection import train_test_split
<<<<<<< Updated upstream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
=======
from app.const import (
    DEFAULT_CSV_RAW_TRAIN_PATH,
    DEFAULT_RANDOM_STATE,
    DATA_FOLDER,
    DEFAULT_REMOVE_DUPLICATES,
    DEFAULT_REMOVE_MISSING_TARGET,
    DEFAULT_TEST_TRAIN_SPLIT,
    DEFAULT_TEST_SIZE,
    DEFAULT_CSV_TRAIN_PATH,
    DEFAULT_CSV_TEST_PATH,
    DEFAULT_CSV_INFERENCE_PATH,
    TARGET_COLUMN,
    WANDB_PROJECT,
>>>>>>> Stashed changes
)

<<<<<<< Updated upstream
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded pandas DataFrame.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Could not find file: {file_path}")
    
    logging.info(f"Loading dataset from: {file_path}")
    data = pd.read_csv(file_path)
    logging.info(f"Dataset loaded with shape: {data.shape}")
    return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by removing duplicates and handling missing target values.
    Also converts categorical column 'gender' into a binary flag.

    Args:
        data (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    initial_shape = data.shape
    data = data.drop_duplicates()
    logging.info(f"Removed {initial_shape[0] - data.shape[0]} duplicate rows.")

    # Drop rows where the target column 'is_click' is missing
    data = data.dropna(subset=['is_click'])
    
    # Convert 'gender' column to numeric (female=0, male=1)
    if 'gender' in data.columns:
        # Map 'Male' to 1, 'Female' to 0, leave NaN as NaN
        data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})

    # Split DateTime into day and hour
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['day'] = data['DateTime'].dt.day
    data['hour'] = data['DateTime'].dt.hour
    data.drop(columns=['DateTime'], inplace=True)

    if 'product' in data.columns:
        data['product'] = data['product'].astype('category').cat.codes

    data['is_click'] = data['is_click'].astype(int)

    logging.info(f"Data shape after cleaning: {data.shape}")
    return data


def split_data(
    data: pd.DataFrame,
    feature_columns: list,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split the data into training and testing sets.

    Args:
        data (pd.DataFrame): Preprocessed dataset.
        feature_columns (list): Columns to be used as features.
        target_column (str): Column to be used as the target.
        test_size (float): Fraction of data to reserve for test.
        random_state (int): Random seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test): Split datasets.
    """
    X = data[feature_columns]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def main():
    """
    Main execution for preprocessing (standalone usage).
    Loads the CSV, preprocesses, splits, and saves train/test data.
    """
    file_path = "data/train_dataset_full.csv"
    data = load_data(file_path)

    data = preprocess_data(data)

    feature_columns = [
        'session_id', 'user_id', 'product', 'campaign_id',
        'webpage_id', 'product_category_1', 'product_category_2',
        'user_group_id', 'gender', 'age_level', 'user_depth',
        'city_development_index', 'var_1', 'day', 'hour'
    ]
    target_column = "is_click"
    
    
    X_train, X_test, y_train, y_test = split_data(data, feature_columns, target_column)

    # Save the split data
    os.makedirs("data", exist_ok=True)
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    logging.info("Preprocessed and split data saved to 'data/' folder.")
=======
def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print("Data loaded. DataFrame info:")
    print(df.info())
    return df

def save_data(df: pd.DataFrame, path: str):
    print(f"Saving data to {path}...")
    df.to_csv(path, index=False)
    print("Data saved.")

def preprocess_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Preprocess the input DataFrame.

    This function:
      - Removes duplicate rows (training data only)
      - Drops rows with missing target values (if training)
      - Converts DateTime to a datetime object (expected format '%Y-%m-%d %H:%M')
      - Imputes missing campaign IDs using a provided dictionary
      - Imputes age and gender from user_group_id (converting gender to string "Male"/"Female")
      - Ensures metadata consistency per user_id by replacing values with the mode for:
            var_1, gender, age_level, user_depth, city_development_index
      - Drops rows with very little data (non-null count below a threshold)
      - Extracts datetime features and creates a binary version of var_1 (var_1_bin)
      - Computes user behavior features: user_session_count, user_click_count, and user_ctr
      - Drops unnecessary identifier columns
      - One-hot encodes the gender column (dropping the original "gender")
      - Drops the original var_1 column (since var_1_bin is used)
      - Converts selected columns to proper types
    """
    print("### STARTING PREPROCESSING ###")
    print("Initial data shape:", df.shape)
    
    # 1. DATA QUALITY & CLEANING
    # (a) Remove duplicate rows (training data only)
    if is_train:
        before_duplicates = df.shape[0]
        df = df.drop_duplicates()
        after_duplicates = df.shape[0]
        print(f"Removed {before_duplicates - after_duplicates} duplicate rows (training data only).")
    
    # (b) For training data, drop rows with missing target values.
    if is_train and TARGET_COLUMN in df.columns:
        before_missing_target = df.shape[0]
        df = df[~df[TARGET_COLUMN].isna()]
        after_missing_target = df.shape[0]
        print(f"Removed {before_missing_target - after_missing_target} rows with missing target values.")
    
    # (c) Convert DateTime to a proper datetime object.
    # Expected format: "YYYY-MM-DD HH:MM"
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M', errors='coerce')
    if df['DateTime'].isna().any():
        print("Warning: Some DateTime values could not be converted and are now NaT.")
    
    # (d) Impute missing campaign_id values using the provided dictionary.
    web_campaign_dict = {
        1734.0: 82320.0,
        6970.0: 98970.0,
        11085.0: 105960.0,
        13787.0: np.mean([359520.0, 360936.0]),
        28529.0: 118601.0,
        45962.0: 414149.0,
        51181.0: 396664.0,
        53587.0: 404347.0,
        60305.0: 405490.0,
    }
    for web in web_campaign_dict:
        df.loc[
            (df["webpage_id"] == web) & (df["campaign_id"].isna()),
            "campaign_id"
        ] = web_campaign_dict[web]
    
    # (e) Impute age and gender from user_group_id.
    df.loc[df["user_group_id"] == 0, "age_level"] = 0
    for i in range(1, 13):
        gender_val = "Male" if i < 7 else "Female"
        age_val = (i % 7) + (1 if i >= 7 else 0)
        df.loc[
            (df["user_group_id"] == i) & (df["gender"].isna()),
            "gender"
        ] = gender_val
        df.loc[
            (df["user_group_id"] == i) & (df["age_level"].isna()),
            "age_level"
        ] = age_val

    # (f) Impute/Correct metadata columns per user_id.
    meta_cols = ['var_1', 'gender', 'age_level', 'user_depth', 'city_development_index']
    for col in meta_cols:
        # For numeric columns, ensure proper conversion.
        if col in ['var_1', 'age_level', 'user_depth', 'city_development_index']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        mode_map = df.groupby('user_id')[col].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        df[col] = df['user_id'].map(mode_map)
    
    # (g) Drop rows that have very little data.
    df['non_null_count'] = df.notna().sum(axis=1)
    threshold = 10  # Adjust threshold as needed.
    before_drop = df.shape[0]
    df = df[df['non_null_count'] >= threshold]
    after_drop = df.shape[0]
    print(f"Dropped {before_drop - after_drop} rows with less than {threshold} non-null values.")
    
    # 2. FEATURE ENGINEERING
    # (a) Extract datetime features.
    df['hour'] = df['DateTime'].dt.hour.fillna(-1).astype('int8')
    df['day'] = df['DateTime'].dt.day.fillna(-1).astype('int8')
    
    # (b) Binarize var_1.
    df['var_1_bin'] = np.where(df['var_1'] >= 1, 1, 0)
    
    # (c) Create user behavior features.
    user_session_counts = df.groupby('user_id')['session_id'].count().rename('user_session_count')
    df = df.merge(user_session_counts, on='user_id', how='left')
    
    if 'is_click' in df.columns:
        user_click_counts = df.groupby('user_id')['is_click'].sum().rename('user_click_count')
        df = df.merge(user_click_counts, on='user_id', how='left')
    else:
        df['user_click_count'] = 0
    
    df['user_ctr'] = df['user_click_count'] / df['user_session_count']
    
    # 3. FINAL STEP: FEATURE SELECTION & ONE-HOT ENCODING
    # (a) Drop unnecessary identifier columns.
    drop_cols = ['session_id', 'DateTime', 'user_group_id', 'webpage_id']
    df.drop(columns=drop_cols, inplace=True)
    
    # (b) One-hot encode the gender column.
    df = pd.get_dummies(df, columns=["gender"], prefix="gender", drop_first=True)
    
    expected_dummy_cols = ["gender_Male"]
    for col in expected_dummy_cols:
        if col not in df.columns:
            df[col] = 0
    
    # (c) Drop the original var_1 column.
    if 'var_1' in df.columns:
        df.drop(columns=['var_1'], inplace=True)
    
    # (d) Convert selected columns to proper types.
    categorical_cols = ['product', 'campaign_id', 'product_category_1', 'product_category_2']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    df['var_1_bin'] = df['var_1_bin'].astype('int8')
    df['user_session_count'] = df['user_session_count'].astype('int32')
    df['user_click_count'] = df['user_click_count'].astype('int32')
    
    df.drop(columns=['non_null_count'], inplace=True)
    
    print("Final data shape:", df.shape)
    print("### PREPROCESSING COMPLETED ###")
    return df

def preprocess_raw_inference(df: pd.DataFrame):
    print("Preprocessing raw inference data...")
    df_processed = preprocess_data(df, is_train=False)
    save_data(df_processed, DEFAULT_CSV_INFERENCE_PATH)
    print("Raw inference data preprocessed.")

def preprocess_raw_train(df: pd.DataFrame, test_train_split: bool, test_size: float, random_state: int):
    print("Preprocessing raw training data...")
    df_processed = preprocess_data(df, is_train=True)
    if not test_train_split:
        save_data(df_processed, DEFAULT_CSV_TRAIN_PATH)
    else:
        train_df, test_df = train_test_split(df_processed, test_size=test_size, random_state=random_state)
        save_data(train_df, DEFAULT_CSV_TRAIN_PATH)
        save_data(test_df, DEFAULT_CSV_TEST_PATH)
    print("Raw training data preprocessed.")
>>>>>>> Stashed changes

if __name__ == "__main__":
<<<<<<< Updated upstream
    main()
=======
    print("preprocess.py started...")
    parser = argparse.ArgumentParser()
    parser.add_argument("-rs", "--random-state", default=DEFAULT_RANDOM_STATE, type=int)
    parser.add_argument("-crp", "--csv-raw-path", default=DEFAULT_CSV_RAW_TRAIN_PATH)
    parser.add_argument("-infr", "--inference-run", default=False, type=str2bool)
    parser.add_argument("-tts", "--test-train-split", default=DEFAULT_TEST_TRAIN_SPLIT, type=str2bool)
    parser.add_argument("-ts", "--test-size", default=DEFAULT_TEST_SIZE, type=float)
    parser.add_argument("-wgid", "--wandb-group-id", default=None)
    args = parser.parse_args()
    
    raw_df = load_data(args.csv_raw_path)
    if args.inference_run:
        preprocess_raw_inference(raw_df)
    else:
        preprocess_raw_train(raw_df, args.test_train_split, args.test_size, args.random_state)
    
    print("preprocess.py finished.")
>>>>>>> Stashed changes
