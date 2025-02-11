"""
preprocess.py

This module loads raw CSV data, cleans and engineers features, and then
saves the preprocessed data (either splitting train/test or for inference).

Key changes:
  - Advanced feature engineering including:
      • Imputation of campaign_id using a dictionary.
      • Imputation of age and gender from user_group_id.
      • Replacement of metadata columns with their mode per user_id.
      • Extraction of DateTime features (hour, day).
      • Binarization of var_1 into var_1_bin.
      • Computation of user behavior features:
           - Training: computes user_session_count, user_click_count, and user_ctr;
             saves per-user stats and computes new-user averages (from users with one session).
           - Inference: merges saved stats for known users; for known users, adds 1 to the session count and recalculates user_ctr;
             for new users, fills missing values using the new-user averages.
  - Inference mode now skips dropping rows based on non-null count (so no rows are lost).
  - Finally, drops unnecessary columns and one-hot encodes the gender column.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from app.const import (
    DEFAULT_CSV_RAW_TRAIN_PATH,
    DEFAULT_RANDOM_STATE,
    DATA_FOLDER,
    DEFAULT_TEST_TRAIN_SPLIT,
    DEFAULT_TEST_SIZE,
    DEFAULT_CSV_TRAIN_PATH,
    DEFAULT_CSV_TEST_PATH,
    DEFAULT_CSV_INFERENCE_PATH,
    TARGET_COLUMN,
    WANDB_PROJECT,
)
from app.utils import str2bool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_data(path: str) -> pd.DataFrame:
    logging.info(f"Loading data from {path}...")
    df = pd.read_csv(path)
    logging.info("Data loaded. DataFrame info:")
    logging.info(df.info())
    return df


def save_data(df: pd.DataFrame, path: str):
    logging.info(f"Saving data to {path}...")
    df.to_csv(path, index=False)
    logging.info("Data saved.")


def preprocess_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Preprocess the input DataFrame.

    Steps performed:
      1. DATA QUALITY & CLEANING:
         - Remove duplicate rows (if training).
         - Drop rows with missing target values (if training).
         - Convert DateTime to datetime (format '%Y-%m-%d %H:%M').
         - Impute missing campaign_id values using a provided dictionary.
         - Impute age and gender from user_group_id.
         - Replace metadata columns with their mode per user_id.
         - (If training) Drop rows with too few non-null values.
         - (If inference) Skip dropping rows.
      2. FEATURE ENGINEERING:
         - Extract hour and day from DateTime.
         - Binarize var_1 into var_1_bin.
         - Create user behavior features:
             • Training: compute user_session_count, user_click_count, and user_ctr;
               save per-user stats and compute new-user averages (from users with one session).
             • Inference: merge saved stats for known users; for known users, add 1 to session count and recalc user_ctr;
               for new users, fill missing values using the new-user averages.
      3. FINAL STEP:
         - Drop unnecessary identifier columns.
         - One-hot encode the gender column.
         - Drop the original var_1 column.
         - Convert selected columns to proper types.
    """
    logging.info("### STARTING PREPROCESSING ###")
    logging.info(f"Initial data shape: {df.shape}")

    # 1. DATA QUALITY & CLEANING

    # (a) Remove duplicate rows (training data only)
    if is_train:
        before_duplicates = df.shape[0]
        df = df.drop_duplicates()
        after_duplicates = df.shape[0]
        logging.info(f"Removed {before_duplicates - after_duplicates} duplicate rows (training data only).")

    # (b) For training data, drop rows with missing target values.
    if is_train and TARGET_COLUMN in df.columns:
        before_missing_target = df.shape[0]
        df = df.dropna(subset=[TARGET_COLUMN])
        after_missing_target = df.shape[0]
        logging.info(f"Removed {before_missing_target - after_missing_target} rows with missing target values.")

    # (c) Convert DateTime to a proper datetime object.
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M', errors='coerce')
    if df['DateTime'].isna().any():
        logging.warning("Some DateTime values could not be converted and are now NaT.")

    # (d) Impute missing campaign_id values using a provided dictionary.
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

    # (f) Replace metadata columns with the mode per user_id.
    meta_cols = ['var_1', 'gender', 'age_level', 'user_depth', 'city_development_index']
    for col in meta_cols:
        if col in ['var_1', 'age_level', 'user_depth', 'city_development_index']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        mode_map = df.groupby('user_id')[col].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        df[col] = df['user_id'].map(mode_map)

    # (g) Drop rows that have very little data (training only)
    if is_train:
        df['non_null_count'] = df.notna().sum(axis=1)
        threshold = 10  # Adjust threshold as needed.
        before_drop = df.shape[0]
        df = df[df['non_null_count'] >= threshold]
        after_drop = df.shape[0]
        logging.info(f"Dropped {before_drop - after_drop} rows with less than {threshold} non-null values.")
    else:
        logging.info("Inference mode: skipping dropping rows based on non-null count.")

    # 2. FEATURE ENGINEERING

    # (a) Extract datetime features.
    df['hour'] = df['DateTime'].dt.hour.fillna(-1).astype('int8')
    df['day'] = df['DateTime'].dt.day.fillna(-1).astype('int8')

    # (b) Binarize var_1.
    if 'var_1' in df.columns:
        df['var_1_bin'] = np.where(df['var_1'] >= 1, 1, 0)
    else:
        logging.warning("Column 'var_1' not found; creating 'var_1_bin' with default 0.")
        df['var_1_bin'] = 0

    # (c) Create user behavior features.
    if TARGET_COLUMN in df.columns:
        # Training mode: compute behavior features.
        user_session_counts = df.groupby('user_id')['session_id'].count().rename('user_session_count')
        df = df.merge(user_session_counts, on='user_id', how='left')
        user_click_counts = df.groupby('user_id')[TARGET_COLUMN].sum().rename('user_click_count')
        df = df.merge(user_click_counts, on='user_id', how='left')
        df['user_ctr'] = df['user_click_count'] / df['user_session_count']

        # Save per-user statistics for inference.
        mapping_df = df[['user_id', 'user_session_count', 'user_click_count', 'user_ctr']].drop_duplicates()
        mapping_path = os.path.join(DATA_FOLDER, 'user_behavior_stats.pkl')
        mapping_df.to_pickle(mapping_path)

        # Compute new-user averages from training (users with only one session).
        new_users = df[df['user_session_count'] == 1]
        if len(new_users) > 0:
            new_user_stats = {
                'user_session_count': 1,  # new users have one session
                'user_click_count': new_users['user_click_count'].mean(),
                'user_ctr': new_users['user_ctr'].mean()
            }
        else:
            new_user_stats = {'user_session_count': 1, 'user_click_count': 0, 'user_ctr': 0.0}
        global_stats_path = os.path.join(DATA_FOLDER, 'global_user_stats.pkl')
        with open(global_stats_path, 'wb') as f:
            pickle.dump(new_user_stats, f)
    else:
        # Inference mode: use precomputed statistics.
        mapping_path = os.path.join(DATA_FOLDER, 'user_behavior_stats.pkl')
        global_stats_path = os.path.join(DATA_FOLDER, 'global_user_stats.pkl')
        if os.path.exists(mapping_path) and os.path.exists(global_stats_path):
            mapping_df = pd.read_pickle(mapping_path)
            with open(global_stats_path, 'rb') as f:
                new_user_stats = pickle.load(f)
        else:
            mapping_df = None
            new_user_stats = {'user_session_count': 1, 'user_click_count': 0, 'user_ctr': 0.0}
        if mapping_df is not None:
            # Merge saved stats for known users (left join preserves all rows).
            df = df.merge(mapping_df, on='user_id', how='left')
            # For known users, add 1 to session count (to account for the current session)
            known_users_mask = df['user_session_count'].notna()
            df.loc[known_users_mask, 'user_session_count'] = df.loc[known_users_mask, 'user_session_count'] + 1
            # Recalculate user_ctr for known users.
            df.loc[known_users_mask, 'user_ctr'] = df.loc[known_users_mask, 'user_click_count'] / df.loc[known_users_mask, 'user_session_count']
        # For new users, fill missing values using the new-user averages.
        df['user_session_count'] = df['user_session_count'].fillna(1).astype('int32')
        df['user_click_count'] = df['user_click_count'].fillna(new_user_stats['user_click_count']).astype('int32')
        df['user_ctr'] = df['user_ctr'].fillna(new_user_stats['user_ctr'])

    # 3. FINAL STEP: FEATURE SELECTION & ONE-HOT ENCODING

    # (a) Drop unnecessary identifier columns.
    drop_cols = ['session_id', 'DateTime', 'user_group_id', 'webpage_id']
    # Note: We no longer drop 'non_null_count' in inference because we didn't add it.
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

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

    # Convert engineered numeric columns.
    df['var_1_bin'] = df['var_1_bin'].astype('int8')
    df['user_session_count'] = df['user_session_count'].astype('int32')
    df['user_click_count'] = df['user_click_count'].astype('int32')
    df['user_ctr'] = df['user_ctr'].astype('float64')

    logging.info(f"Final data shape: {df.shape}")
    logging.info("### PREPROCESSING COMPLETED ###")
    return df


def preprocess_raw_inference(df: pd.DataFrame):
    logging.info("Preprocessing raw inference data...")
    df_processed = preprocess_data(df, is_train=False)
    save_data(df_processed, DEFAULT_CSV_INFERENCE_PATH)
    logging.info("Raw inference data preprocessed.")


def preprocess_raw_train(df: pd.DataFrame, test_train_split: bool, test_size: float, random_state: int):
    logging.info("Preprocessing raw training data...")
    df_processed = preprocess_data(df, is_train=True)
    if not test_train_split:
        save_data(df_processed, DEFAULT_CSV_TRAIN_PATH)
    else:
        train_df, test_df = train_test_split(df_processed, test_size=test_size, random_state=random_state)
        save_data(train_df, DEFAULT_CSV_TRAIN_PATH)
        save_data(test_df, DEFAULT_CSV_TEST_PATH)
    logging.info("Raw training data preprocessed.")


if __name__ == "__main__":
    logging.info("preprocess.py started...")
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
    
    logging.info("preprocess.py finished.")
