"""
Module for data loading and preprocessing.

Best practices:
1. Separation of data loading and cleaning steps into distinct functions.
2. Use of logging for traceability.
3. Clear docstrings for each function.
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

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


if __name__ == "__main__":
    main()
