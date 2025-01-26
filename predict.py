"""
Module to generate predictions using a trained XGBoost model.

Best practices:
1. Load the saved model artifact.
2. Apply the same preprocessing steps (if necessary).
3. Predict on new data and save results.
"""

import logging
import os
import pandas as pd
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

def load_model(file_path: str):
    """
    Load the trained model from a file.

    Args:
        file_path (str): Path to the saved model joblib file.

    Returns:
        xgb.XGBClassifier: Loaded model.
    """
    if not os.path.exists(file_path):
        logging.error(f"Model file not found: {file_path}")
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    logging.info(f"Loading model from: {file_path}")
    model = joblib.load(file_path)
    return model


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the test data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing features for prediction.

    Returns:
        pd.DataFrame: DataFrame with the features.
    """
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    logging.info(f"Loading raw test data from: {file_path}")
    data = pd.read_csv(file_path)
    logging.info(f"Raw test data shape: {data.shape}")
    return data


def generate_predictions(model, data: pd.DataFrame):
    """
    Generate predictions using the trained model.

    Args:
        model: Trained model (XGBClassifier).
        data (pd.DataFrame): DataFrame with feature columns for prediction.

    Returns:
        pd.Series or np.ndarray: Predictions from the model.
    """
    # Ensure the data has the same columns as training, handle missing columns if needed
    # If additional feature transformations are required, do them here
    predictions = model.predict(data)
    return predictions


def main():
    """
    Main execution function for prediction:
    1. Load the trained model.
    2. Load new (raw) test data.
    3. Generate predictions.
    4. Save predictions to CSV.
    """
    model_path = "xgboost_model.joblib"
    data_path = "data/raw_test.csv"
    output_path = "data/predictions.csv"

    model = load_model(model_path)
    test_data = load_data(data_path)
    
    predictions = generate_predictions(model, test_data)

    # Save predictions
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(predictions, columns=["predictions"]).to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
