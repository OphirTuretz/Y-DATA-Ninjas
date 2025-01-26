"""
Module for evaluating a trained model on a given dataset and logging results to W&B.

Best practices:
1. Consistent evaluation approach (accuracy, classification report).
2. Logging for transparency.
3. Additional metrics or plots can be added here.
"""

import logging
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

def load_model(file_path: str):
    """
    Load the trained model from a joblib file.

    Args:
        file_path (str): Path to the saved model.

    Returns:
        xgb.XGBClassifier: The trained model.
    """
    logging.info(f"Loading model from: {file_path}")
    return joblib.load(file_path)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the test data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame of features or target, depending on usage.
    """
    logging.info(f"Loading data from: {file_path}")
    return pd.read_csv(file_path)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """
    Evaluate the model using accuracy score and classification report.
    Log results to Weights & Biases.

    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for test data.
    """
    logging.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred))

    wandb.log({"accuracy": accuracy})
    # Log additional metrics if desired
    wandb.log({
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_1": report["1"]["f1-score"]
    })


def main():
    """
    Main function to evaluate the model and log the metrics to W&B.
    """
    wandb.init(project="new_project", notes="Evaluating model", tags=["evaluation"])

    model = load_model("xgboost_model.joblib")
    X_test = load_data("data/X_test.csv")
    y_test = load_data("data/y_test.csv")

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    evaluate_model(model, X_test, y_test)
    
    wandb.finish()


if __name__ == "__main__":
    main()
