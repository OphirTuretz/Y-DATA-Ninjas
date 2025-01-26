"""
Module for training an XGBoost model on preprocessed data.

Best practices:
1. Clear separation of data loading, training, and evaluation steps.
2. Logging for training progress and errors.
3. Weights & Biases integration for experiment tracking.
"""

import logging
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

def load_data():
    """
    Load the training and testing data from CSV files.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """
    logging.info("Loading train/test split data.")
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv")
    y_test = pd.read_csv("data/y_test.csv")

    # Convert target from DataFrame to Series if needed
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    logging.info(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, **kwargs) -> xgb.XGBClassifier:
    """
    Train an XGBoost model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        **kwargs: Additional hyperparameters for XGBoost.

    Returns:
        xgb.XGBClassifier: Trained XGBoost model.
    """
    logging.info("Initializing XGBoost model...")
    model = xgb.XGBClassifier(**kwargs)
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and log results to Weights & Biases.

    Args:
        model (xgb.XGBClassifier): Trained XGBoost model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
    """
    logging.info("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred))

    # Log metrics to wandb
    wandb.log({
        "accuracy": accuracy,
        # Convert the classification report (dict) into separate logs if desired
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1-score": report["1"]["f1-score"]
    })


def save_model(model, file_path: str):
    """
    Save the trained model to a file.

    Args:
        model (xgb.XGBClassifier): Trained model.
        file_path (str): Output file path.
    """
    logging.info(f"Saving model to: {file_path}")
    joblib.dump(model, file_path)
    logging.info("Model saved.")


def main():
    """
    Main execution function for training.
    1. Initializes wandb for experiment tracking.
    2. Loads data.
    3. Trains the model.
    4. Evaluates the model and logs metrics.
    5. Saves the trained model artifact.
    """
    wandb.init(project="new_project", notes="Training XGBoost model", tags=["training"])

    X_train, X_test, y_train, y_test = load_data()

    n_neg = sum(y_train == 0)
    n_pos = sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos


    # You can insert hyperparameters here (e.g., n_estimators=100, learning_rate=0.1)
    model = train_model(
        X_train, y_train, 
        n_estimators=100, 
        learning_rate=0.1, 
        enable_categorical=True, 
        scale_pos_weight=scale_pos_weight, 
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8
    )

    evaluate_model(model, X_test, y_test)
    save_model(model, "xgboost_model.joblib")

    wandb.finish()


if __name__ == "__main__":
    main()
