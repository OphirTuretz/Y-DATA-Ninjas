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

<<<<<<< Updated upstream
    Returns:
        xgb.XGBClassifier: Loaded model.
    """
    if not os.path.exists(file_path):
        logging.error(f"Model file not found: {file_path}")
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    logging.info(f"Loading model from: {file_path}")
    model = joblib.load(file_path)
=======
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    if predictions_only:
        pd.DataFrame(predictions).to_csv(path, header=False, index=False)
    else:
        data[PREDICTED_COLUMN] = predictions
        data.to_csv(path, index=False)

    print("Predictions saved.")


def make_predictions(model, X, threshold=None):
    print("Making predictions (probabilities)...")
    probabilities = model.predict_proba(X)[:, 1]
    print("Predictions made.")
    if threshold is not None:
        print(f"Applying threshold: {threshold}")
        return (probabilities > threshold).astype(int)
    return probabilities


def load_model(path: str):
    print(f"Loading model from {path}...")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded.")
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    main()
=======
    print("predict.py started...")

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-data-path", required=True)
    parser.add_argument("-m", "--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("-o", "--predictions-file-name", default=None)
    parser.add_argument("-po", "--predictions-only", default=DEFAULT_PREDICTIONS_ONLY)
    parser.add_argument("-wgid", "--wandb-group-id", default=None)
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Threshold for classifying probability as positive"
    )
    args = parser.parse_args()

    # wandb.init(project=WANDB_PROJECT, group=args.wandb_group_id, job_type="predict")

    df = load_data(args.input_data_path)
    X = extract_features(df)

    model = load_model(args.model_path)

    # TODO: download model from wandb (downloaded_model_path = run.use_model(name="<your-model-name>"))

    predictions = make_predictions(model, X, threshold=(None if args.predictions_only else args.threshold))

    # Resume the run to log the predictions
    api = wandb.Api()
    runs = api.runs(f"Y-DATA-Ninjas/{WANDB_PROJECT}")
    runs_list = list(runs)  # Convert to a list
    last_run = runs_list[-1]
    run = wandb.init(
        entity="Y-DATA-Ninjas", project=WANDB_PROJECT, id=last_run.id, resume="must"
    )
    run.log({"predictions_array": predictions.tolist()})

    if args.predictions_file_name is not None:
        out_path = os.path.join(RESULTS_FOLDER, args.predictions_file_name)
    else:
        input_file_name_without_extension = os.path.splitext(
            os.path.basename(args.input_data_path)
        )[0]

        out_path = os.path.join(
            RESULTS_FOLDER,
            # Commented out because throws an error y.f.
            f"predictions_{input_file_name_without_extension}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )

    save_predictions(
        predictions,
        path=out_path,
        predictions_only=args.predictions_only,
        data=df,
    )

    print("predict.py finished.")
>>>>>>> Stashed changes
