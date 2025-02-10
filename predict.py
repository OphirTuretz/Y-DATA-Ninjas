import os
import numpy as np
import argparse
import wandb
import pickle
import pandas as pd
from app.const import (
    WANDB_PROJECT,
    DEFAULT_MODEL_PATH,
    FEATURES_LIST,
    DEFAULT_PREDICTIONS_ONLY,
    RESULTS_FOLDER,
    PREDICTED_COLUMN,
    TARGET_COLUMN,
    WANDB_PROJECT,
    FEATURES_TYPE_MAP,
    CATEGORICAL_FEATURES_CATBOOST,
)


def save_predictions(
    predictions, path: str, predictions_only: bool = False, data: pd.DataFrame = None
):
    print(f"Saving predictions to {path}...")

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    if predictions_only:
        pd.DataFrame(predictions).to_csv(path, header=False, index=False)
    else:
        data[PREDICTED_COLUMN] = predictions
        data.to_csv(path, index=False)

    print("Predictions saved.")


def make_predictions(model, X: np.ndarray, predict_class: bool = False) -> np.ndarray:
    if not predict_class:
        print("Making predictions in probability form...")
        predictions = model.predict_proba(X)[:, 1]
    else:
        print("Making predictions in binary form...")
        predictions = model.predict(X)
    print("Predictions made.")
    return predictions


def load_model(path: str):
    print(f"Loading model from {path}...")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded.")
    return model


def extract_features(df: pd.DataFrame) -> np.ndarray:
    print("Extracting features...")
    X = df[FEATURES_LIST]
    # Should be done during preprocessing
    # cat_features must be integer or string, real number values and NaN values should be converted to string.
    X[CATEGORICAL_FEATURES_CATBOOST] = X[CATEGORICAL_FEATURES_CATBOOST].astype(str)
    print("Features extracted.")
    return X


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, dtype=FEATURES_TYPE_MAP)
    print(df.info())
    print("Data loaded.")
    return df


if __name__ == "__main__":
    print("predict.py started...")

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-data-path", required=True)
    parser.add_argument("-m", "--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("-o", "--predictions-file-name", default=None)
    parser.add_argument("-po", "--predictions-only", default=DEFAULT_PREDICTIONS_ONLY)
    parser.add_argument("-wgid", "--wandb-group-id", default=None)
    parser.add_argument('-pc', '--predict-class', default=False, action='store_true')
    parser.add_argument("-iw", "--ignore-wandb", action="store_true")

    args = parser.parse_args()

    # wandb.init(project=WANDB_PROJECT, group=args.wandb_group_id, job_type="predict")

    df = load_data(args.input_data_path)
    X = extract_features(df)

    model = load_model(args.model_path)

    # TODO: download model from wandb (downloaded_model_path = run.use_model(name="<your-model-name>"))

    predictions = make_predictions(model, X, args.predict_class)

    # Resume the run to log the predictions
    if not args.ignore_wandb:
        api = wandb.Api()
        runs = api.runs(f"Y-DATA-Ninjas/{WANDB_PROJECT}")
        runs_list = list(runs)  # Convert to a list
        last_run = runs_list[-1]
        run = wandb.init(
            entity="Y-DATA-Ninjas", project=WANDB_PROJECT, id=last_run.id, resume="must")

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
