import os
import wandb
import pickle
import argparse
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.dummy import DummyClassifier
from catboost import CatBoostClassifier
from app.const import (
    DEFAULT_CSV_TRAIN_PATH,
    FEATURES_LIST,
    TARGET_COLUMN,
    MODELS_FOLDER,
    DEFAULT_MODEL_PATH,
    WANDB_PROJECT,
    CATEGORICAL_FEATURES_CATBOOST,
    FEATURES_TYPE_MAP,
)


def save_model(model):
    print(f"Saving model...")

    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    with open(DEFAULT_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Model saved.")


def fit_model(model, X, y, categorical_features):
    print("Fitting model...")
    model.fit(X, y, cat_features=categorical_features)
    print("Model fitted.")
    return model


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    print("Preparing data...")
    print("Extracting features...")
    X = df[FEATURES_LIST]
    # cat_features must be integer or string, real number values and NaN values should be converted to string.
    X[CATEGORICAL_FEATURES_CATBOOST] = X[CATEGORICAL_FEATURES_CATBOOST].astype(str)
    print("Features extracted.")
    print("Extracting target...")
    y = df[TARGET_COLUMN]
    print("Target extracted.")
    print("Data prepared.")
    return X, y


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, dtype=FEATURES_TYPE_MAP)
    print(df.info())
    print("Data loaded.")
    return df


if __name__ == "__main__":
    print("train.py started...")

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ctp", "--csv-train-path", default=DEFAULT_CSV_TRAIN_PATH, type=str
    )
    parser.add_argument("-wgid", "--wandb-group-id", default=None)
    parser.add_argument(
        "-m", "--model", default="CatBoostClassifier", help="Model to use"
    )
    parser.add_argument(
        "-mn", "--model-name", default="CatBoost", help="Custom model name"
    )

    # Hyperparameters
    parser.add_argument(
        "-nl",
        "--num-leaves",
        default=None,
        type=int,
        help="Number of leaves in the tree",
    )
    parser.add_argument(
        "-i", "--iterations", default=100, type=int, help="Number of iterations"
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=0.2, type=float, help="Learning rate"
    )
    parser.add_argument("-em", "--eval-metric", default="F1", help="Evaluation metric")
    parser.add_argument("-d", "--depth", default=6, type=int, help="Depth of the tree")
    parser.add_argument(
        "-l2", "--l2-leaf-reg", default=3, type=float, help="L2 regularization"
    )
    parser.add_argument("-nm", "--nan-mode", default="Min", help="NaN handling mode")
    parser.add_argument(
        "-esr",
        "--early-stopping-rounds",
        default=10,
        type=int,
        help="Early stopping rounds",
    )
    parser.add_argument(
        "-l", "--loss-function", default="Logloss", help="Loss function"
    )
    parser.add_argument("-cw", "--class-weights", default='Balanced', help="Class weights")

    args = parser.parse_args()

    # Load and prepare data
    train_df = load_data(args.csv_train_path)
    X_train, y_train = prepare_data(train_df)

    # Check if class weights are provided
    if args.class_weights is not None:
        if args.class_weights != "Balanced":
            args.class_weights = None
        else:
            # Compute class weights
            pos_ratio = y_train.sum() / len(y_train)
            neg_ratio = 1 - pos_ratio
            args.class_weights = [1, neg_ratio / pos_ratio]

    ModelClass = CatBoostClassifier

    hyperparams = {
        "num_leaves": args.num_leaves,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "eval_metric": args.eval_metric,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "nan_mode": args.nan_mode,
        "early_stopping_rounds": args.early_stopping_rounds,
        "loss_function": args.loss_function,
        "class_weights": args.class_weights,
    }

    model = ModelClass(**hyperparams)

    # Initialize W&B
    run = wandb.init(
        project=WANDB_PROJECT,
        group=args.wandb_group_id,
        job_type="train_predict",
        config=None,
    )
    run.log({"model": args.model_name})
    run.log(hyperparams)

    # Train model
    model = fit_model(model, X_train, y_train, CATEGORICAL_FEATURES_CATBOOST)

    # Save model
    save_model(model)

    print("train.py finished.")