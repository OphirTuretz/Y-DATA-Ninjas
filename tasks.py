"""
Module to orchestrate the entire workflow via Invoke tasks.

Best practices:
1. Provide tasks for each major pipeline step (preprocess, train, predict).
2. Provide a 'run_all' task to do everything sequentially.
3. Provide an 'archive' or 'cleanup' task for experiment management.
"""

import os
from datetime import datetime
from invoke import task

DATE_TIME_PATTERN = "%Y%m%d_%H%M%S"
ARCHIVED_EXPERIMENTS_DIR = "archived_experiments"

@task
def preprocess(c):
    """
    Run the preprocessing script to create train/test splits.
    """
    c.run("python preprocess.py")


@task
def train(c):
    """
    Run the training script to train the XGBoost model and log to wandb.
    """
    c.run("python train.py")


@task
def predict(c):
    """
    Run the prediction script on raw_test data using the saved XGBoost model.
    """
    c.run("python predict.py")


@task
def evaluate(c):
    """
    Evaluate the saved model on the test set and log metrics to wandb.
    """
    c.run("python results.py")


@task
def run_all(c):
    """
    Run the entire workflow in sequence:
    1. Preprocess (split data).
    2. Train (train and save model).
    3. Predict (generate predictions on raw_test).
    4. Evaluate (optional step to evaluate on X_test, y_test).
    """
    preprocess(c)
    train(c)
    # predict(c)
    evaluate(c)


@task
def archive(c, name, base_folder=ARCHIVED_EXPERIMENTS_DIR):
    """
    Archive the experiment results by copying
    data, model artifacts, and wandb logs into a timestamped folder.

    Args:
        c: Invoke context.
        name (str): A short descriptive name for the experiment.
        base_folder (str): Folder to store archived experiments.
    """
    timestamp = datetime.now().strftime(DATE_TIME_PATTERN)
    archive_name = f"{timestamp}_{name}"
    archive_path = os.path.join(base_folder, archive_name)
    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    c.run(f"mkdir -p {archive_path}")

    # Copy data, model, wandb logs, etc.
    c.run(f"cp -r data {archive_path} || true")
    c.run(f"cp xgboost_model.joblib {archive_path} || true")
    c.run(f"cp -r wandb {archive_path} || true")

    print(f"Archived experiment to: {archive_path}")
