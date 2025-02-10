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


<<<<<<< Updated upstream
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
=======
@task(optional=["name", "base_folder", "leave_data"])
def archive_experiment(c, name=UKNOWN_EXPERIMENT_NAME, base_folder=ARCHIVE_FOLDER, leave_data=False):
    from datetime import datetime
    import os
    import shutil
    # Create experiment folder with timestamp
    name = f"{datetime.now().strftime(DATE_TIME_PATTERN)}_{name}"
    print(f"Archiving experiment {name}...")
    os.makedirs(base_folder, exist_ok=True)
    exp_path = os.path.join(base_folder, name)
    os.makedirs(exp_path, exist_ok=True)
    
    # Archive data folder
    if os.path.exists("data"):
        shutil.move("data", exp_path)
        os.makedirs("data", exist_ok=True)
        if leave_data:
            for file in os.listdir(os.path.join(exp_path, "data")):
                shutil.copy(os.path.join(exp_path, "data", file), "data")
        else:
            for file in [CSV_RAW_TRAIN_FILENAME, CSV_RAW_INFERENCE_FILENAME]:
                src_file = os.path.join(exp_path, "data", file)
                if os.path.exists(src_file):
                    shutil.copy(src_file, "data")
    
    # Archive results folder
    if os.path.exists("results"):
        shutil.move("results", exp_path)
        os.makedirs("results", exist_ok=True)
    
    # Do not move models folder - keep it intact
    if os.path.exists("models"):
        print("Skipping archiving of the 'models' folder.")
    
    print("Experiment archived.")

>>>>>>> Stashed changes
