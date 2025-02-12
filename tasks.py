import itertools
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, auc
from invoke import task
from datetime import datetime
import os
import wandb
import shutil
from dotenv import load_dotenv
from app.const import (
    ARCHIVE_FOLDER,
    DATE_TIME_PATTERN,
    DEFAULT_CSV_INFERENCE_PATH,
    DEFAULT_CSV_RAW_INFERENCE_PATH,
    DEFAULT_CSV_TRAIN_PATH,
    DEFAULT_CSV_TEST_PATH,
    CSV_PREDICTIONS_TRAIN_FILENAME,
    CSV_PREDICTIONS_TEST_FILENAME,
    DEFAULT_CSV_PREDICTIONS_TRAIN_PATH,
    DEFAULT_CSV_PREDICTIONS_TEST_PATH,
    CSV_PREDICTIONS_INFERENCE_FILENAME,
    UKNOWN_EXPERIMENT_NAME,
    CSV_RAW_TRAIN_FILENAME,
    CSV_RAW_INFERENCE_FILENAME,
    MODEL_GS_PARAM_GRID,
    DEFAULT_ARCHIVE_EXPERIMENT,
    DEFAULT_PREDICT_INFERENCE,
    RANDOM_STATE_LIST,
    FINAL_MODEL_PARAM,
    COMPUTE_METRICS_DEFAULT_THR,
)
from app.utils import str2bool


@task(optional=["archive_experiment", "predict_inference"])
def pipeline(
    c,
    archive_experiment=DEFAULT_ARCHIVE_EXPERIMENT,
    predict_inference=DEFAULT_PREDICT_INFERENCE,
):

    print("Running pipeline...")

    # Validate flags input
    archive_experiment = str2bool(archive_experiment)
    predict_inference = str2bool(predict_inference)

    # Loading .env environment variables
    load_dotenv()

    # Archive/delete last experiment
    if archive_experiment:
        # Archive the last experiment
        c.run("inv archive-experiment")
    else:
        # Delete the last experiment
        c.run("inv clear-experiment")

    # Get all experiment's hyperparameters (the parameters and their corresponding value lists)
    keys = MODEL_GS_PARAM_GRID.keys()
    values = MODEL_GS_PARAM_GRID.values()

    # Generate all combinations
    combinations = list(itertools.product(*values))

    print("Running experiments...")

    # Iterate over all random states
    for random_state in RANDOM_STATE_LIST:

        # Iterate over combinations and create a list/tuple of keys and current values
        for idx, combination in enumerate(combinations):

            # Create a group id that will be shared within the same run
            wandb_group_id = "experiment-" + wandb.util.generate_id()

            print(f"Running experiment {idx}: {wandb_group_id}...")

            # Get the current combination of parameters
            param_combination = dict(zip(keys, combination))
            print(
                f"Experiment parameters: 'random_state': {random_state}, {param_combination}"
            )

            # Create a formatted string with the parameters
            formatted_str = " ".join(
                [
                    f"--{key.replace('_', '-')} {value}"
                    for key, value in param_combination.items()
                ]
            )

            # If this is the first experiment, preprocess the raw data
            if idx == 0:
                # preprocess raw train data
                c.run(
                    f"python preprocess.py --wandb-group-id {wandb_group_id} --random-state {random_state}"
                )

            # train
            c.run(f"python train.py --wandb-group-id {wandb_group_id} {formatted_str}")

            # # predict on train
            # c.run(
            #     f"python predict.py --wandb-group-id {wandb_group_id} --input-data-path {DEFAULT_CSV_TRAIN_PATH} --predictions-file-name {CSV_PREDICTIONS_TRAIN_FILENAME}"
            # )

            # # process train results
            # c.run(
            #     f"python results.py --wandb-group-id {wandb_group_id} --input-data-path {DEFAULT_CSV_PREDICTIONS_TRAIN_PATH}"
            # )

            # predict on test
            c.run(
                f"python predict.py --wandb-group-id {wandb_group_id} --input-data-path {DEFAULT_CSV_TEST_PATH} --predictions-file-name {CSV_PREDICTIONS_TEST_FILENAME}"
            )

            # process test results
            c.run(
                f"python results.py --wandb-group-id {wandb_group_id} --input-data-path {DEFAULT_CSV_PREDICTIONS_TEST_PATH}"
            )

            # Check if to predict on inference as well
            if predict_inference:
                # preprocess inference data
                c.run(
                    f"python preprocess.py --wandb-group-id {wandb_group_id} --inference-run True --csv-raw-path {DEFAULT_CSV_RAW_INFERENCE_PATH}"
                )

                # predict on inference
                c.run(
                    f"python predict.py --wandb-group-id {wandb_group_id} --input-data-path {DEFAULT_CSV_INFERENCE_PATH} --predictions-only True --predictions-file-name {CSV_PREDICTIONS_INFERENCE_FILENAME}"
                )

            print("Experiment finished.")

            if archive_experiment:
                c.run(
                    f"inv archive-experiment --name {wandb_group_id} --leave-data True"
                )
            else:
                c.run(f"inv clear-experiment --name {wandb_group_id} --leave-data True")

    print("All experiments finished.")

    print("Pipeline finished.")


@task(optional=["archive_experiment", "predict_inference"])
def train_all_train(
    c,
    archive_experiment=True,
    predict_inference=True,
):
    print("Running train on all raw train...")

    # Validate flags input
    archive_experiment = str2bool(archive_experiment)
    predict_inference = str2bool(predict_inference)

    # Loading .env environment variables
    load_dotenv()

    # Archive/delete last experiment
    if archive_experiment:
        # Archive the last experiment
        c.run("inv archive-experiment")
    else:
        # Delete the last experiment
        c.run("inv clear-experiment")

    # Create a group id that will be shared within the same run
    wandb_group_id = "experiment-" + wandb.util.generate_id()

    print(f"Running experiment: {wandb_group_id}...")

    param_combination = FINAL_MODEL_PARAM

    print(f"Experiment parameters: {param_combination}")

    # Create a formatted string with the parameters
    formatted_str = " ".join(
        [
            f"--{key.replace('_', '-')} {value}"
            for key, value in param_combination.items()
        ]
    )

    # preprocess raw train data without train-test split
    c.run(
        f"python preprocess.py --wandb-group-id {wandb_group_id} --test-train-split False"
    )

    # train
    c.run(f"python train.py --wandb-group-id {wandb_group_id} {formatted_str}")

    # Check if to predict on inference as well
    if predict_inference:
        # preprocess inference data
        c.run(
            f"python preprocess.py --wandb-group-id {wandb_group_id} --inference-run True --csv-raw-path {DEFAULT_CSV_RAW_INFERENCE_PATH}"
        )

        # predict on inference
        c.run(
            f"python predict.py --wandb-group-id {wandb_group_id} --input-data-path {DEFAULT_CSV_INFERENCE_PATH} --predictions-only True --predictions-file-name {CSV_PREDICTIONS_INFERENCE_FILENAME}"
        )

    if archive_experiment:
        c.run(f"inv archive-experiment --name {wandb_group_id}")

    print("Train on all raw train finished.")


@task
def predict_inference(c):

    print("Running predict inference...")

    # Validate there is a model file in the models folder
    if not os.path.isdir("models"):
        raise ValueError("Models folder does not exist.")
    else:
        model_path = os.path.join("models", "model.pkl")
        if not os.path.isfile(model_path):
            raise ValueError("Model file does not exist.")

    predictions_file_name = (
        f"predictions_{datetime.now().strftime(DATE_TIME_PATTERN)}.csv"
    )

    # preprocess inference data
    c.run(
        f"python preprocess.py --inference-run True --csv-raw-path {DEFAULT_CSV_RAW_INFERENCE_PATH}"
    )

    # predict on inference
    c.run(
        f"python predict.py --ignore-wandb --input-data-path {DEFAULT_CSV_INFERENCE_PATH} --predictions-only True --predictions-file-name {predictions_file_name}"
    )

    print("Predict inference finished.")


@task(optional=["thr"])
def compute_metrics(
    c, y_true_file_path, y_pred_file_path, thr=COMPUTE_METRICS_DEFAULT_THR
):

    y_true = pd.read_csv(y_true_file_path, names=["is_click"])
    y_pred = pd.read_csv(y_pred_file_path, names=["is_click_predicted"])

    print(f"f1: {f1_score(y_true.is_click, y_pred.is_click_predicted > thr)}")

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    print(f"PRAUC: {pr_auc}")


@task(optional=["name", "base_folder", "leave_data"])
def archive_experiment(
    c, name=UKNOWN_EXPERIMENT_NAME, base_folder=ARCHIVE_FOLDER, leave_data=False
):

    name = f"{datetime.now().strftime(DATE_TIME_PATTERN)}_{name}"

    print(f"archiving experiment {name}...")

    # check if base folder exists (archive folder)
    # if not os.path.exists(base_folder):
    #     os.makedirs(base_folder)
    # Ensure base folder exists
    os.makedirs(base_folder, exist_ok=True)

    # Create experiment folder
    # exp_path = f"{base_folder}/{name}"
    # c.run(f"mkdir {exp_path}")
    exp_path = os.path.join(base_folder, name)
    os.makedirs(exp_path, exist_ok=True)

    # Initialize "data" folder
    if os.path.exists("data"):
        # c.run(f"mv data {exp_path}")
        # c.run("mkdir data")
        shutil.move("data", exp_path)
        os.makedirs("data", exist_ok=True)

        if leave_data:
            # c.run(f"cp {exp_path}/data/* data/")
            for file in os.listdir(os.path.join(exp_path, "data")):
                shutil.copy(os.path.join(exp_path, "data", file), "data")
        else:
            # c.run(f"cp {exp_path}/data/{CSV_RAW_TRAIN_FILENAME} data/")
            # c.run(f"cp {exp_path}/data/{CSV_RAW_INFERENCE_FILENAME} data/")
            for file in [CSV_RAW_TRAIN_FILENAME, CSV_RAW_INFERENCE_FILENAME]:
                src_file = os.path.join(exp_path, "data", file)
                if os.path.exists(src_file):
                    shutil.copy(src_file, "data")

    # Initialize results folder
    if os.path.exists("results"):
        # c.run(f"mv results {exp_path}")
        # c.run("mkdir results")
        shutil.move("results", exp_path)
        os.makedirs("results", exist_ok=True)

    # Initialize models folder
    if os.path.exists("models"):
        # c.run(f"mv models {exp_path}")
        # c.run("mkdir models")
        shutil.move("models", exp_path)
        os.makedirs("models", exist_ok=True)

    print("experiment archived.")


@task(optional=["name", "leave_data"])
def clear_experiment(
    c,
    name=UKNOWN_EXPERIMENT_NAME,
    leave_data=False,
):
    print(f"clearing experiment {name}...")

    # Handle data folder
    if os.path.exists("data"):
        if not leave_data:
            for filename in os.listdir("data"):
                if filename not in [CSV_RAW_TRAIN_FILENAME, CSV_RAW_INFERENCE_FILENAME]:
                    file_path = os.path.join("data", filename)
                    os.unlink(file_path)

    # Initialize results folder
    if os.path.exists("results"):
        for filename in os.listdir("results"):
            file_path = os.path.join("results", filename)
            os.unlink(file_path)

    # Initialize models folder
    if os.path.exists("models"):
        for filename in os.listdir("models"):
            file_path = os.path.join("models", filename)
            os.unlink(file_path)

    print("experiment cleared.")
