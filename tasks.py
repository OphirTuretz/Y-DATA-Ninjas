import itertools
from invoke import task
from datetime import datetime
import os
import wandb
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
)


def prepare_data_for_pipeline():
    print("Preparing data for pipeline...")
    # put the data in central location


@task
def pipeline(c):

    print("Running pipeline...")

    # Loading .env environment variables
    load_dotenv()

    # Archive the last experiment
    c.run("inv archive-experiment")

    # prepare_data_for_pipeline()

    # Get all experiment's hyperparameters (the parameters and their corresponding value lists)
    keys = MODEL_GS_PARAM_GRID.keys()
    values = MODEL_GS_PARAM_GRID.values()

    # Generate all combinations
    combinations = itertools.product(*values)

    print("Running experiments...")

    # Iterate over combinations and create a list/tuple of keys and current values
    for idx, combination in enumerate(combinations):

        # Create a group id that will be shared within the same run
        wandb_group_id = "experiment-" + wandb.util.generate_id()

        print(f"Running experiment {idx}: {wandb_group_id}...")

        # Get the current combination of parameters
        param_combination = dict(zip(keys, combination))
        print(f"Experiment parameters: {param_combination}")

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
            c.run(f"python preprocess.py --wandb-group-id {wandb_group_id}")

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

        # preprocess inference data
        c.run(
            f"python preprocess.py --wandb-group-id {wandb_group_id} --inference-run True --csv-raw-path {DEFAULT_CSV_RAW_INFERENCE_PATH}"
        )

        # predict on inference
        c.run(
            f"python predict.py --wandb-group-id {wandb_group_id} --input-data-path {DEFAULT_CSV_INFERENCE_PATH} --predictions-only True --predictions-file-name {CSV_PREDICTIONS_INFERENCE_FILENAME}"
        )

        print("Experiment finished.")

        c.run(f"inv archive-experiment --name {wandb_group_id} --leave-data True")

    print("All experiments finished.")

    print("Pipeline finished.")


@task(optional=["name", "base_folder", "leave_data"])
def archive_experiment(
    c, name=UKNOWN_EXPERIMENT_NAME, base_folder=ARCHIVE_FOLDER, leave_data=False
):

    name = f"{datetime.now().strftime(DATE_TIME_PATTERN)}_{name}"

    print(f"archiving experiment {name}...")

    # check if base folder exists (archive folder)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # create experiment folder
    exp_path = f"{base_folder}/{name}"
    c.run(f"mkdir {exp_path}")

    # check if data folder exists
    if os.path.exists("data"):
        c.run(f"mv data {exp_path}")
        c.run("mkdir data")

        if leave_data:
            c.run(f"cp {exp_path}/data/* data/")
        else:
            c.run(f"cp {exp_path}/data/{CSV_RAW_TRAIN_FILENAME} data/")
            c.run(f"cp {exp_path}/data/{CSV_RAW_INFERENCE_FILENAME} data/")

    # check if results folder exists
    if os.path.exists("results"):
        c.run(f"mv results {exp_path}")
        c.run("mkdir results")

    # check if models folder exists
    if os.path.exists("models"):
        c.run(f"mv models {exp_path}")
        c.run("mkdir models")

    print("experiment archived.")
