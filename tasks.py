from invoke import task
import datetime
import os
import wandb
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
)


def prepare_data_for_pipeline():
    print("Preparing data for pipeline...")
    # put the data in central location


@task
def pipeline(c):

    # Create a group id that will be shared within the same run
    wandb_group_id = "experiment-" + wandb.util.generate_id()

    # prepare_data_for_pipeline()

    # preprocess raw train data
    c.run(f"python preprocess.py --wandb-group-id {wandb_group_id}")

    # train
    c.run(f"python train.py --wandb-group-id {wandb_group_id}")

    # predict on train
    c.run(
        f"python predict.py --wandb-group-id {wandb_group_id} --input-data-path {DEFAULT_CSV_TRAIN_PATH} --predictions-file-name {CSV_PREDICTIONS_TRAIN_FILENAME}"
    )

    # process train results
    c.run(
        f"python results.py --wandb-group-id {wandb_group_id} --input-data-path {DEFAULT_CSV_PREDICTIONS_TRAIN_PATH}"
    )

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


@task
def archive_experiment(c, name, base_folder=ARCHIVE_FOLDER):
    name = f"{datetime.now().strftime(DATE_TIME_PATTERN)}_{name}"
    print(f"archived experiment: {name}")
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    exp_path = f"{base_folder}/{name}"
    c.run(f"mkdir {exp_path}")

    # check if data folder exists
    if os.path.exists("data"):
        c.run(f"mv data {exp_path}")
        c.run("mkdir data")
    # check if results folder exists
    if os.path.exists("results"):
        c.run(f"mv results {exp_path}")
        c.run("mkdir results")
    # check if models folder exists
    if os.path.exists("models"):
        c.run(f"mv models {exp_path}")
        c.run("mkdir models")
