import os
import numpy as np

DATA_FOLDER = "data"
MODELS_FOLDER = "models"
RESULTS_FOLDER = "results"
ARCHIVE_FOLDER = "archived_experiments"

UKNOWN_EXPERIMENT_NAME = "unknown_experiment"

CSV_RAW_TRAIN_FILENAME = "train_dataset_full.csv"
DEFAULT_CSV_RAW_TRAIN_PATH = os.path.join(DATA_FOLDER, CSV_RAW_TRAIN_FILENAME)

CSV_RAW_INFERENCE_FILENAME = "X_test_1st.csv"
DEFAULT_CSV_RAW_INFERENCE_PATH = os.path.join(DATA_FOLDER, CSV_RAW_INFERENCE_FILENAME)

CSV_TRAIN_FILENAME = "train.csv"
DEFAULT_CSV_TRAIN_PATH = os.path.join(DATA_FOLDER, CSV_TRAIN_FILENAME)
CSV_TEST_FILENAME = "test.csv"
DEFAULT_CSV_TEST_PATH = os.path.join(DATA_FOLDER, CSV_TEST_FILENAME)
CSV_INFERENCE_FILENAME = "inference.csv"
DEFAULT_CSV_INFERENCE_PATH = os.path.join(DATA_FOLDER, CSV_INFERENCE_FILENAME)

MODEL_FILENAME = "model.pkl"
DEFAULT_MODEL_PATH = os.path.join(MODELS_FOLDER, MODEL_FILENAME)

CSV_PREDICTIONS_TRAIN_FILENAME = "predictions_train.csv"
DEFAULT_CSV_PREDICTIONS_TRAIN_PATH = os.path.join(
    RESULTS_FOLDER, CSV_PREDICTIONS_TRAIN_FILENAME
)
CSV_PREDICTIONS_TEST_FILENAME = "predictions_test.csv"
DEFAULT_CSV_PREDICTIONS_TEST_PATH = os.path.join(
    RESULTS_FOLDER, CSV_PREDICTIONS_TEST_FILENAME
)
CSV_PREDICTIONS_INFERENCE_FILENAME = "predictions_inference.csv"
DEFAULT_CSV_PREDICTIONS_INFERENCE_PATH = os.path.join(
    RESULTS_FOLDER, CSV_PREDICTIONS_INFERENCE_FILENAME
)

DEFAULT_REMOVE_DUPLICATES = True
DEFAULT_REMOVE_MISSING_TARGET = True
DEFAULT_MAKE_INFERENCE = False
DEFAULT_TEST_TRAIN_SPLIT = True
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

FEATURES_LIST = [
    "product",
    "campaign_id",
    "product_category_1",
    "product_category_2",
    "gender_Male",           # one-hot encoded gender
    "age_level",
    "user_depth",
    "city_development_index",
    "var_1_bin",             # binarized version of var_1
    "hour",
    "day",
    "user_session_count",    # new feature
    "user_click_count",      # new feature
    "user_ctr"               # new feature
]

FEATURES_TYPE_MAP = {
    "product": "str",
    "campaign_id": "float64",
    "product_category_1": "float64",
    "product_category_2": "float64",
    "gender_Male": "int8",
    "age_level": "float64",
    "user_depth": "float64",
    "city_development_index": "float64",
    "var_1_bin": "int8",
    "hour": "int8",
    "day": "int8",
    "user_session_count": "int32",
    "user_click_count": "int32",
    "user_ctr": "float64"
}

CATEGORICAL_FEATURES_CATBOOST = [
    "product",
    "campaign_id",
    "product_category_1",
    "product_category_2",
    # "gender",
    # "var_1",
]
TARGET_COLUMN = "is_click"
PREDICTED_COLUMN = TARGET_COLUMN + "_predicted"

DEFAULT_PREDICTIONS_ONLY = False

DATE_TIME_PATTERN = "%Y-%m-%d_%H-%M-%S"

WANDB_PROJECT = "asaf-run"

# https://catboost.ai/docs/en/references/custom-metric__supported-metrics
# https://catboost.ai/docs/en/references/training-parameters/common
MODEL_GS_PARAM_GRID = {
    "num_leaves": [31],
    "iterations": [500],
    "learning_rate": [0.01],
    "eval_metric": ["F1"],
    "depth": [11],  # np.arange(4, 11, 2),
    "l2_leaf_reg": [3],
    "nan_mode": ["Min"],
    "early_stopping_rounds": [10],
    "loss_function": ["Logloss"],
    "class_weights": ["Balanced"],  # ["Default", "Balanced"],
}

REVENUE_COST_DICT  = {'revenue': 2, 'cost': 0.02}

STREAMLIT_OUTPUT_FILE_NAME_PREFIX = "predictions"
STREAMLIT_INPUT_FILE_NAME_PREFIX = "raw_inference"
STREAMLIT_PREPROCESSED_FILE_NAME_PREFIX = "inference"