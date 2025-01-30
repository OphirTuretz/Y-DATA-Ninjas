import os

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
    "product_category_1",
    "product_category_2",
    "user_depth",
    "age_level",
    "city_development_index",
    "var_1",
    "gender",
]
COLUMNS_TO_CATEGORIZE = []
TARGET_COLUMN = "is_click"
PREDICTED_COLUMN = TARGET_COLUMN + "_predicted"

DEFAULT_PREDICTIONS_ONLY = False

DATE_TIME_PATTERN = "%Y-%m-%d_%H-%M-%S"

WANDB_PROJECT = "pre-main"

MODEL_GS_PARAM_GRID = {
    "eval_metric": [
        "F1"
    ],  # https://catboost.ai/docs/en/references/custom-metric__supported-metrics
    "iterations": [1000],
    "learning_rate": [0.1],
    "depth": [6],
    "l2_leaf_reg": [3],
    "border_count": [32],
    "thread_count": [4],
    "random_seed": [42],
}
