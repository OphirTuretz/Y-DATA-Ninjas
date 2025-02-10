import os

DATA_FOLDER = "data"
CSV_RAW_FILENAME = "train_dataset_full.csv"

<<<<<<< Updated upstream
DEFAULT_CSV_RAW_PATH = os.path.join(DATA_FOLDER, CSV_RAW_FILENAME)
=======
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
<<<<<<< Updated upstream
    "gender_Male",            # one-hot encoded gender dummy
    "age_level",
    "user_depth",
    "city_development_index",
    "var_1_bin",
    "hour",
    "day",
    "user_session_count",
    "user_click_count",
    "user_ctr"
]

FEATURES_TYPE_MAP = {
    "product": "str",                      # or "category" if you prefer
=======
    "gender_Male",           # one-hot encoded gender
    "age_level",
    "user_depth",
    "city_development_index",
    "var_1_bin",             # binarized version of var_1
    "hour",
    "day",
    "user_session_count",    # new user behavior feature
    "user_click_count",      # new user behavior feature
    "user_ctr"               # new user behavior feature
]

FEATURES_TYPE_MAP = {
    "product": "str",                      # or "category" if preferred
>>>>>>> Stashed changes
    "campaign_id": "float64",
    "product_category_1": "float64",
    "product_category_2": "float64",
    "gender_Male": "int8",                  # one-hot encoded binary column (0 or 1)
    "age_level": "float64",
    "user_depth": "float64",
    "city_development_index": "float64",
    "var_1_bin": "int8",                    # binary version of var_1
    "hour": "int8",
    "day": "int8",
    "user_session_count": "int32",          # new feature: count of sessions per user
    "user_click_count": "int32",            # new feature: count of clicks per user
    "user_ctr": "float64"                   # new feature: click-through rate per user
}

<<<<<<< Updated upstream
# For CatBoost, you typically specify the names of the categorical features.
# Since you have one-hot encoded 'gender', and replaced 'var_1' with 'var_1_bin' (which is numeric),
# the remaining categorical features may just be:
=======
>>>>>>> Stashed changes
CATEGORICAL_FEATURES_CATBOOST = [
    "product",
    "campaign_id",
    "product_category_1",
    "product_category_2"
]

TARGET_COLUMN = "is_click"
PREDICTED_COLUMN = TARGET_COLUMN + "_predicted"

DEFAULT_PREDICTIONS_ONLY = False

DATE_TIME_PATTERN = "%Y-%m-%d_%H-%M-%S"

<<<<<<< Updated upstream
WANDB_PROJECT = "asaf_run"
=======
WANDB_PROJECT = "asaf-run"
>>>>>>> Stashed changes

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
>>>>>>> Stashed changes
