import os

DATA_FOLDER = "data"

CSV_RAW_TRAIN_FILENAME = "train_dataset_full.csv"
DEFAULT_CSV_RAW_TRAIN_PATH = os.path.join(DATA_FOLDER, CSV_RAW_TRAIN_FILENAME)

CSV_TRAIN_FILENAME = "train.csv"
DEFAULT_CSV_TRAIN_PATH = os.path.join(DATA_FOLDER, CSV_TRAIN_FILENAME)
CSV_TEST_FILENAME = "test.csv"
DEFAULT_CSV_TEST_PATH = os.path.join(DATA_FOLDER, CSV_TEST_FILENAME)

DEFAULT_REMOVE_DUPLICATES = True

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
TARGET_COLUMN = "is_click"
