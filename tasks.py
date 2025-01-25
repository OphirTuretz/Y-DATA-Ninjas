from invoke import task
import os
from datetime import datetime
import json

# Column categories
cols_for_one_hot_encoding = ['hour', 'day_of_week', 'campaign_id', 'webpage_id',
                             'product_category_1', 'product_category_2',
                             'product', 'user_group_id']
binary_cols = ['var_1', 'gender',  'current_is_more_than_once',
               ]  # 'is_more_than_once', 'current_session_freq_above_6', 'current_session_freq_below_6',
               # 'current_same_campaign_freq_above_6', 'current_same_campaign_freq_below_6',
               # 'current_same_product_freq_above_6', 'current_same_product_freq_below_6'
ordinal_cols = ['user_depth', 'age_level', 'city_development_index']
scale_cols = ['current_is_more_than_once',  'current_session_freq', 'current_same_campaign_freq',
              'current_same_product_freq']  # 'same_campaign_freq', 'session_freq', 'same_product_freq',
target_col = 'is_click'

# Hyperparameters for Grid Search
param_grids = {
    'random_forest': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        # 'bootstrap': [True, False]
    },
    'decision_tree': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

#,
    # 'light_gbm': {
    #     'num_leaves': [31, 40],
    #     'max_depth': [-1, 10, 20],
    #     'learning_rate': [0.05, 0.1],
    #     'n_estimators': [100, 200]
    # },
    # 'xgboost': {
    #     'max_depth': [3, 6],
    #     'learning_rate': [0.1, 0.01],
    #     'n_estimators': [100, 200],
    #     'subsample': [0.8, 1.0]
    # },
    # 'catboost': {
    #     'iterations': [100, 200],
    #     'depth': [6, 10],
    #     'learning_rate': [0.1, 0.05],
    #     'l2_leaf_reg': [3, 5]
    # },
    # 'histgradient_boosting': {
    #     'max_iter': [100, 200],
    #     'max_depth': [3, 5],
    #     'learning_rate': [0.05, 0.1],
    #     'min_samples_leaf': [20, 50]
    #}
}

# ADDED: Paths for the experiments
BEST_MODEL = "models/best_model.joblib"
RAW_DATA_PATH = "data/train_dataset_full.csv"
PREPROCESSED_DATA_PATH = "data/preprocessed_data.csv"
PREDICTIONS_PATH = "data/predictions.csv"
DEFAULT_N_ESTIMATORS = 100
DEFAULT_TREAT_NULL = 'A'


@task
def prepare_data_for_pipeline(c, csv_path=RAW_DATA_PATH):
    """Downloads or copies the input data to the project directory."""
    if csv_path.startswith("s3://"):
        c.run(f"aws s3 cp {csv_path} {RAW_DATA_PATH}")
    else:
        c.run(f"cp {csv_path} {RAW_DATA_PATH}")


@task
def preprocess(c, treat_null=DEFAULT_TREAT_NULL, model_name="random_forest"):
    """Runs the preprocessing step."""
    c.run(f"python preprocess.py --csv-path {RAW_DATA_PATH} "
          f"--one-hot-cols {' '.join(cols_for_one_hot_encoding)} "
          f"--binary-cols {' '.join(binary_cols)} "
          f"--ordinal-cols {' '.join(ordinal_cols)} "
          f"--scale-cols {' '.join(scale_cols)} "
          f"--target-col {target_col} "
          f"--train-val True "
          f"--treat-null {treat_null} "
          f"--model-name {model_name}")

@task
def train(c, model_name="random_forest", n_estimators=DEFAULT_N_ESTIMATORS, param_grid=None):
    """Runs the training step."""
    if param_grid is None:
        param_grid = json.dumps(param_grids[model_name])  # Convert Python dict to JSON string
    else:
        # Validate if param_grid is a JSON string
        try:
            json.loads(param_grid)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string for param_grid. Please check the syntax.")

    # Ensure param_grid  is wrapped in quotes to pass correctly as a CLI argument
    param_grid = param_grid.replace('"', '\\"')

    c.run(f"python train.py --model-name {model_name} "
          f"--n_estimators {n_estimators} "
          f"--param-grid \"{param_grid}\"")

@task
def predict(c, csv_path=PREPROCESSED_DATA_PATH, model_path=BEST_MODEL, output_path=PREDICTIONS_PATH):
    """Runs the prediction step."""
    c.run(f"python predict.py --csv-path {csv_path} "
          f"--model-path {model_path} "
          f"--output-csv-path {output_path}")


@task
def evaluate(c, predictions_path=PREDICTIONS_PATH):
    """Runs the evaluation step."""
    c.run(f"python results.py --csv-path {predictions_path}")


@task
def pipeline(c, model_name="random_forest", n_estimators=DEFAULT_N_ESTIMATORS, treat_null=DEFAULT_TREAT_NULL):
    """Runs the entire data processing and training pipeline."""
    prepare_data_for_pipeline(c)
    preprocess(c, treat_null, model_name)
    train(c, model_name, n_estimators)
    predict(c)
    evaluate(c)

@task
def preprocess_and_train(c, model_name="random_forest", n_estimators=DEFAULT_N_ESTIMATORS, treat_null=DEFAULT_TREAT_NULL):
    """Runs the preprocessing and training steps."""
    preprocess(c, treat_null, model_name)
    train(c, model_name, n_estimators)


@task
def archive(c, name, base_folder="archived_experiments"):
    """Archives an experiment by creating a new directory with a timestamped name."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    name = f"{timestamp}_{name}"
    print(f"Archived experiment: {name}")
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    exp_path = os.path.join(base_folder, name)
    os.makedirs(exp_path)

    # Copy data, models, and results to the archive folder
    if os.path.exists("data"):
        c.run(f"cp -r data {exp_path}/")
    if os.path.exists("models"):
        c.run(f"cp -r models {exp_path}/")
    if os.path.exists("results"):
        c.run(f"cp -r results {exp_path}/")

    print(f"Experiment archived in: {exp_path}")


# invoke preprocess-and-train --model-name random_forest --n-estimators 100 --treat-null A
# invoke pipeline --model-name random_forest --n_estimators 100 --treat-null A
# invoke train --model-name random_forest --n-estimators 50 --param-grid '{"max_depth": [null, 10, 20], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]}'