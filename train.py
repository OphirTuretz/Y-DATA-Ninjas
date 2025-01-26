import pandas as pd
import argparse
import json
import os
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier  # Corrected import
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# get data
def get_latest_file(suffix):
    """Find the latest file in the 'data' directory with the specified suffix."""
    files = [f for f in os.listdir("data") if f.endswith(suffix)]
    if not files:
        raise FileNotFoundError(f"No files found with suffix '{suffix}' in the 'data' directory.")
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join("data", x)))
    return os.path.join("data", latest_file)

def get_data():
    """Automatically fetch the latest train, test, and validation datasets."""
    train_data = pd.read_csv(get_latest_file("_train.csv"))
    test_data = pd.read_csv(get_latest_file("_test.csv"))
    val_data = pd.read_csv(get_latest_file("_val.csv"))
    return train_data, test_data, val_data

def train(model_name: str, param_grid_file: str, target_col: str, max_features: int = None):
    """
    Trains a model with the specified parameters.

    :param model_name: Name of the model (e.g., 'random_forest').
    :param param_grid_file: Path to the JSON file containing the parameter grid.
    :param target_col: Name of the target column.
    :param max_features: Number of estimators for the model.
    """
    # **Load parameter grid from JSON file**
    try:
        with open(param_grid_file, 'r') as file:
            all_param_grids = json.load(file)  # Load all parameter grids
    except FileNotFoundError:
        raise ValueError(f"Parameter grid file not found: {param_grid_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in parameter grid file: {e}")

    # **Filter the parameter grid for the specific model**
    param_grid = all_param_grids.get(model_name)
    if not param_grid:
        raise ValueError(f"No parameter grid found for model '{model_name}' in {param_grid_file}")

    print(f"Training model '{model_name}' with {max_features} max features and param_grid:")
    print(param_grid)

    train_data, test_data, val_data = get_data()
    x_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    x_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    x_val = val_data.drop(columns=[target_col])
    y_val = val_data[target_col]

    models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        #'light_gbm': LGBMClassifier(random_state=42), #TODO
        #'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        #'catboost': CatBoostClassifier(random_state=42),
        # 'hist_grad_boost': HistGradientBoostingClassifier(random_state=42, categorical_features=[] )
        # 'gradient_boosting'
    }

    # Select the model
    model = models.get(model_name)
    print("Model name:", model_name)

    # Feature selection
    if max_features:

        # Get the model instance and class
        model_instance = models[model_name]  # Model instance with parameters
        model_class = model_instance.__class__  # Extract class without parameters

        # Feature selection: Create a clean model for feature selection
        model_for_fs = model_class(random_state=42)

        # Fit the selector on x_train and y_train
        selector = SelectFromModel(model_for_fs, max_features=max_features)
        selector.fit(x_train, y_train)

        # Transform both x_train, x_val, and x_test
        x_train_selected = selector.transform(x_train)
        x_val_selected = selector.transform(x_val)
        x_test_selected = selector.transform(x_test)

        # Get the names of selected features
        selected_feature_names = x_train.columns[selector.get_support()]

        # Create new DataFrames with selected features
        x_train = pd.DataFrame(x_train_selected, columns=selected_feature_names)
        x_val = pd.DataFrame(x_val_selected, columns=selected_feature_names)
        x_test = pd.DataFrame(x_test_selected, columns=selected_feature_names)

    # Use validation set for grid search
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=2)  # TODO make grid sear optional?
    grid_search.fit(x_val, y_val)

    best_model = grid_search.best_estimator_
    print(f"Best parameters found by grid search: {grid_search.best_params_}")

    # Train the best model on the training set
    best_model.fit(x_train, y_train)

    # Evaluation
    y_pred_train = best_model.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    report_train = classification_report(y_train, y_pred_train)
    print("training accuracy:", accuracy_train)
    print("training classification report:", report_train)

    y_pred_val = best_model.predict(x_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    report_val = classification_report(y_val, y_pred_val)
    print(f"best validation accuracy with grid Search: {accuracy_val:.4f}")
    print("Validation classification report:", report_val)

    y_pred_test = best_model.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    report_test = classification_report(y_test, y_pred_test)
    print("test accuracy:", accuracy_test)
    print("test classification report:", report_test)

    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
        print("Feature Importance:", feature_importance)
        # DataFrame to show feature names with their importance
        feature_importance_df = pd.DataFrame({'feature': x_train.columns, 'importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        print(feature_importance_df)
    else:
        print("Feature importance is not available for this model.")


    return accuracy_train, report_train, accuracy_val, report_val, accuracy_test, report_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument('-m', '--model-name', default="random_forest", help="Name of the model")
    parser.add_argument('--param-grid-file', type=str, required=True, help="Path to the parameter grid JSON file")
    parser.add_argument('--target-col', type=str, default='is_click', help="Target column name")
    parser.add_argument('--max-features', type=int, default=100, help="Number of features")


    args = parser.parse_args()

    # Directly pass the file path to the train function
    train(
        model_name=args.model_name,
        param_grid_file=args.param_grid_file,  # Pass the file path
        target_col=args.target_col,
        max_features=args.max_features
    )





