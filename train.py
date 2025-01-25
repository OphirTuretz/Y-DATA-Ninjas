import pandas as pd
import argparse
import json
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier  # Corrected import
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, List, Any


# Remove the unnecessary part that loads data
def get_data():
    train_data = pd.read_csv("data/random_forest_A_20250125_211951_train.csv") # TODO automatically choose the last file that ends with _test
    test_data = pd.read_csv("data/random_forest_A_20250125_211951_test.csv")
    val_data = pd.read_csv("data/random_forest_A_20250125_211951_val.csv")
    return train_data, test_data, val_data

def train(model_name: str, param_grid: Dict[str, Dict[str, List[Any]]],
          target_col: str, n_estimators: int = None):

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
    }

    # Select the model
    model = models[model_name]
    print("Model name:", model_name)

    # Feature selection
    if n_estimators:
        # Use the selected model for feature selection
        if hasattr(model, 'n_estimators'):
            model_for_fs = model.__class__(n_estimators=n_estimators, random_state=42)
        else:
            # For models without n_estimators, use RandomForestClassifier as default
            model_for_fs = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        selector = SelectFromModel(model_for_fs)

        # Fit the selector on x_train and y_train
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
    grid_search = GridSearchCV(model, param_grid[model_name], n_jobs=-1, cv=1)  # TODO make grid sear optional?
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
    parser.add_argument('-m', '--model-name', default="random_forest")
    parser.add_argument('--param-grid', type=str, required=True,
                        help='JSON string of parameter grid')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--target_col', type=str, default='is_click')

    args = parser.parse_args()

    try:
        param_grid = json.loads(args.param_grid)  # Safely parse the JSON string
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string for param-grid. Please check the input format.")

    train(args.model_name, param_grid, args.target_col, args.n_estimators)





