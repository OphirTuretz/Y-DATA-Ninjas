import pandas as pd
from sklearn.model_selection import train_test_split
import os


file_path = "data/train_dataset_full.csv"  # Path to the original dataset
data = pd.read_csv(file_path)

print("Dataset loaded successfully. Columns:", data.columns)

feature_columns = ['session_id', 'DateTime', 'user_id', 'product', 'campaign_id',
       'webpage_id', 'product_category_1', 'product_category_2',
       'user_group_id', 'gender', 'age_level', 'user_depth',
       'city_development_index', 'var_1']
target_column = "is_click"

assert all(col in data.columns for col in feature_columns + [target_column]), "Column mismatch in dataset"

X = data[feature_columns]
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

output_folder = "data"

train_file_path = os.path.join(output_folder, "train.csv")
test_file_path = os.path.join(output_folder, "test.csv")

train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

print(f"Train and test datasets saved:\n - Train: {train_file_path}\n - Test: {test_file_path}")