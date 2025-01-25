import pandas as pd
import argparse
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import datetime

# Adjusted function for argparse to run preprocess in pipeline
def parse(csv_path, one_hot_cols, binary_cols, ordinal_cols, scale_cols, target_col,
          train_val=False, treat_null='A', model_name='random_forest'):
    # Adjusted preprocessing logic
    df = pd.read_csv(csv_path)

    if train_val:
        # Drop rows with null in the target column (only during training/validation)
        df = df.dropna(subset=[target_col])
        # Remove duplicate rows (only during training/validation)
        df = df.drop_duplicates()
        # Drop rows with nulls in `user_id` (only during training/validation)
        df = df.dropna(subset=['user_id'])

    # Convert column types
    df['session_id'] = df['session_id'].astype('string')
    df['user_id'] = df['user_id'].astype('string')
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M', errors='coerce')

    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('Int64')

    # Fill missing session_id with unique consecutive values
    if df['session_id'].isnull().any():
        missing_count = df['session_id'].isnull().sum()
        new_session_ids = [f"F{i}" for i in range(1, missing_count + 1)]
        df['session_id'] = df['session_id'].fillna(pd.Series(new_session_ids))

    # Create additional columns
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.day_name()
    df['current_session_freq'] = df.groupby('user_id', observed=True).cumcount() + 1
    df['current_is_more_than_once'] = (df['current_session_freq'] > 1).astype(int)
    df['current_same_campaign_freq'] = df.groupby(['user_id', 'campaign_id'], observed=True).cumcount() + 1
    df['current_same_product_freq'] = df.groupby(['user_id', 'product'], observed=True).cumcount() + 1

    # New Mapping of columns to sequential integers
    mapping = {}
    for col in ordinal_cols + one_hot_cols + binary_cols:
        unique_vals = df[col].dropna().unique()
        sorted_vals = sorted(unique_vals)
        val_mapping = {v: i for i, v in enumerate(sorted_vals)}
        val_mapping[np.nan] = np.nan  # Retain null values
        mapping[col] = val_mapping
        df[col] = df[col].map(val_mapping)

    mapping_df = pd.DataFrame([
        {'column': col, 'original_value': k, 'mapped_value': v}
        for col, val_map in mapping.items() for k, v in val_map.items()
    ])

    # Pipeline for handling missing values and scaling
    if treat_null == 'A':  # Drop NaN
        df = df.dropna()
    else:
        transformers = []

        # Handle one_hot_cols and binary_cols
        if one_hot_cols or binary_cols:
            if treat_null == 'C':  # Impute with mode
                cat_imputer = SimpleImputer(strategy='most_frequent')
            elif treat_null == 'D':  # Impute with max+1 and update mapping
                for col in one_hot_cols + binary_cols:
                    unique_vals = df[col].dropna().unique()
                    max_val = max(unique_vals)
                    df[col] = df[col].fillna(max_val + 1)
                    mapping[col][None] = max_val + 1
                mapping_df = pd.DataFrame([
                    {'column': col, 'original_value': k, 'mapped_value': v}
                    for col, val_map in mapping.items() for k, v in val_map.items()
                ])
                mapping_df.to_csv('mapping.csv', index=False)
                cat_imputer = 'passthrough'
            transformers.append(('cat_imputer', cat_imputer, one_hot_cols + binary_cols))

        # Handle ordinal_cols and scale_cols
        if ordinal_cols or scale_cols:
            num_imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()
            transformers.append(('num_pipeline', Pipeline([
                ('imputer', num_imputer),
                ('scaler', scaler)
            ]), ordinal_cols + scale_cols))

        # Apply transformations
        preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        df = pd.DataFrame(preprocessor.fit_transform(df), columns=df.columns)

    # Adjust preprocessing for the chosen model
    if model_name in ['random_forest', 'decision_tree']:
        # One-hot encode categorical columns (leave one group out)
        for col in one_hot_cols + [c for c in binary_cols if df[c].nunique() > 3]:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)

    # Train-test split
    x = df.drop(columns=[target_col])
    y = df[target_col]
    x_train_val , x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.4, random_state=42)

    train_data = pd.concat([x_train, y_train], axis=1)  # Combine features and target for training data
    val_data = pd.concat([x_val, y_val], axis=1)  # Combine features and target for testing data
    test_data = pd.concat([x_test, y_test], axis=1)  # Combine features and target for testing data

    # Create the filename format
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_format = f"{model_name}_{treat_null}_{timestamp}_{{data_type}}.csv"

    # Save the datasets
    train_data.to_csv(f"data/{filename_format.format(data_type='train')}", index=False)
    val_data.to_csv(f"data/{filename_format.format(data_type='val')}", index=False)
    test_data.to_csv(f"data/{filename_format.format(data_type='test')}", index=False)

    return x_train, x_test, y_train, y_test, x_val, y_val, mapping_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--one-hot-cols", nargs='+', type=str, required=True)
    parser.add_argument("--binary-cols", nargs='+', type=str, required=True)
    parser.add_argument("--ordinal-cols", nargs='+', type=str, required=True)
    parser.add_argument("--scale-cols", nargs='+', type=str, required=True)
    parser.add_argument("--target-col", type=str, required=True)
    parser.add_argument("--train-val", type=bool, default=False)
    parser.add_argument("--treat-null", type=str, choices=['A', 'B', 'C', 'D'], default='A')
    parser.add_argument("--model-name", type=str, choices=['random_forest', 'decision_tree'], default='random_forest')

    args = parser.parse_args()
    parse(args.csv_path, args.one_hot_cols, args.binary_cols, args.ordinal_cols, args.scale_cols, args.target_col, args.train_val, args.treat_null, args.model_name)







