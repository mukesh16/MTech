import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from bayes_opt import BayesianOptimization
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_path, test_path, sub_path):
    """Load datasets and drop unnecessary columns."""
    try:
        df_train = pd.read_csv(train_path).drop(['id'], axis=1)
        df_test = pd.read_csv(test_path).drop(['id'], axis=1)
        df_sub = pd.read_csv(sub_path)
        logging.info("Datasets loaded successfully.")
        return df_train, df_test, df_sub
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        raise

def feature_engineering(df_train, df_test):
    """Perform feature engineering."""
    current_year = datetime.now().year
    df_train['car_age'] = current_year - df_train['model_year']
    df_train['mileage_per_year'] = df_train['milage'] / df_train['car_age'].replace(0, 1)
    df_train['accident_clean_interaction'] = df_train.apply(
        lambda x: 'No Accidents & Clean Title' if x['accident'] == 'None reported' and x['clean_title'] == 'Yes' else
                  'Accident & Clean Title' if x['accident'] != 'None reported' and x['clean_title'] == 'Yes' else
                  'Accident & No Clean Title',
        axis=1
    )
    df_train['horsepower'] = df_train['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
    # luxury_brands = ['Mercedes-Benz', 'BMW', 'Genesis', 'Audi', 'Lexus']
    # df_train['luxury_car'] = df_train['brand'].apply(lambda x: 1 if x in luxury_brands else 0)

    df_test['car_age'] = current_year - df_test['model_year']
    df_test['mileage_per_year'] = df_test['milage'] / df_test['car_age'].replace(0, 1)
    df_test['accident_clean_interaction'] = df_test.apply(
        lambda x: 'No Accidents & Clean Title' if x['accident'] == 'None reported' and x['clean_title'] == 'Yes' else
                  'Accident & Clean Title' if x['accident'] != 'None reported' and x['clean_title'] == 'Yes' else
                  'Accident & No Clean Title',
        axis=1
    )
    df_test['horsepower'] = df_test['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
    # df_test['luxury_car'] = df_test['brand'].apply(lambda x: 1 if x in luxury_brands else 0)

    logging.info("Feature engineering complete.")
    return df_train, df_test

def preprocess_and_encode(df_train, df_test):
    """Preprocess and encode data."""
    cat_cols = df_train.select_dtypes(include=['object']).columns
    num_cols = df_train.select_dtypes(include=['number']).columns

    num_cols = num_cols[num_cols != 'price']  # Exclude target column

    knn_imputer = KNNImputer(n_neighbors=5)
    df_train[num_cols] = knn_imputer.fit_transform(df_train[num_cols])
    df_test[num_cols] = knn_imputer.transform(df_test[num_cols])

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_train[cat_cols] = ordinal_encoder.fit_transform(df_train[cat_cols].astype(str))
    df_test[cat_cols] = ordinal_encoder.transform(df_test[cat_cols].astype(str))

    logging.info("Preprocessing complete.")
    return df_train, df_test

def root_mean_squared_error(y_true, y_pred):
    """Calculate RMSE."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def xgb_evaluate(X_train, y_train, max_depth, learning_rate, n_estimators, subsample, colsample_bytree, min_child_weight):
    """Evaluate XGBoost model with given hyperparameters."""
    params = {
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': int(min_child_weight),
        'eval_metric': 'rmse',
        'objective': 'reg:squarederror',
        'random_state': 42
    }
    model = XGBRegressor(**params)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []

    for train_idx, test_idx in kf.split(X_train):
        X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
        y_fold_train, y_fold_test = y_train.iloc[train_idx], y_train.iloc[test_idx]

        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_test)
        rmse = root_mean_squared_error(y_fold_test, predictions)
        rmse_list.append(rmse)

    return np.mean(rmse_list)

def main():
    # Load data
    df_train, df_test, df_sub = load_data('train.csv', 'test.csv', 'sample_submission.csv')

    # Feature engineering
    df_train, df_test = feature_engineering(df_train, df_test)

    # Preprocessing and encoding
    df_train, df_test = preprocess_and_encode(df_train, df_test)

    # Separate and transform target variable
    y = df_train['price'].astype('float32')
    y_log = np.log1p(y)
    df_train = df_train.drop(['price'], axis=1)

    # Standardize features
    scaler = StandardScaler()
    Scale_train = scaler.fit_transform(df_train)
    Scale_test = scaler.transform(df_test)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(Scale_train, y_log, test_size=0.2, random_state=42)

    # Bayesian Optimization
    logging.info("Starting Bayesian Optimization...")
    xgb_bo = BayesianOptimization(
        lambda max_depth, learning_rate, n_estimators, subsample, colsample_bytree, min_child_weight: xgb_evaluate(
            X_train, y_train, max_depth, learning_rate, n_estimators, subsample, colsample_bytree, min_child_weight
        ),
        {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (100, 500),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.3, 1),
            'min_child_weight': (1, 10),
        },
        random_state=42
    )
    xgb_bo.maximize(init_points=5, n_iter=25)

    # Retrieve the best parameters
    best_params = xgb_bo.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])

    # Train final model
    logging.info("Training final XGBoost model...")
    best_xgb = XGBRegressor(**best_params, random_state=42)
    best_xgb.fit(X_train, y_train)
    best_xgb_pred = best_xgb.predict(X_test)
    best_xgb_rmse = root_mean_squared_error(y_test, best_xgb_pred)
    logging.info(f"Best XGBoost RMSE with Bayesian Optimization: {best_xgb_rmse}")

    # Predict on test data
    logging.info("Making predictions on test data...")
    final_pred = best_xgb.predict(Scale_test)
    df_sub['price'] = np.expm1(final_pred)  # Reverse the log transformation
    submission_file = f'submission_{best_xgb_rmse:.4f}.csv'
    df_sub.to_csv(submission_file, index=False)
    logging.info(f"Predictions saved to {submission_file}")

if __name__ == "__main__":
    main()