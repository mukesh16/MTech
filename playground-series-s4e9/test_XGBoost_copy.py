import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
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

# def feature_engineering(df_train, df_test):
#     """Perform feature engineering."""
#     current_year = datetime.now().year
#     df_train['car_age'] = current_year - df_train['model_year']
#     df_train['mileage_per_year'] = df_train['milage'] / df_train['car_age'].replace(0, 1)

#     # Interaction feature between accident and clean title
#     df_train['accident_clean_interaction'] = df_train.apply(
#         lambda x: 'No Accidents & Clean Title' if x['accident'] == 'None reported' and x['clean_title'] == 'Yes' else
#                   'Accident & Clean Title' if x['accident'] != 'None reported' and x['clean_title'] == 'Yes' else
#                   'Accident & No Clean Title',
#         axis=1
#     )
#     # Horsepower extraction
#     df_train['horsepower'] = df_train['engine'].str.extract(r'(\d+\.?\d*)').astype(float)

#     # Polynomial feature: car_age^2
#     df_train['car_age_squared'] = df_train['car_age'] ** 2

#     df_test['car_age'] = current_year - df_test['model_year']
#     df_test['mileage_per_year'] = df_test['milage'] / df_test['car_age'].replace(0, 1)
#     df_test['accident_clean_interaction'] = df_test.apply(
#         lambda x: 'No Accidents & Clean Title' if x['accident'] == 'None reported' and x['clean_title'] == 'Yes' else
#                   'Accident & Clean Title' if x['accident'] != 'None reported' and x['clean_title'] == 'Yes' else
#                   'Accident & No Clean Title',
#         axis=1
#     )
#     df_test['horsepower'] = df_test['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
#     df_test['car_age_squared'] = df_test['car_age'] ** 2

#     logging.info("Feature engineering complete.")
#     return df_train, df_test

def feature_engineering(df_train, df_test):
    """Perform advanced feature engineering."""
    current_year = datetime.now().year
    
    # Create car_age and mileage_per_year
    df_train['car_age'] = current_year - df_train['model_year']
    df_train['mileage_per_year'] = df_train['milage'] / df_train['car_age'].replace(0, 1)
    
    df_test['car_age'] = current_year - df_test['model_year']
    df_test['mileage_per_year'] = df_test['milage'] / df_test['car_age'].replace(0, 1)

    # Interaction feature between accident and clean title
    df_train['accident_clean_interaction'] = df_train.apply(
        lambda x: 'No Accidents & Clean Title' if x['accident'] == 'None reported' and x['clean_title'] == 'Yes' else
                  'Accident & Clean Title' if x['accident'] != 'None reported' and x['clean_title'] == 'Yes' else
                  'Accident & No Clean Title',
        axis=1
    )
    
    df_test['accident_clean_interaction'] = df_test.apply(
        lambda x: 'No Accidents & Clean Title' if x['accident'] == 'None reported' and x['clean_title'] == 'Yes' else
                  'Accident & Clean Title' if x['accident'] != 'None reported' and x['clean_title'] == 'Yes' else
                  'Accident & No Clean Title',
        axis=1
    )
    
    # Horsepower extraction and missing handling
    df_train['horsepower'] = df_train['engine'].str.extract(r'(\d+\.?\d*)').astype(float).fillna(df_train['engine'].str.extract(r'(\d+)').astype(float).median())
    df_test['horsepower'] = df_test['engine'].str.extract(r'(\d+\.?\d*)').astype(float).fillna(df_test['engine'].str.extract(r'(\d+)').astype(float).median())

    # Polynomial feature: car_age^2
    df_train['car_age_squared'] = df_train['car_age'] ** 2
    df_test['car_age_squared'] = df_test['car_age'] ** 2

    # Log transform on mileage to handle skewness
    df_train['log_mileage'] = np.log1p(df_train['milage'])
    df_test['log_mileage'] = np.log1p(df_test['milage'])

    # Interaction: horsepower per year and engine power-to-weight ratio
    df_train['hp_per_year'] = df_train['horsepower'] / df_train['car_age'].replace(0, 1)
    df_test['hp_per_year'] = df_test['horsepower'] / df_test['car_age'].replace(0, 1)
    
    # New Feature: Fuel efficiency approximation (engine size vs mileage)
    df_train['fuel_efficiency'] = df_train['horsepower'] / df_train['milage'].replace(0, 1)
    df_test['fuel_efficiency'] = df_test['horsepower'] / df_test['milage'].replace(0, 1)

    # Engine power categorization: low, medium, high
    bins = [0, 150, 300, np.inf]
    # labels = ['Low Power', 'Medium Power', 'High Power']
    labels = [1, 2, 3]
    df_train['engine_power_category'] = pd.cut(df_train['horsepower'], bins=bins, labels=labels)
    df_test['engine_power_category'] = pd.cut(df_test['horsepower'], bins=bins, labels=labels)

    # Clean mileage outliers: Handling extreme values for better model performance
    mileage_threshold = df_train['milage'].quantile(0.99)
    df_train['milage'] = np.where(df_train['milage'] > mileage_threshold, mileage_threshold, df_train['milage'])
    df_test['milage'] = np.where(df_test['milage'] > mileage_threshold, mileage_threshold, df_test['milage'])

    logging.info("Advanced feature engineering complete.")
    return df_train, df_test

# def preprocess_and_encode(df_train, df_test):
#     """Preprocess and encode data."""
#     cat_cols = df_train.select_dtypes(include=['object']).columns
#     num_cols = df_train.select_dtypes(include=['number']).columns

#     num_cols = num_cols[num_cols != 'price']  # Exclude target column

#     # Use SimpleImputer to handle missing data
#     imputer = SimpleImputer(strategy='median')
#     df_train[num_cols] = imputer.fit_transform(df_train[num_cols])
#     df_test[num_cols] = imputer.transform(df_test[num_cols])

#     # Ordinal encoding for categorical variables
#     ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
#     df_train[cat_cols] = ordinal_encoder.fit_transform(df_train[cat_cols].astype(str))
#     df_test[cat_cols] = ordinal_encoder.transform(df_test[cat_cols].astype(str))

#     logging.info("Preprocessing complete.")
#     return df_train, df_test

def preprocess_and_encode(df_train, df_test):
    """Preprocess and encode data."""
    
    # Identify categorical and numerical columns
    cat_cols = df_train.select_dtypes(include=['object']).columns
    num_cols = df_train.select_dtypes(include=['number']).columns
    
    # Exclude target column 'price' from numerical features
    num_cols = num_cols[num_cols != 'price']
    
    # Use SimpleImputer to handle missing data
    # - Median for numerical columns
    # - Most frequent for categorical columns
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Impute missing values for numerical columns
    df_train[num_cols] = num_imputer.fit_transform(df_train[num_cols])
    df_test[num_cols] = num_imputer.transform(df_test[num_cols])
    
    # Impute missing values for categorical columns
    df_train[cat_cols] = cat_imputer.fit_transform(df_train[cat_cols].astype(str))
    df_test[cat_cols] = cat_imputer.transform(df_test[cat_cols].astype(str))

    # One-hot encode the 'engine_power_category' column
    df_train = pd.get_dummies(df_train, columns=['engine_power_category'], prefix='power_cat', drop_first=True)
    df_test = pd.get_dummies(df_test, columns=['engine_power_category'], prefix='power_cat', drop_first=True)
    
    # Handle outliers by capping numerical columns at the 99th percentile
    for col in num_cols:
        threshold = df_train[col].quantile(0.99)
        df_train[col] = np.where(df_train[col] > threshold, threshold, df_train[col])
        df_test[col] = np.where(df_test[col] > threshold, threshold, df_test[col])

    # Scaling numerical features using StandardScaler
    scaler = StandardScaler()
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test[num_cols] = scaler.transform(df_test[num_cols])

    # Ordinal encoding for categorical variables
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_train[cat_cols] = ordinal_encoder.fit_transform(df_train[cat_cols])
    df_test[cat_cols] = ordinal_encoder.transform(df_test[cat_cols])

    # Optional: Apply One-Hot Encoding for low-cardinality categorical variables (if needed)
    # Uncomment if you want to handle low-cardinality categorical columns with one-hot encoding
    # df_train = pd.get_dummies(df_train, columns=cat_cols, drop_first=True)
    # df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)

    logging.info("Preprocessing and encoding complete.")
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

    print(df_train.head())
    print(df_train.info())

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