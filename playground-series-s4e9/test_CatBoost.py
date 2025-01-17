import numpy as np
import pandas as pd
from datetime import datetime
from catboost import CatBoostRegressor, Pool
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
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
    # Feature Engineering for Training Data
    df_train['car_age'] = current_year - df_train['model_year']
    df_train['mileage_per_year'] = df_train['milage'] / df_train['car_age'].replace(0, 1)
    df_train['accident_clean_interaction'] = df_train.apply(
        lambda x: 'No Accidents & Clean Title' if x['accident'] == 'None reported' and x['clean_title'] == 'Yes' else
                  'Accident & Clean Title' if x['accident'] != 'None reported' and x['clean_title'] == 'Yes' else
                  'Accident & No Clean Title',
        axis=1
    )
    df_train['horsepower'] = df_train['engine'].str.extract(r'(\d+\.?\d*)').astype(float)

    # Feature Engineering for Test Data
    df_test['car_age'] = current_year - df_test['model_year']
    df_test['mileage_per_year'] = df_test['milage'] / df_test['car_age'].replace(0, 1)
    df_test['accident_clean_interaction'] = df_test.apply(
        lambda x: 'No Accidents & Clean Title' if x['accident'] == 'None reported' and x['clean_title'] == 'Yes' else
                  'Accident & Clean Title' if x['accident'] != 'None reported' and x['clean_title'] == 'Yes' else
                  'Accident & No Clean Title',
        axis=1
    )
    df_test['horsepower'] = df_test['engine'].str.extract(r'(\d+\.?\d*)').astype(float)

    logging.info("Feature engineering complete.")
    return df_train, df_test

def preprocess_and_encode(df_train, df_test):
    """Preprocess and encode data."""
    # Identify categorical and numerical columns
    cat_cols = df_train.select_dtypes(include=['object', 'category']).columns
    num_cols = df_train.select_dtypes(include=['number']).columns

    num_cols = num_cols[num_cols != 'price']  # Exclude the target column

    # KNN Imputation for missing numerical data
    knn_imputer = KNNImputer(n_neighbors=5)
    df_train[num_cols] = knn_imputer.fit_transform(df_train[num_cols])
    df_test[num_cols] = knn_imputer.transform(df_test[num_cols])

    imputer = SimpleImputer(strategy='most_frequent')
    df_train[cat_cols] = imputer.fit_transform(df_train[cat_cols])
    df_test[cat_cols] = imputer.transform(df_test[cat_cols])

    # Convert any numeric categorical features to strings, including non-integer numbers
    for col in df_train.columns:
        if col == 'price':
            continue
        if col in cat_cols or df_train[col].dtype in ['float64', 'int64']:
            df_train[col] = df_train[col].astype(str)
            df_test[col] = df_test[col].astype(str)

    # Ordinal encoding for categorical variables
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_train[cat_cols] = ordinal_encoder.fit_transform(df_train[cat_cols])
    df_test[cat_cols] = ordinal_encoder.transform(df_test[cat_cols])

    # Get the indices of the categorical columns
    cat_features_indices = [df_train.columns.get_loc(col) for col in cat_cols]

    logging.info("Preprocessing complete.")
    return df_train, df_test, cat_features_indices

def root_mean_squared_error(y_true, y_pred):
    """Calculate RMSE."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def catboost_evaluate(learning_rate, depth, l2_leaf_reg, iterations, X_train, y_train, cat_features):
    """Evaluate CatBoost model with given hyperparameters."""
    params = {
        'learning_rate': learning_rate,
        'depth': int(depth),
        'l2_leaf_reg': l2_leaf_reg,
        'iterations': int(iterations),
        'loss_function': 'RMSE',
        'verbose': 0,
        'random_seed': 42
    }
    model = CatBoostRegressor(**params)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []

    for train_idx, test_idx in kf.split(X_train):
        X_fold_train, X_fold_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_fold_train, y_fold_test = y_train.iloc[train_idx], y_train.iloc[test_idx]

        model.fit(X_fold_train, y_fold_train, cat_features=cat_features, verbose=0)
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
    df_train, df_test, cat_features_indices = preprocess_and_encode(df_train, df_test)

    # Separate and transform target variable
    y = df_train['price'].astype('float32')
    y_log = np.log1p(y)
    df_train = df_train.drop(['price'], axis=1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df_train, y_log, test_size=0.2, random_state=42)

    # Bayesian Optimization for CatBoost
    logging.info("Starting Bayesian Optimization...")
    # Define the optimization function with fixed X_train, y_train, and cat_features
    def optimization_func(learning_rate, depth, l2_leaf_reg, iterations):
        return catboost_evaluate(
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            iterations=iterations,
            X_train=X_train,
            y_train=y_train,
            cat_features=cat_features_indices
        )

    catboost_bo = BayesianOptimization(
        f=optimization_func,
        pbounds={
            'learning_rate': (0.01, 0.3),
            'depth': (3, 10),
            'l2_leaf_reg': (1, 10),
            'iterations': (100, 1000)
        },
        random_state=42,
        verbose=2  # verbose=2 prints all steps
    )
    catboost_bo.maximize(init_points=5, n_iter=25)

    # Retrieve the best parameters
    best_params = catboost_bo.max['params']
    best_params['depth'] = int(best_params['depth'])
    best_params['iterations'] = int(best_params['iterations'])

    # Train final model with best parameters
    logging.info("Training final CatBoost model...")
    best_catboost = CatBoostRegressor(**best_params, random_seed=42, loss_function='RMSE')
    best_catboost.fit(X_train, y_train, cat_features=cat_features_indices, verbose=0)

    # Evaluate on test set
    best_catboost_pred = best_catboost.predict(X_test)
    best_catboost_rmse = root_mean_squared_error(y_test, best_catboost_pred)
    logging.info(f"Best CatBoost RMSE with Bayesian Optimization: {best_catboost_rmse}")

    # Predict on test data
    logging.info("Making predictions on test data...")
    final_pred = best_catboost.predict(df_test)
    df_sub['price'] = np.expm1(final_pred)  # Reverse the log transformation
    submission_file = f'submission_{best_catboost_rmse:.4f}.csv'
    df_sub.to_csv(submission_file, index=False)
    logging.info(f"Predictions saved to {submission_file}")

if __name__ == "__main__":
    main()