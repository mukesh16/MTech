import numpy as np
import pandas as pd
from datetime import datetime

from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from bayes_opt import BayesianOptimization

# Load datasets and drop unnecessary columns
print("Loading datasets...")
df_train = pd.read_csv('train.csv').drop(['id'], axis=1)
df_test = pd.read_csv('test.csv').drop(['id'], axis=1)
df_sub = pd.read_csv('sample_submission.csv')
print("Datasets loaded successfully.")

# Feature engineering
print("Performing feature engineering...")
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
luxury_brands = ['Mercedes-Benz', 'BMW', 'Genesis', 'Audi', 'Lexus']
df_train['luxury_car'] = df_train['brand'].apply(lambda x: 1 if x in luxury_brands else 0)

df_test['car_age'] = current_year - df_test['model_year']
df_test['mileage_per_year'] = df_test['milage'] / df_test['car_age'].replace(0, 1)
df_test['accident_clean_interaction'] = df_test.apply(
    lambda x: 'No Accidents & Clean Title' if x['accident'] == 'None reported' and x['clean_title'] == 'Yes' else
              'Accident & Clean Title' if x['accident'] != 'None reported' and x['clean_title'] == 'Yes' else
              'Accident & No Clean Title',
    axis=1
)
df_test['horsepower'] = df_test['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
df_test['luxury_car'] = df_test['brand'].apply(lambda x: 1 if x in luxury_brands else 0)
print("Feature engineering complete.")

# Preprocessing function with KNN Imputer
print("Preprocessing and encoding data...")
def preprocess_and_encode(df_train, df_test):
    cat_cols = df_train.select_dtypes(include=['object']).columns
    num_cols = df_train.select_dtypes(include=['number']).columns
    
    num_cols = num_cols[num_cols != 'price']  # Exclude target column

    knn_imputer = KNNImputer(n_neighbors=5)
    df_train[num_cols] = knn_imputer.fit_transform(df_train[num_cols])
    df_test[num_cols] = knn_imputer.transform(df_test[num_cols])

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_train[cat_cols] = ordinal_encoder.fit_transform(df_train[cat_cols].astype(str))
    df_test[cat_cols] = ordinal_encoder.transform(df_test[cat_cols].astype(str))    
    
    return df_train, df_test

df_train, df_test = preprocess_and_encode(df_train, df_test)
print("Preprocessing complete.")

# Separate and transform target variable
print("Separating and transforming target variable...")
y = df_train['price'].astype('float32')
y_log = np.log1p(y)  # Log transformation to handle skewness
df_train = df_train.drop(['price'], axis=1)
print("Target variable separated and transformed.")

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
Scale_train = scaler.fit_transform(df_train)
Scale_test = scaler.transform(df_test)
print("Feature scaling complete.")

# Split the data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(Scale_train, y_log, test_size=0.2, random_state=42)
print("Data split complete.")

# Define RMSE function
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Bayesian Optimization for XGBoost with extended hyperparameters
def xgb_evaluate(max_depth, learning_rate, n_estimators, subsample, colsample_bytree, min_child_weight):
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
    
    # KFold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []
    
    for train_idx, test_idx in kf.split(X_train):
        X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
        y_fold_train, y_fold_test = y_train.iloc[train_idx], y_train.iloc[test_idx]  # Use .iloc for Pandas indexing
        
        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_test)
        rmse = root_mean_squared_error(y_fold_test, predictions)
        rmse_list.append(rmse)
    
    return np.mean(rmse_list)

# Bayesian Optimization
print("Starting Bayesian Optimization...")
xgb_bo = BayesianOptimization(
    xgb_evaluate,
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
print("Bayesian Optimization complete.")

# Retrieve the best parameters
print("Retrieving best parameters...")
best_params = xgb_bo.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['min_child_weight'] = int(best_params['min_child_weight'])
print("Best parameters retrieved.")

# Train final model with best parameters
print("Training final XGBoost model...")
best_xgb = XGBRegressor(**best_params, random_state=42)
best_xgb.fit(X_train, y_train)
best_xgb_pred = best_xgb.predict(X_test)
best_xgb_rmse = root_mean_squared_error(y_test, best_xgb_pred)
print(f"Best XGBoost RMSE with Bayesian Optimization: {best_xgb_rmse}")

# Predict on test data
print("Making predictions on test data...")
final_pred = best_xgb.predict(Scale_test)
df_sub['price'] = np.expm1(final_pred)  # Reverse the log transformation
submission_file = f'submission_{best_xgb_rmse}.csv'
df_sub.to_csv(submission_file, index=False)
print(f"Predictions saved to {submission_file}")