import numpy as np
import pandas as pd
from datetime import datetime

from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# Load datasets and drop unnecessary columns
df_train = pd.read_csv('train.csv').drop(['id'], axis=1)
df_test = pd.read_csv('test.csv').drop(['id'], axis=1)
df_sub = pd.read_csv('sample_submission.csv')

# Check for missing values
df_train.isnull().sum()
df_test.isnull().sum()

# Feature engineering
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

# Basic exploration
df_train.shape
df_train.describe()
df_train.shape, df_test.shape
df_train.info()

# Preprocessing function
def preprocess_and_encode(df_train, df_test): 

    cat_cols = df_train.select_dtypes(include=['object']).columns
    num_cols = df_train.select_dtypes(include=['number']).columns
    
    num_cols = num_cols[num_cols != 'price']   

    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='median')

    df_train[cat_cols] = cat_imputer.fit_transform(df_train[cat_cols])
    df_test[cat_cols] = cat_imputer.transform(df_test[cat_cols])
    df_train[num_cols] = num_imputer.fit_transform(df_train[num_cols])
    df_test[num_cols] = num_imputer.transform(df_test[num_cols])

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_train[cat_cols] = ordinal_encoder.fit_transform(df_train[cat_cols].astype(str))
    df_test[cat_cols] = ordinal_encoder.transform(df_test[cat_cols].astype(str))    
    
    return df_train, df_test

df_train, df_test = preprocess_and_encode(df_train, df_test)

# Separate target variable
y = df_train['price'].astype('float32')
df_train = df_train.drop(['price'], axis=1)

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Scale_train = scaler.fit_transform(df_train)
Scale_test = scaler.transform(df_test)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(Scale_train, y, test_size=0.2, random_state=42)

# XGBoost Regressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_rmse = root_mean_squared_error(y_test, xgb_pred)
print(f"XGBoost RMSE: {xgb_rmse}")

# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'n_estimators': [100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.3, 0.5, 0.6, 0.7]
}

grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_
# Evaluate the best model
best_xgb_pred = best_xgb.predict(X_test)
best_xgb_rmse = root_mean_squared_error(y_test, best_xgb_pred)
print(f"Best XGBoost RMSE: {best_xgb_rmse}")

# Print best model hyperparameters
print(best_xgb.n_estimators, best_xgb.max_depth, best_xgb.learning_rate, best_xgb.subsample)

# Predict on test data
finalpred = best_xgb.predict(Scale_test)
df_sub['price'] = finalpred
df_sub.to_csv(f'submission_{best_xgb_rmse}.csv', index=False)