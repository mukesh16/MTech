import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from bayes_opt import BayesianOptimization
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


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
    """Perform advanced feature engineering."""
    current_year = datetime.now().year
    df_train['car_age'] = current_year - df_train['model_year']
    df_train['mileage_per_year'] = df_train['milage'] / df_train['car_age'].replace(0, 1)

    df_test['car_age'] = current_year - df_test['model_year']
    df_test['mileage_per_year'] = df_test['milage'] / df_test['car_age'].replace(0, 1)

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
    
    df_train['horsepower'] = df_train['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
    # df_train['horsepower'].fillna(df_train['horsepower'].median(), inplace=True)
    df_train['horsepower'] = df_train['horsepower'].fillna(df_train['horsepower'].median())
    df_test['horsepower'] = df_test['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
    # df_test['horsepower'].fillna(df_test['horsepower'].median(), inplace=True)
    df_test['horsepower'] = df_test['horsepower'].fillna(df_test['horsepower'].median())

    df_train['car_age_squared'] = df_train['car_age'] ** 2
    df_test['car_age_squared'] = df_test['car_age'] ** 2

    df_train['log_mileage'] = np.log1p(df_train['milage'])
    df_test['log_mileage'] = np.log1p(df_test['milage'])

    df_train['hp_per_year'] = df_train['horsepower'] / df_train['car_age'].replace(0, 1)
    df_test['hp_per_year'] = df_test['horsepower'] / df_test['car_age'].replace(0, 1)
    
    df_train['fuel_efficiency'] = df_train['horsepower'] / df_train['milage'].replace(0, 1)
    df_test['fuel_efficiency'] = df_test['horsepower'] / df_test['milage'].replace(0, 1)

    bins = [0, 150, 300, np.inf]
    labels = [1, 2, 3]
    df_train['engine_power_category'] = pd.cut(df_train['horsepower'], bins=bins, labels=labels)
    df_test['engine_power_category'] = pd.cut(df_test['horsepower'], bins=bins, labels=labels)

    mileage_threshold = df_train['milage'].quantile(0.99)
    df_train['milage'] = np.where(df_train['milage'] > mileage_threshold, mileage_threshold, df_train['milage'])
    df_test['milage'] = np.where(df_test['milage'] > mileage_threshold, mileage_threshold, df_test['milage'])

    logging.info("Advanced feature engineering complete.")
    return df_train, df_test

def preprocess_and_encode(df_train, df_test):
    """Preprocess and encode data."""
    
    cat_cols = df_train.select_dtypes(include=['object']).columns
    num_cols = df_train.select_dtypes(include=['number']).columns
    num_cols = num_cols[num_cols != 'price']
    
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df_train[num_cols] = num_imputer.fit_transform(df_train[num_cols])
    df_test[num_cols] = num_imputer.transform(df_test[num_cols])
    
    df_train[cat_cols] = cat_imputer.fit_transform(df_train[cat_cols].astype(str))
    df_test[cat_cols] = cat_imputer.transform(df_test[cat_cols].astype(str))

    df_train = pd.get_dummies(df_train, columns=['engine_power_category'], prefix='power_cat', drop_first=True)
    df_test = pd.get_dummies(df_test, columns=['engine_power_category'], prefix='power_cat', drop_first=True)
    
    for col in num_cols:
        threshold = df_train[col].quantile(0.99)
        df_train[col] = np.where(df_train[col] > threshold, threshold, df_train[col])
        df_test[col] = np.where(df_test[col] > threshold, threshold, df_test[col])

    scaler = StandardScaler()
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test[num_cols] = scaler.transform(df_test[num_cols])

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_train[cat_cols] = ordinal_encoder.fit_transform(df_train[cat_cols])
    df_test[cat_cols] = ordinal_encoder.transform(df_test[cat_cols])

    logging.info("Preprocessing and encoding complete.")
    return df_train, df_test

def root_mean_squared_error(y_true, y_pred):
    """Calculate RMSE."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def build_dnn_model(input_shape, learning_rate, dropout_rate, n_neurons, n_layers):
    """Build and compile a DNN model."""
    model = Sequential()
    
    # Adding the Input layer
    model.add(Input(shape=(input_shape,)))
    
    # Adding hidden layers
    for i in range(int(n_layers)):
        model.add(Dense(int(n_neurons), activation='relu'))
        
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    
    # Adding the output layer
    model.add(Dense(1))  # Output layer for regression task
    
    # Compiling the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    
    return model

def dnn_evaluate(X_train, y_train, learning_rate, n_neurons, n_layers, dropout_rate):
    """Evaluate DNN model with given hyperparameters."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []

    for train_idx, test_idx in kf.split(X_train):
        X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
        y_fold_train, y_fold_test = y_train.iloc[train_idx], y_train.iloc[test_idx]

        model = build_dnn_model(X_fold_train.shape[1], learning_rate, dropout_rate, n_neurons, n_layers)
        model.fit(X_fold_train, y_fold_train, epochs=100, batch_size=32, verbose=0)
        
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
    logging.info("Starting Bayesian Optimization.")
    
    def bo_evaluate(learning_rate, n_neurons, n_layers, dropout_rate):
        """Wrapper function for Bayesian Optimization."""
        return -dnn_evaluate(X_train, y_train, learning_rate, n_neurons, n_layers, dropout_rate)
    
    pbounds = {
        'learning_rate': (1e-5, 1e-2),
        'n_neurons': (10, 200),
        'n_layers': (1, 5),
        'dropout_rate': (0, 0.5)
    }

    optimizer = BayesianOptimization(
        f=bo_evaluate,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    optimizer.maximize(init_points=5, n_iter=15)

    best_params = optimizer.max['params']
    logging.info(f"Best parameters found: {best_params}")

    # Final evaluation
    final_rmse = dnn_evaluate(X_train, y_train, **best_params)
    logging.info(f"Final RMSE on training data: {final_rmse}")

    # Train final model on full training set
    final_model = build_dnn_model(X_train.shape[1], **best_params)
    final_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    
    predictions = final_model.predict(Scale_test)
    df_sub['price'] = np.expm1(predictions)
    
    df_sub.to_csv('final_submission.csv', index=False)
    logging.info("Submission file created.")

if __name__ == "__main__":
    main()
