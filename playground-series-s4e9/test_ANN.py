import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
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

def create_neural_network(input_shape):
    """Create a neural network model."""
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

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

    # Create and train the neural network
    model = create_neural_network(X_train.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    logging.info(f"Neural Network RMSE: {rmse}")

    # Predict on test data
    final_pred = model.predict(Scale_test)
    df_sub['price'] = np.expm1(final_pred)  # Reverse the log transformation
    submission_file = f'submission_nn_{rmse:.4f}.csv'
    df_sub.to_csv(submission_file, index=False)
    logging.info(f"Predictions saved to {submission_file}")

if __name__ == "__main__":
    main()
