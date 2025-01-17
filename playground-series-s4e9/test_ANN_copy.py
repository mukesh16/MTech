import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def load_data(train_path, test_path, sub_path):
    df_train = pd.read_csv(train_path).drop(['id'], axis=1)
    df_test = pd.read_csv(test_path).drop(['id'], axis=1)
    df_sub = pd.read_csv(sub_path)
    return df_train, df_test, df_sub

def feature_engineering(df_train, df_test):
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

    return df_train, df_test

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

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def augment_data(X, y, augmentation_factor=2):
    X_augmented = []
    y_augmented = []
    for _ in range(augmentation_factor):
        noise = np.random.normal(0, 0.01, X.shape)
        X_augmented.append(X + noise)
        y_augmented.append(y + noise[:, 0])  # Assuming y is the first column of X for noise addition
    
    return np.vstack(X_augmented), np.hstack(y_augmented)

def create_neural_network(input_shape):
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
    df_train, df_test, df_sub = load_data('train.csv', 'test.csv', 'sample_submission.csv')

    df_train, df_test = feature_engineering(df_train, df_test)

    df_train, df_test = preprocess_and_encode(df_train, df_test)

    y = df_train['price'].astype('float32')
    y_log = np.log1p(y)
    df_train = df_train.drop(['price'], axis=1)

    scaler = StandardScaler()
    Scale_train = scaler.fit_transform(df_train)
    Scale_test = scaler.transform(df_test)

    X_train, X_test, y_train, y_test = train_test_split(Scale_train, y_log, test_size=0.2, random_state=42)

    X_aug, y_aug = augment_data(X_train, y_train)

    model = create_neural_network(X_train.shape[1])
    model.fit(X_aug, y_aug, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    print(f"Neural Network RMSE: {rmse}")

    final_pred = model.predict(Scale_test)
    df_sub['price'] = np.expm1(final_pred)  # Reverse the log transformation
    df_sub.to_csv(f'submission_nn_{rmse:.4f}.csv', index=False)

if __name__ == "__main__":
    main()
    