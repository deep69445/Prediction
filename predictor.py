import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df['MA_3'] = df['Close'].rolling(window=3).mean()
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    df = df.dropna()
    return df

def train_model(df: pd.DataFrame):
    df = prepare_features(df)

    features = ['MA_3', 'MA_7', 'Lag_1', 'Lag_2', 'Lag_3']
    X = df[features]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return model, predictions[-1], mae, rmse  # last predicted value, errors

def predict_next_day(model, df: pd.DataFrame):
    df = prepare_features(df)
    latest = df.iloc[-1][['MA_3', 'MA_7', 'Lag_1', 'Lag_2', 'Lag_3']].values.reshape(1, -1)
    return model.predict(latest)[0]