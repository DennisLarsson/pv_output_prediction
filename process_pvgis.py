import pandas as pd
import numpy as np

def process_pvgis (filepath):
    df = pd.read_csv(filepath, header=8, nrows=17520)
    df['time'] = df['time'].astype(str).str[:-2]
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H')
    df['day_of_year'] = df['time'].dt.dayofyear
    df['hour_of_day'] = df['time'].dt.hour

    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    X = df[['G(i)', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']]
    y = df[['P']]

    X_train, X_test = X.iloc[:8760], X.iloc[8760:]
    y_train, y_test = y.iloc[:8760], y.iloc[8760:]
    return X_train, X_test, y_train, y_test
