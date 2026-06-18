import pandas as pd
import numpy as np

def process_smhi(filename):
    df = pd.read_csv(filename, sep=" ", names=("YY", "MM", "DD", "TT", "G(i)"))

    df['datetime'] = (
            df['YY'].astype(str) +
            df['MM'].astype(str).str.zfill(2) +
            df['DD'].astype(str).str.zfill(2) + ':' +
            df['TT'].astype(str).str.zfill(2)
    )
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d:%H')
    time = df['datetime'].copy()

    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['hour_of_day'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    X = df[['G(i)', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']]

    return X, time
