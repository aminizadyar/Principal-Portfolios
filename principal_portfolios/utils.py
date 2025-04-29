import pandas as pd
import numpy as np
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter(action='ignore', category=PerformanceWarning)

def convert_date_column_for_monthly_data(df):
    return pd.to_datetime(df['date'], format='%Y%m') + pd.offsets.MonthEnd(1)

def build_signal_df_for_1month_momentum(df):
    signal_df = pd.DataFrame()
    signal_df["date"] = df["date"]
    # Note that I shift signals one period forward to make computations easier. 
    signal_df= signal_df.join(df.iloc[:, 1:].shift(1))
    return signal_df

def compute_period_returns(df, periods):
    # Ensure 'date' column is kept intact
    date_column = df['date']
    # Columns to compute percentage change for
    columns_to_calculate = df.columns.difference(['date'])
    # Calculate the percentage change over the specified number of periods
    returns_df = df[columns_to_calculate].pct_change(periods=periods)
    # Add the 'date' column back into the DataFrame
    returns_df.insert(0, 'date', date_column)
    return returns_df
