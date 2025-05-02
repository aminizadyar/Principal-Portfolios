import pandas as pd
import numpy as np
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter(action='ignore', category=PerformanceWarning)

def convert_date_column_for_monthly_data(df):
    """
    Convert a 'YYYYMM' string or integer date column into a proper pandas datetime,
    then shift each date to the end of that month.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'date' column in the format YYYYMM.

    Returns
    -------
    pandas.DatetimeIndex
        Date values converted to the last calendar day of each month.
    """
    # Parse 'date' strings or integers as datetime using year-month format,
    # then add a MonthEnd offset of 1 to move to the month's last day.
    return pd.to_datetime(df['date'], format='%Y%m') + pd.offsets.MonthEnd(1)


def build_signal_df_for_1month_momentum(df):
    """
    Construct a signal DataFrame for one-month momentum by shifting returns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a 'date' column and asset values/returns in other columns.

    Returns
    -------
    pandas.DataFrame
        New DataFrame with:
        - 'date' column copied from the input,
        - All other columns shifted forward by one period to represent last month's signal.
    """
    # Initialize the signal DataFrame with the same dates
    signal_df = pd.DataFrame()
    signal_df["date"] = df["date"]

    # Shift all asset columns by one row so that each date’s signal comes from the prior period
    signal_df = signal_df.join(df.iloc[:, 1:].shift(1))

    return signal_df


def compute_period_returns(df, periods):
    """
    Compute period-over-period percentage returns for each asset column,
    while preserving the original 'date' column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'date' column plus numeric asset price/value columns.
    periods : int
        Number of periods over which to compute the percentage change.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - 'date' column intact in position 0,
        - Asset columns replaced by their pct_change over the specified periods.
    """
    # Preserve the original 'date' column for re-insertion
    date_column = df['date']

    # Identify columns to calculate returns (exclude 'date')
    columns_to_calculate = df.columns.difference(['date'])

    # Compute the percentage change over the given number of periods
    returns_df = df[columns_to_calculate].pct_change(periods=periods)

    # Reinsert the 'date' column as the first column
    returns_df.insert(0, 'date', date_column)

    return returns_df
