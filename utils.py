import pandas as pd
import numpy as np
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter(action='ignore', category=PerformanceWarning)


def invert_usd_columns(df):
    # Identify columns where 'Base' is USD in the first row
    usd_columns = df.loc[0] == 'USD'
    # Invert the values for these columns starting from index 1
    df.loc[1:, usd_columns] = 1 / df.loc[1:, usd_columns]
    
    # Drop the 'Base' row (row with index 0)
    df = df.drop(0).reset_index(drop=True)
    
    return df


def convert_date_column_for_monthly_data(df):
    return pd.to_datetime(df['date'], format='%Y%m') + pd.offsets.MonthEnd(1)




# I no longer use this function. I will delete it.
def calculate_log_returns(df):
    # Ensure the 'date' column is excluded from log return calculation
    df_numeric = df.drop(columns=['date'])
    df_numeric = df_numeric.astype(float)
    # Calculate the logarithmic returns using numpy's log function
    log_returns = np.log(df_numeric / df_numeric.shift(1)) * 100
    
    # Add back the 'date' column
    log_returns.insert(0, 'date', df['date'])
    
    # Drop the first row as it will have NaN values due to the shift
    log_returns = log_returns.dropna().reset_index(drop=True)
    
    return log_returns



# Note that this function can be used to calculate simple(non-excess) returns too. Just set x=y.
def calculate_log_FX_excess_returns(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of x and y
    x_copy = x.copy()
    y_copy = y.copy()
    
    # Ensure the date column is in datetime format for both dataframes
    x_copy['date'] = pd.to_datetime(x_copy['date'])
    y_copy['date'] = pd.to_datetime(y_copy['date'])
    
    # Shift the Y dataframe by 1 to align Y_t-1 with X_t
    y_shifted = y_copy.shift(1)
    
    # Calculate log returns for every column except the 'date' column
    log_return_df = pd.DataFrame()
    log_return_df['date'] = x_copy['date']  # Preserve the date column
    
    for column in x_copy.columns:
        if column != 'date':
            # Ensure we handle cases where the data is non-numeric or NaN
            x_numeric = pd.to_numeric(x_copy[column], errors='coerce')
            y_numeric = pd.to_numeric(y_shifted[column], errors='coerce')
            
            # Apply log transformation only to positive values, ignore others (e.g., NaN or non-positive values)
            log_return_df[column] = (np.log(x_numeric) - np.log(y_numeric)) * 100
    
    # Reset the index and ensure index is as a normal column
    log_return_df.reset_index(drop=True, inplace=True)
    
    return log_return_df


def build_signal_df_for_1month_momentum(df):
    signal_df = pd.DataFrame()
    signal_df["date"] = df["date"]
    # Note that I shift signals one period forward to make computations easier. 
    signal_df= signal_df.join(df.iloc[:, 1:].shift(1))
    return signal_df




def calculate_fx_carry_signal(spot: pd.DataFrame, futures: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of x and y
    spot_copy = spot.copy()
    futures_copy = futures.copy()
    
    # Ensure the date column is in datetime format for both dataframes
    spot_copy['date'] = pd.to_datetime(spot_copy['date'])
    futures_copy['date'] = pd.to_datetime(futures_copy['date'])
    
    
    # Calculate log returns for every column except the 'date' column
    carry_df = pd.DataFrame()
    carry_df['date'] = spot_copy['date']  # Preserve the date column
    
    for column in spot_copy.columns:
        if column != 'date':
            # Ensure we handle cases where the data is non-numeric or NaN
            x_numeric = pd.to_numeric(spot_copy[column], errors='coerce')
            y_numeric = pd.to_numeric(futures_copy[column], errors='coerce')
            
            # Apply log transformation only to positive values, ignore others (e.g., NaN or non-positive values)
            carry_df[column] = (x_numeric - y_numeric) * 100
    
    # Reset the index and ensure index is as a normal column
    carry_df.reset_index(drop=True, inplace=True)
    signal_df = pd.DataFrame()
    signal_df["date"] = carry_df["date"]
    # Note that I shift signals one period forward to make computations easier. 
    signal_df= signal_df.join(carry_df.iloc[:, 1:].shift(1))
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


def process_fx_single_currency_dataset(adress):
    df = pd.read_excel(adress, skiprows=2)
    df = invert_usd_columns(df)
    df = df.rename(columns={"Currency":"date"})
    df['date'] = convert_date_column_for_monthly_data(df)
    return df


def construct_monthly_return_FX_portfolios_datasets(df):
    df['date'] = pd.to_datetime(df['date'])

    # Step 2: Set 'Date' as the index to enable resampling
    df.set_index('date', inplace=True)

    # Step 3: Resample to get the last price of each month (monthly frequency 'M')
    monthly_prices = df.resample('M').last()

    # Step 4: Calculate the monthly returns for each column (except 'Date')
    monthly_returns = monthly_prices.pct_change() * 100

    # Step 5: Reset the index to have 'Date' as a column again
    monthly_returns.reset_index(inplace=True)
    monthly_returns.dropna(inplace=True)

    return monthly_returns


def exclude_redundant_columns_FX_portfolios_datasets(df):
    corr_matrix = df.iloc[:,1:].corr()
    to_drop = []

    # Iterate over the columns of the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.98 or corr_matrix.iloc[i, j] < -0.98:
                colname = corr_matrix.columns[j]
                if colname not in to_drop:
                    to_drop.append(colname)

    # Now drop the identified columns from the dataframe, keeping only the first
    filtered_corr_matrix = corr_matrix.drop(columns=to_drop, index=to_drop)
    columns_to_keep = filtered_corr_matrix.columns.to_list()

    df = df[['date']+columns_to_keep]
    
    return df