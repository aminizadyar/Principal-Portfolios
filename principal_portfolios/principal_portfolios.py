# Import essential libraries for data handling, numerical computing, modeling, and visualization
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter(action='ignore', category=PerformanceWarning)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def rank_and_map(df):
    """
    Transform cross-sectional signals by ranking raw values and normalizing them
    to the interval [-0.5, 0.5] for each date.
    """
    # Create a shallow copy to preserve the original DataFrame
    df_copy = df.copy()
    # Identify columns containing asset signals (assumes 'date' is the first column)
    data_columns = df_copy.columns[1:]
    
    def rank_row(row):
        """
        Given a cross-section of asset values for a single date, perform:
        1. Integer ranking (smallest value -> rank 1).
        2. Scaling of ranks to [0, 1]: (rank - 1) / (n - 1).
        3. Centering to [-0.5, 0.5] by subtracting 0.5.
        """
        # Compute ranks with ties assigned the minimum rank
        ranks = row.rank(method='min')
        # Scale to [0, 1] proportional to rank position
        ranks_normalized = (ranks - 1) / (len(row) - 1)
        # Shift center to zero for long/short symmetry
        return ranks_normalized - 0.5
    
    # Apply the ranking and mapping procedure to each row of signal columns
    df_copy[data_columns] = df_copy[data_columns].apply(rank_row, axis=1)
    return df_copy


def cross_sectional_demean(df):
    """
    Remove cross-sectional mean across assets for each time period, centering data on zero.

    Parameters:
    df : pd.DataFrame
        Input DataFrame where the first column is 'date' and remaining columns are asset values.

    Returns:
    pd.DataFrame
        A new DataFrame where each asset column has had its cross-sectional mean
        subtracted for every date, yielding zero-mean exposures across assets.
    """
    # Create a copy to preserve original data integrity
    df_copy = df.copy()
    # Identify asset columns (assumes 'date' in first position)
    data_columns = df_copy.columns[1:]
    
    def demean_row(row):
        """
        Center a single cross-section by subtracting the mean of that row.

        Steps:
        1. Compute the arithmetic mean of row values.
        2. Subtract this mean from each element to zero-center the row.
        """
        # Compute the average of current cross-section
        row_mean = row.mean()
        # Subtract the mean to center values around zero
        return row - row_mean
    
    # Apply cross-sectional demeaning to every date's asset values
    df_copy[data_columns] = df_copy[data_columns].apply(demean_row, axis=1)
    return df_copy


def compute_rs_product(df1, df2):
    """
    Compute the cross-sectional outer product matrix R S′ for each date.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame containing a 'date' column and numerical data columns (e.g., returns).
    df2 : pd.DataFrame
        DataFrame containing a 'date' column and numerical data columns (e.g., signals).
        Must have identical 'date' values (in the same order) as df1.

    Returns
    -------
    dict
        Mapping from each date to its corresponding (n × n) numpy array,
        computed as R (from df1) times S′ (from df2).

    Raises
    ------
    ValueError
        If the 'date' columns of df1 and df2 do not match exactly.
    """
    # Ensure that both DataFrames share the same dates in the same order
    if not df1['date'].equals(df2['date']):
        raise ValueError("Date columns of both dataframes must match.")

    # Cast all non-date columns to float64, coercing any invalid entries to NaN
    df1 = df1.astype({col: 'float64' for col in df1.columns if col != 'date'})
    df2 = df2.astype({col: 'float64' for col in df2.columns if col != 'date'})

    # Initialize a dictionary to hold the RS′ matrix for each date
    result = {}

    # Iterate row-by-row: index gives the row, date gives the current timestamp
    for index, date in enumerate(df1['date']):
        # Build R as an (n × 1) column vector from df1 (all columns except 'date')
        R = df1.iloc[index, 1:].values.reshape(-1, 1)
        # Build S′ as a (1 × n) row vector from df2 (all columns except 'date')
        S_transpose = df2.iloc[index, 1:].values.reshape(1, -1)
        # Compute the outer product R × S′, yielding an (n × n) matrix
        matrix_rs = np.dot(R, S_transpose)
        # Store the result under the corresponding date key
        result[date] = matrix_rs

    return result



def get_prediction_matrix(input_date, result_matrices, n_periods):
    """
    Generate the average prediction matrix for a given date by averaging the
    previous `n_periods` matrices from the `result_matrices` dictionary.

    Parameters
    ----------
    input_date : hashable
        The date key for which the prediction matrix is desired.
    result_matrices : dict
        Mapping from dates to their corresponding (n × n) numpy arrays.
    n_periods : int
        Number of prior periods to include in the average (excluding input_date).

    Returns
    -------
    numpy.ndarray
        The averaged (n × n) matrix over the selected prior dates.

    Raises
    ------
    ValueError
        If `input_date` is not present in `result_matrices`, or if there are
        no prior dates to compute the average for the specified `n_periods`.
    """
    # Sort all available dates so they are in chronological order
    sorted_dates = sorted(result_matrices.keys())

    # Confirm the requested date exists in the sorted list
    if input_date not in sorted_dates:
        raise ValueError("The input date is not found in the result_matrices.")

    # Find the position of the input date in the sorted list
    input_date_index = sorted_dates.index(input_date)

    # Determine where to start selecting previous dates, not going below zero
    start_index = max(0, input_date_index - n_periods)

    # Extract the dates immediately before the input_date (up to n_periods)
    selected_dates = sorted_dates[start_index:input_date_index]

    # If there are no dates to average, alert the user
    if len(selected_dates) == 0:
        raise ValueError(f"There are no previous periods to calculate the average for the given number: {n_periods}.")

    # Initialize an accumulator matrix of zeros matching the shape of the stored matrices
    matrix_shape = result_matrices[sorted_dates[0]].shape
    sum_matrix = np.zeros(matrix_shape, dtype=float)

    # Sum each of the matrices corresponding to the selected dates
    for date in selected_dates:
        sum_matrix += np.array(result_matrices[date], dtype=float)

    # Compute the element-wise average across the summed matrices
    average_matrix = sum_matrix / len(selected_dates)
    return average_matrix


# ============================================================================
# Principal Portfolio (PP), Principal Exposure(PEP), and
# Principal Alpha Portfolio (PAP) helper functions
# ============================================================================

def get_ith_PPs_expected_return(S, i):
    """
    Retrieve the expected return (singular value) of the i-th Principal Portfolio.

    Parameters
    ----------
    S : array-like
        1D array of singular values from SVD of the prediction matrix.
    i : int
        Index of the Principal Portfolio (0-based).

    Returns
    -------
    float
        The i-th singular value, representing the expected return for PP i.
    """
    return S[i]


def get_ith_position_matrix(U, VT, i):
    """
    Construct the position matrix for the i-th Principal Portfolio.

    This is the outer product of the i-th right singular vector (row of VT)
    and the i-th left singular vector (column of U).

    Parameters
    ----------
    U : ndarray, shape (n, n)
        Left singular vectors from SVD (columns are u_i).
    VT : ndarray, shape (n, n)
        Right singular vectors from SVD (rows are v_i^T).
    i : int
        Index of the Principal Portfolio (0-based).

    Returns
    -------
    ndarray, shape (n, n)
        Position matrix for PP i: v_i (row) outer u_i (column).
    """
    u_column = U[:, i]
    v_row    = VT[i, :]
    return np.outer(v_row, u_column)


def first_n_PPs_expected_return(S, n):
    """
    Sum the expected returns of the first n Principal Portfolios.

    Parameters
    ----------
    S : array-like
        Singular values vector from SVD.
    n : int
        Number of top Principal Portfolios to include (0-based count).

    Returns
    -------
    float
        Sum of the first n singular values.
    """
    total = 0.0
    for i in range(n):
        total += get_ith_PPs_expected_return(S, i)
    return total


def first_n_PPs_position_matrix(U, VT, number_of_PPs):
    """
    Compute the average position matrix of the first n Principal Portfolios.

    Parameters
    ----------
    U : ndarray, shape (n, n)
        Left singular vectors from SVD.
    VT : ndarray, shape (n, n)
        Right singular vectors from SVD.
    number_of_PPs : int
        Number of top Principal Portfolios to average.

    Returns
    -------
    ndarray, shape (n, n)
        Averaged position matrix of the first n PPs.
    """
    n_rows, _ = U.shape
    sum_matrix = np.zeros((n_rows, n_rows), dtype=float)
    for i in range(number_of_PPs):
        sum_matrix += get_ith_position_matrix(U, VT, i)
    return sum_matrix / number_of_PPs


def get_ith_PEPs_expected_return(eigenvalues, i):
    """
    Retrieve the expected return (eigenvalue) of the i-th Principal Exposure Portfolio.

    Parameters
    ----------
    eigenvalues : array-like
        Eigenvalues from the symmetric part of the prediction matrix.
    i : int
        Index of the PEP (0-based).

    Returns
    -------
    float
        The i-th eigenvalue.
    """
    return eigenvalues[i]


def get_ith_symmetric_position_matrix(eigenvectors, i):
    """
    Build the position matrix for the i-th Principal Exposure(PEP).

    This is the outer product of eigenvector w_i with itself (w_i w_i^T).

    Parameters
    ----------
    eigenvectors : ndarray, shape (n, n)
        Columns are eigenvectors of the symmetric prediction matrix.
    i : int
        Index of the PEP (0-based).

    Returns
    -------
    ndarray, shape (n, n)
        Symmetric position matrix for PEP i.
    """
    w = eigenvectors[:, i]
    return np.outer(w, w)


def first_n_PEPs_expected_return(eigenvalues, n):
    """
    Sum the absolute expected returns of the first n Principal Exposure Portfolios.

    Uses absolute values to capture both long and short contributions.

    Parameters
    ----------
    eigenvalues : array-like
        Sorted eigenvalues from the symmetric prediction matrix.
    n : int
        Number of PEPs to include (0-based count).

    Returns
    -------
    float
        Sum of |eigenvalues[0:n]|.
    """
    total = 0.0
    for i in range(n):
        total += abs(get_ith_PEPs_expected_return(eigenvalues, i))
    return total


def first_n_PEPs_position_matrix(eigenvectors, number_of_PEPs):
    """
    Compute the average position matrix of the first n Principal Exposure Portfolios.

    Parameters
    ----------
    eigenvectors : ndarray, shape (n, n)
        Eigenvectors of the symmetric prediction matrix (columns w_i).
    number_of_PEPs : int
        Number of top PEPs to average.

    Returns
    -------
    ndarray, shape (n, n)
        Averaged symmetric position matrix of the first n PEPs.
    """
    n_rows, _ = eigenvectors.shape
    sum_matrix = np.zeros((n_rows, n_rows), dtype=float)
    for i in range(number_of_PEPs):
        sum_matrix += get_ith_symmetric_position_matrix(eigenvectors, i)
    return sum_matrix / number_of_PEPs


def last_n_PEPs_position_matrix(eigenvectors, number_of_PEPs):
    """
    Compute the average position matrix of the last n Principal Exposure Portfolios.

    Parameters
    ----------
    eigenvectors : ndarray, shape (n, n)
        Eigenvectors of the symmetric prediction matrix.
    number_of_PEPs : int
        Number of bottom-ranked PEPs to average.

    Returns
    -------
    ndarray, shape (n, n)
        Averaged symmetric position matrix of the last n PEPs.
    """
    n_rows, _ = eigenvectors.shape
    sum_matrix = np.zeros((n_rows, n_rows), dtype=float)
    for i in range(number_of_PEPs):
        idx = n_rows - i - 1
        sum_matrix += get_ith_symmetric_position_matrix(eigenvectors, idx)
    return sum_matrix / number_of_PEPs


def get_ith_PAPs_expected_return(filtered_eigenvalues_ta, i):
    """
    Retrieve the expected return of the i-th Principal Alpha Portfolio.

    PAP expected return is defined as 2 × the imaginary part of the i-th
    eigenvalue of the antisymmetric prediction matrix.

    Parameters
    ----------
    filtered_eigenvalues_ta : array-like
        Positive imaginary parts of eigenvalues from the antisymmetric matrix.
    i : int
        Index of the PAP (0-based).

    Returns
    -------
    float
        2 × filtered_eigenvalues_ta[i].
    """
    return 2 * filtered_eigenvalues_ta[i]


def get_ith_asymmetric_position_matrix(real_part, imag_part, i):
    """
    Construct the position matrix for the i-th Principal Alpha Portfolio (PAP).

    This is the skew-symmetric outer product:
        w_real[:,i] w_imag[:,i]^T − w_imag[:,i] w_real[:,i]^T

    Parameters
    ----------
    real_part : ndarray, shape (n, m)
        Real components of sorted antisymmetric eigenvectors.
    imag_part : ndarray, shape (n, m)
        Imaginary components of sorted antisymmetric eigenvectors.
    i : int
        Index of the PAP (0-based).

    Returns
    -------
    ndarray, shape (n, n)
        Asymmetric (skew-symmetric) position matrix for PAP i.
    """
    w_real = real_part[:, i]
    w_imag = imag_part[:, i]
    return np.outer(w_real, w_imag) - np.outer(w_imag, w_real)


def first_n_PAPs_expected_return(filtered_eigenvalues_ta, n):
    """
    Sum the expected returns of the first n Principal Alpha Portfolios.

    Parameters
    ----------
    filtered_eigenvalues_ta : array-like
        Filtered positive imaginary parts from antisymmetric eigenvalues.
    n : int
        Number of PAPs to include (0-based count).

    Returns
    -------
    float
        Sum of 2 × filtered_eigenvalues_ta[0:n].
    """
    total = 0.0
    for i in range(n):
        total += get_ith_PAPs_expected_return(filtered_eigenvalues_ta, i)
    return total


def first_n_PAPs_position_matrix(real_part, imag_part, number_of_PAPs):
    """
    Compute the average position matrix of the first n Principal Alpha Portfolios.

    Parameters
    ----------
    real_part : ndarray, shape (n, m)
        Real components of antisymmetric eigenvectors.
    imag_part : ndarray, shape (n, m)
        Imaginary components of antisymmetric eigenvectors.
    number_of_PAPs : int
        Number of top PAPs to average.

    Returns
    -------
    ndarray, shape (n, n)
        Averaged asymmetric position matrix of the first n PAPs.
    """
    n_rows, _ = real_part.shape
    sum_matrix = np.zeros((n_rows, n_rows), dtype=float)
    for i in range(number_of_PAPs):
        sum_matrix += get_ith_asymmetric_position_matrix(real_part, imag_part, i)
    return sum_matrix / number_of_PAPs


def calculate_sharpe_ratio(returns):
    """
    Compute the Sharpe Ratio for a series of returns.

    Parameters
    ----------
    returns : array-like or pandas.Series
        Sequence of excess returns (e.g., monthly or daily returns minus risk-free rate).

    Returns
    -------
    float
        Sharpe Ratio, defined as the mean of the returns divided by their standard deviation.
    """
    # Compute the mean of the return series
    average_return = returns.mean()
    # Compute the standard deviation of the return series
    std_dev_returns = returns.std()
    # Divide average by volatility to obtain the Sharpe Ratio
    sharpe_ratio = average_return / std_dev_returns
    return sharpe_ratio


def filter_dataframes_by_common_dates(df1, df2, is_date_index=True):
    """
    Align two DataFrames by keeping only the rows corresponding to dates they share.

    Depending on `is_date_index`, dates are taken either from the DataFrame index
    or from a 'date' column.

    Parameters
    ----------
    df1, df2 : pandas.DataFrame
        Input DataFrames to be aligned.
    is_date_index : bool, default True
        If True, use the DataFrame index to identify dates; 
        if False, use each DataFrame’s 'date' column.

    Returns
    -------
    df1_filtered, df2_filtered : pandas.DataFrame
        Copies of the original DataFrames filtered down to rows whose dates appear in both.
    """
    if is_date_index:
        # Find the intersection of the two DataFrame indices (dates)
        common_dates = df1.index.intersection(df2.index)
        # Select only those rows whose index is in the common date set
        df1_filtered = df1.loc[common_dates]
        df2_filtered = df2.loc[common_dates]
    else:
        # Find the intersection of the 'date' column values in each DataFrame
        common_dates = set(df1['date']).intersection(df2['date'])
        # Filter rows to include only those with a 'date' in the intersection
        df1_filtered = df1[df1['date'].isin(common_dates)]
        df2_filtered = df2[df2['date'].isin(common_dates)]

    return df1_filtered, df2_filtered


def regression_results(X, Y):
    """
    Perform an annualized regression analysis of scaled returns on factor exposures.

    Parameters
    ----------
    X : pandas.DataFrame
        Independent variables (factor returns); each column represents a factor.
    Y : pandas.Series or array-like
        Dependent variable (realized returns) to be regressed.

    Returns
    -------
    list
        [ 
          coefficients : ndarray
              Estimated regression coefficients (intercept first, then factors),
              with the intercept annualized.
          t_stats : ndarray
              Corresponding t‐statistics for each coefficient.
          intercept_over_std_residuals : float
              Annualized information ratio of the intercept 
              (alpha divided by residual volatility).
          r_squared : float
              Coefficient of determination of the regression.
          variable_names : list of str
              Names of the regression variables, including "const" for the intercept.
        ]

    Notes
    -----
    - Y is scaled so that its standard deviation matches that of the first non-constant
      column in X, ensuring comparable units.
    - Intercept and information ratio are annualized by multiplying by √12.
    """
    # Compute the standard deviation of the first non-constant factor in X
    std_X1 = X.iloc[:, 1].std()

    # Compute the standard deviation of Y
    std_Y = Y.std()
    # Scale Y so its volatility matches that of X.iloc[:, 1]
    Y_scaled = Y * (std_X1 / std_Y)

    # Add an intercept column ("const") to X for the regression
    X = sm.add_constant(X)

    # Fit the OLS model of scaled Y on X (with intercept)
    model = sm.OLS(Y_scaled, X).fit()

    # Extract estimated coefficients (intercept + factor loadings)
    coefficients = model.params.values
    # Extract t-statistics for each estimated parameter
    t_stats = model.tvalues.values

    # Compute the sample standard deviation of the residuals
    residuals = model.resid
    std_residuals = residuals.std(ddof=1)

    # Calculate the information ratio of the intercept (alpha)
    # Annualize by multiplying by sqrt(12)
    intercept_over_std_residuals = (coefficients[0] / std_residuals) * math.sqrt(12)

    # Record the R-squared goodness-of-fit
    r_squared = model.rsquared

    # Annualize the intercept coefficient itself for comparability
    coefficients[0] = coefficients[0] * 12

    # Capture the names of all regression variables (including "const")
    variable_names = X.columns.tolist()

    # Return results in the prescribed format
    return [coefficients, t_stats, intercept_over_std_residuals, r_squared, variable_names]


def build_PP(
    input_return_dataset_df,
    signal_df,
    number_of_lookback_periods,
    starting_year_to_filter,
    end_year_to_filter,
    portfolio_formation_df=None,
    factor_data_monthly=None,
    number_of_PPs_to_consider=3,
    number_of_PEPs_to_consider=3,
    number_of_PAPs_to_consider=3,
    use_demeaned_returns=True
):
    """
    Construct and evaluate Principal Portfolios (PP), Principal Exposure (PEP),
    and Principal Alpha Portfolios (PAP) using historical returns and signals.

    Parameters
    ----------
    input_return_dataset_df : pandas.DataFrame
        Must contain a 'date' column and asset returns (excess or simple).
        Used to compute the R matrix for prediction.
    signal_df : pandas.DataFrame
        Must contain a 'date' column and matching asset columns.
        Used to compute the S matrix (signals) for prediction.
    number_of_lookback_periods : int
        Number of past periods to average when forming the prediction matrix.
    starting_year_to_filter : int
        Exclude all data on or before this year.
    end_year_to_filter : int
        Exclude all data on or after this year.
    portfolio_formation_df : pandas.DataFrame, optional
        Must contain a 'date' column and asset returns for realized performance.
        If None, defaults to input_return_dataset_df.
    factor_data_monthly : pandas.DataFrame, optional
        Must contain a 'date' column and factor returns (e.g., Fama–French).
        If provided, will run regressions of each strategy on these factors.
    number_of_PPs_to_consider : int, default=3
        How many of the top singular‐value portfolios to aggregate in summary stats.
    number_of_PEPs_to_consider : int, default=3
        How many of the top symmetric exposure portfolios to aggregate.
    number_of_PAPs_to_consider : int, default=3
        How many of the top asymmetric alpha portfolios to aggregate.
    use_demeaned_returns : bool, default=True
        If True, cross‐sectionally demean returns before computing RS′.

    Returns
    -------
    dict
        - realized_returns_df : DataFrame indexed by date, containing realized and
          expected returns for the simple factor, PP, PEP, and PAP strategies.
        - sharpe_df : Series of annualized Sharpe ratios for each realized series.
        - pp_realized_mean_df, pp_expected_mean_df : Series of average realized and
          expected returns across individual PPs.
        - pep_realized_mean_df, pep_expected_mean_df : Same for PEPs.
        - pap_realized_mean_df, pap_expected_mean_df : Same for PAPs.
        - regression_result_* : Regression outputs vs. factor_data_monthly, if provided.
    """
    # Use input returns for portfolio formation if none provided
    if portfolio_formation_df is None:
        portfolio_formation_df = input_return_dataset_df

    # Align returns and signals on common dates (using 'date' column)
    input_return_dataset_df, signal_df = filter_dataframes_by_common_dates(
        input_return_dataset_df.dropna(),
        signal_df.dropna(),
        is_date_index=False
    )

    # Rank and normalize signals to the range [-0.5, 0.5] (interpreted as S_{t-1})
    normalized_signal_df = rank_and_map(signal_df)
    # Filter both returns and signals to the specified year window
    mask = lambda df: (
        (df['date'].dt.year > starting_year_to_filter) &
        (df['date'].dt.year < end_year_to_filter)
    )
    normalized_signal_df = normalized_signal_df[mask(normalized_signal_df)].reset_index(drop=True)
    input_return_dataset_df = input_return_dataset_df[mask(input_return_dataset_df)].reset_index(drop=True)

    # Optionally demean returns cross‐sectionally for each date (R_{t-1})
    if use_demeaned_returns:
        return_matrix_df = cross_sectional_demean(input_return_dataset_df)
    else:
        return_matrix_df = input_return_dataset_df.copy()

    # Compute RS′ for each date: a dict mapping date → (n×n) matrix
    rs_matrix = compute_rs_product(return_matrix_df, normalized_signal_df)

    # Prepare a DataFrame to collect realized & expected returns
    realized_returns_df = pd.DataFrame(columns=[
        "date",
        "return_of_simple_factor",
        "realized_return_of_first_n_PP",
        "expected_return_of_first_n_PP",
        "realized_return_of_first_n_PEP",
        "realized_return_of_last_n_PEP",
        "long_short_realized_PEP",
        "expected_return_of_first_n_PEP",
        "realized_return_of_first_n_PAP",
        "expected_return_of_first_n_PAP"
    ])

    # Loop over dates, skipping the initial lookback periods
    for date_index in return_matrix_df.iloc[number_of_lookback_periods:]['date']:
        date_to_consider = pd.Timestamp(date_index)

        # Form the averaged prediction matrix for this date
        prediction_matrix = get_prediction_matrix(
            date_to_consider, rs_matrix, number_of_lookback_periods
        )

        # Decompose prediction_matrix via SVD for Principal Portfolios (PP)
        U, S, VT = np.linalg.svd(prediction_matrix)

        # Symmetric part for Principal Exposure (PEP)
        Sym = (prediction_matrix + prediction_matrix.T) / 2
        eigenvalues, eigenvectors = np.linalg.eig(Sym)
        idx = eigenvalues.argsort()[::-1]  # descending sort
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Antisymmetric part for Principal Alpha Portfolios (PAP)
        Asym = 0.5 * (prediction_matrix - prediction_matrix.T)
        eigvals_ta, eigvecs_ta = np.linalg.eig(Asym.T)
        order_ta = np.argsort(-eigvals_ta.imag)
        sorted_vals_ta = eigvals_ta[order_ta].imag
        sorted_vecs_ta = eigvecs_ta[:, order_ta] * math.sqrt(2)  # normalize
        pos_idx = np.where(sorted_vals_ta > 0)
        filtered_vals_ta = sorted_vals_ta[pos_idx]
        filtered_vecs_ta = sorted_vecs_ta[:, pos_idx].squeeze()
        real_ta, imag_ta = filtered_vecs_ta.real, filtered_vecs_ta.imag

        # Extract the 1×n signal vector and n×1 return vector for this date
        signal_vector = normalized_signal_df[
            normalized_signal_df.date == date_to_consider
        ].values[0, 1:].reshape(1, -1)
        return_vector = portfolio_formation_df[
            portfolio_formation_df.date == date_to_consider
        ].values[0, 1:].reshape(-1, 1)

        # Compute realized return of the simple factor
        return_of_simple_factor = (signal_vector @ return_vector)[0][0]

        # First‐n PP realized & expected returns
        realized_return_of_first_n_PP = float(
            signal_vector @ first_n_PPs_position_matrix(U, VT, number_of_PPs_to_consider) @ return_vector
        )
        expected_return_of_first_n_PP = first_n_PPs_expected_return(S, number_of_PPs_to_consider)

        # First & last n PEP realized returns, and expected return of top‐n
        realized_return_of_first_n_PEP = float(
            signal_vector @ first_n_PEPs_position_matrix(eigenvectors, number_of_PEPs_to_consider) @ return_vector
        )
        realized_return_of_last_n_PEP = float(
            signal_vector @ last_n_PEPs_position_matrix(eigenvectors, number_of_PEPs_to_consider) @ return_vector
        )
        long_short_realized_PEP = realized_return_of_first_n_PEP - realized_return_of_last_n_PEP
        expected_return_of_first_n_PEP = first_n_PEPs_expected_return(eigenvalues, number_of_PEPs_to_consider)

        # First‐n PAP realized & expected returns
        realized_return_of_first_n_PAP = float(
            signal_vector @ first_n_PAPs_position_matrix(real_ta, imag_ta, number_of_PAPs_to_consider) @ return_vector
        )
        expected_return_of_first_n_PAP = first_n_PAPs_expected_return(filtered_vals_ta, number_of_PAPs_to_consider)

        # Build the row of core metrics for this date
        row_values = [
            date_index,
            return_of_simple_factor,
            realized_return_of_first_n_PP,
            expected_return_of_first_n_PP,
            realized_return_of_first_n_PEP,
            realized_return_of_last_n_PEP,
            long_short_realized_PEP,
            expected_return_of_first_n_PEP,
            realized_return_of_first_n_PAP,
            expected_return_of_first_n_PAP
        ]

        # Append detailed metrics for each individual PP and PEP
        for i in range(len(S)):
            # PP i
            row_values.extend([
                float(signal_vector @ get_ith_position_matrix(U, VT, i) @ return_vector),
                get_ith_PPs_expected_return(S, i)
            ])
            # PEP i
            row_values.extend([
                float(signal_vector @ get_ith_symmetric_position_matrix(eigenvectors, i) @ return_vector),
                get_ith_PEPs_expected_return(eigenvalues, i)
            ])
            # Dynamically ensure columns exist
            for suffix in ['PP', 'PEP']:
                for kind in ['realized', 'expected']:
                    col = f"{kind}_return_of_{i+1}_{suffix}"
                    if col not in realized_returns_df.columns:
                        realized_returns_df[col] = None

        # Append detailed metrics for each PAP
        for i in range(imag_ta.shape[1]):
            row_values.extend([
                float(signal_vector @ get_ith_asymmetric_position_matrix(real_ta, imag_ta, i) @ return_vector),
                get_ith_PAPs_expected_return(filtered_vals_ta, i)
            ])
            for kind in ['realized', 'expected']:
                col = f"{kind}_return_of_{i+1}_PAP"
                if col not in realized_returns_df.columns:
                    realized_returns_df[col] = None

        # Add this date’s row to the DataFrame
        realized_returns_df.loc[len(realized_returns_df)] = row_values

    # Finalize and index by date
    realized_returns_df.set_index("date", inplace=True)

    # Scale PAP to match PEP volatility and form combined series
    pap_std = realized_returns_df['realized_return_of_first_n_PAP'].std()
    pep_std = realized_returns_df['realized_return_of_first_n_PEP'].std()
    realized_returns_df['adjusted_PAP'] = (
        realized_returns_df['realized_return_of_first_n_PAP'] * (pep_std / pap_std)
    )
    realized_returns_df['PEP and PAP 1-n'] = (
        realized_returns_df['adjusted_PAP'] + realized_returns_df['realized_return_of_first_n_PEP']
    ) / 2
    realized_returns_df.drop(columns='adjusted_PAP', inplace=True)
    realized_returns_df['long-short PEP and PAP 1-n'] = 0.5 * (
        realized_returns_df['realized_return_of_first_n_PAP'] +
        realized_returns_df['long_short_realized_PEP']
    )

    # Compute annualized Sharpe ratios for all realized series
    sharpe_df = realized_returns_df.drop(
        realized_returns_df.filter(like="expected").columns,
        axis=1
    ).apply(lambda col: calculate_sharpe_ratio(col)) * math.sqrt(12)

    # Average realized vs. expected returns across PP, PEP, and PAP groups
    pp_cols = realized_returns_df.filter(like="PP")
    pep_cols = realized_returns_df.filter(like="PEP")
    pap_cols = realized_returns_df.filter(like="PAP")

    pp_realized_mean_df = pp_cols.filter(like="realized").mean(axis=0)
    pp_expected_mean_df = pp_cols.filter(like="expected").mean(axis=0)
    pep_realized_mean_df = pep_cols.filter(like="realized").mean(axis=0)
    pep_expected_mean_df = pep_cols.filter(like="expected").mean(axis=0)
    pap_realized_mean_df = pap_cols.filter(like="realized").mean(axis=0)
    pap_expected_mean_df = pap_cols.filter(like="expected").mean(axis=0)

    # Assemble core outputs
    output_dict = {
        'realized_returns_df': realized_returns_df,
        'sharpe_df': sharpe_df,
        'pp_realized_mean_df': pp_realized_mean_df,
        'pp_expected_mean_df': pp_expected_mean_df,
        'pep_realized_mean_df': pep_realized_mean_df,
        'pep_expected_mean_df': pep_expected_mean_df,
        'pap_realized_mean_df': pap_realized_mean_df,
        'pap_expected_mean_df': pap_expected_mean_df
    }

    # If factor data provided, run regressions against Fama–French factors
    if factor_data_monthly is not None:
        factor_data_monthly = factor_data_monthly.set_index("date")
        realized_returns_df, factor_data_monthly = filter_dataframes_by_common_dates(
            realized_returns_df, factor_data_monthly
        )
        X = factor_data_monthly[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        Y = realized_returns_df['return_of_simple_factor']
        output_dict["regression_result_return_of_simple_factor"] = regression_results(X, Y)

        # Add simple factor return as an additional regressor for PP/PEP/PAP
        X = pd.concat([X, realized_returns_df['return_of_simple_factor']], axis=1)
        for col in [
            'realized_return_of_first_n_PP',
            'realized_return_of_first_n_PEP',
            'realized_return_of_first_n_PAP',
            'PEP and PAP 1-n'
        ]:
            output_dict[f'regression_result_{col}'] = regression_results(X, realized_returns_df[col])

    return output_dict

def singular_values_vs_realized_returns_graph(output_dict, portfolios_key, number_of_portfolios, title):
    """
    Plot singular values, symmetric eigenvalues, antisymmetric eigenvalues,
    and corresponding average realized returns for Principal Portfolios (PP),
    Principal Exposure (PEP), and Principal Alpha Portfolios (PAP).

    Parameters
    ----------
    output_dict : dict
        Dictionary containing results under `portfolios_key`, including:
        - pp_expected_mean_df
        - pep_expected_mean_df
        - pap_expected_mean_df
        - pp_realized_mean_df
        - pep_realized_mean_df
        - pap_realized_mean_df
    portfolios_key : str
        Key in `output_dict` pointing to the nested dict of DataFrames.
    number_of_portfolios : int
        Total number of portfolios used to determine the x-axis range.
    title : str
        Main title for the figure.
    """
    # X-axis indices from 1 up to (but not including) number_of_portfolios
    x = np.arange(1, number_of_portfolios)

    # Drop the trivial first element (index 0) when slicing expected values
    singular_values = output_dict[portfolios_key]["pp_expected_mean_df"][1:].values
    eigenvalues_symmetric = output_dict[portfolios_key]["pep_expected_mean_df"][1:].values
    eigenvalues_antisymmetric = output_dict[portfolios_key]["pap_expected_mean_df"][1:].values

    # Corresponding realized returns for PP (skip index 0)
    pp_returns = output_dict[portfolios_key]["pp_realized_mean_df"][1:].values
    # For PEP, skip the first three entries (first-n, last-n, and long-short PEP)
    pep_returns = output_dict[portfolios_key]["pep_realized_mean_df"][3:].values
    # PAP realized returns (skip index 0)
    pap_returns = output_dict[portfolios_key]["pap_realized_mean_df"][1:].values

    # Turn on gridlines for all axes
    plt.rcParams['axes.grid'] = True

    # Create a 3×2 grid of subplots for the six panels
    fig, axs = plt.subplots(3, 2, figsize=(8, 12))

    # Panel A: Singular values Π
    axs[0, 0].plot(x, singular_values, 'k.-')
    axs[0, 0].set_title('Panel A. Π Singular Values')
    axs[0, 0].set_xlabel('Eigenvalue Number')
    axs[0, 0].set_ylabel('Singular Value')

    # Panel B: Symmetric eigenvalues Π^s
    axs[1, 0].plot(x, eigenvalues_symmetric, 'k.-')
    axs[1, 0].set_title('Panel B. Π^s Eigenvalues')
    axs[1, 0].set_xlabel('Eigenvalue Number')
    axs[1, 0].set_ylabel('Eigenvalue (λ)')

    # Panel C: Antisymmetric eigenvalues Π^a
    # Note: there are only half as many nonzero antisymmetric eigenvalues
    axs[2, 0].plot(np.arange(1, int(number_of_portfolios / 2)), eigenvalues_antisymmetric, 'k.-')
    axs[2, 0].set_title('Panel C. Π^a Eigenvalues')
    axs[2, 0].set_xlabel('Eigenvalue Number')
    axs[2, 0].set_ylabel('Eigenvalue (λ)')

    # Panel D: Average realized returns of PPs
    axs[0, 1].plot(x, pp_returns, 'k.-')
    axs[0, 1].set_title('Panel D. PP Average Returns')
    axs[0, 1].set_xlabel('Eigenvalue Number')
    axs[0, 1].set_ylabel('PP Returns (%)')

    # Panel E: Average realized returns of PEPs
    axs[1, 1].plot(x, pep_returns, 'k.-')
    axs[1, 1].set_title('Panel E. PEP Average Returns')
    axs[1, 1].set_xlabel('Eigenvalue Number')
    axs[1, 1].set_ylabel('PEP Returns (%)')

    # Panel F: Average realized returns of PAPs
    axs[2, 1].plot(np.arange(1, int(number_of_portfolios / 2)), pap_returns, 'k.-')
    axs[2, 1].set_title('Panel F. PAP Average Returns')
    axs[2, 1].set_xlabel('Eigenvalue Number')
    axs[2, 1].set_ylabel('PAP Returns (%)')

    # Set a uniform y‐axis major tick interval for clarity
    for ax in axs.flat:
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Add the overall title and tighten layout
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    