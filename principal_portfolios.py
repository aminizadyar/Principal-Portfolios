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
    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    # Exclude the 'date' column for ranking
    data_columns = df_copy.columns[1:]
    
    # Apply ranking and scaling row-wise (for each date)
    def rank_row(row):
        # Get the ranks (min rank is 1)
        ranks = row.rank(method='min')
        # Normalize the ranks to range between 0 and 1
        ranks_normalized = (ranks - 1) / (len(row) - 1)
        # Map to range [-0.5, 0.5]
        return ranks_normalized - 0.5
    
    # Apply rank_row function to each row, excluding the 'date' column
    df_copy[data_columns] = df_copy[data_columns].apply(rank_row, axis=1)
    return df_copy


def cross_sectional_demean(df):
    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    # Exclude the 'date' column
    data_columns = df_copy.columns[1:]
    
    # Apply demeaning row-wise (for each date)
    def demean_row(row):
        row_mean = row.mean()  # Compute the mean of the row
        return row - row_mean  # Subtract the mean from each element in the row
    
    # Apply demean_row function to each row, excluding the 'date' column
    df_copy[data_columns] = df_copy[data_columns].apply(demean_row, axis=1)
    return df_copy


def compute_rs_product(df1, df2):
    # Ensure the date columns match
    if not df1['date'].equals(df2['date']):
        raise ValueError("Date columns of both dataframes must match.")
    
  # Convert to numeric, set invalid values as NaN
    df1 = df1.astype({col: 'float64' for col in df1.columns if col != 'date'})
    df2 = df2.astype({col: 'float64' for col in df2.columns if col != 'date'})
    result = {}
    
    # Iterate over each row (each date)
    for index, date in enumerate(df1['date']):
        # Get the R vector (from df1) and S' vector (from df2) for the current date
        R = df1.iloc[index, 1:].values.reshape(-1, 1)  # n x 1 vector
        S_transpose = df2.iloc[index, 1:].values.reshape(1, -1)  # 1 x n vector
        # Compute the outer product (RS')
        matrix_rs = np.dot(R, S_transpose)  # n x n matrix
        # Store the result in a dictionary, with date as the key
        result[date] = matrix_rs

    return result


def get_prediction_matrix(input_date, result_matrices, n_periods):
    # Sort the dates in result_matrices to ensure they're in order
    sorted_dates = sorted(result_matrices.keys())
    # Find the index of the input date in the sorted list of dates
    if input_date not in sorted_dates:
        raise ValueError("The input date is not found in the result_matrices.")
    
    input_date_index = sorted_dates.index(input_date)
    # Select the last n_periods (excluding the input date)
    start_index = max(0, input_date_index - n_periods)  # Ensure we don't go below index 0
    selected_dates = sorted_dates[start_index:input_date_index]
    
    if len(selected_dates) == 0:
        raise ValueError(f"There are no previous periods to calculate the average for the given number: {n_periods}.")
    
    # Initialize a matrix to accumulate the sum
    matrix_shape = result_matrices[sorted_dates[0]].shape
    sum_matrix = np.zeros(matrix_shape, dtype=float)
    # Sum all the selected matrices
    for date in selected_dates:
        sum_matrix += np.array(result_matrices[date], dtype=float)
    
    # Calculate the element-wise average
    average_matrix = sum_matrix / len(selected_dates)
    return average_matrix


# i should start from 0. In other words, to get the first PP's expected return you must set i=0.
def get_ith_PPs_expected_return(S,i):
    return S[i]

# i should start from 0. In other words, to get the first PP you must set i=0.
def get_ith_position_matrix(U,VT,i):
    u_column = U[:, i]
    v_column = VT[i, :]
    return np.outer(v_column,u_column)

def first_n_PPs_expected_return(S,n):
    sum = 0
    for i in range(n):
        sum += get_ith_PPs_expected_return(S,i)
    return sum

def first_n_PPs_position_matrix(U,VT,number_of_PPs):
    matrix_shape = U.shape
    sum_matrix = np.zeros(matrix_shape, dtype=float)
    for i in range(number_of_PPs):
        sum_matrix += get_ith_position_matrix(U,VT,i)
    return sum_matrix/number_of_PPs


# i should start from 0. In other words, to get the first PEP you must set i=0.
def get_ith_PEPs_expected_return(eigenvalues,i):
    return eigenvalues[i]

def get_ith_symmetric_position_matrix(eigenvectors,i):
    w = eigenvectors[:, i]
    return np.outer(w,w)

def first_n_PEPs_expected_return(eigenvalues,n):
    sum = 0
    for i in range(n):
        sum += abs(get_ith_PEPs_expected_return(eigenvalues,i))
    return sum

def first_n_PEPs_position_matrix(eigenvectors,number_of_PEPs):
    matrix_shape = eigenvectors.shape
    sum_matrix = np.zeros(matrix_shape, dtype=float)
    for i in range(number_of_PEPs):
        sum_matrix += get_ith_symmetric_position_matrix(eigenvectors,i)
    return sum_matrix/number_of_PEPs

# i should start from 0. In other words, to get the first PEP you must set i=0.
def get_ith_PAPs_expected_return(filtered_eigenvalues_ta,i):
    return 2 * filtered_eigenvalues_ta[i]

def get_ith_asymmetric_position_matrix(sorted_eigenvectors_ta_real_part,sorted_eigenvectors_ta_imaginary_part,i):
    return np.outer(sorted_eigenvectors_ta_real_part[:,i],sorted_eigenvectors_ta_imaginary_part[:,i]) - np.outer(sorted_eigenvectors_ta_imaginary_part[:,i],sorted_eigenvectors_ta_real_part[:,i])
    
def first_n_PAPs_expected_return(filtered_eigenvalues_ta,n):
    sum = 0
    for i in range(n):
        sum += get_ith_PAPs_expected_return(filtered_eigenvalues_ta,i)
    return sum

def first_n_PAPs_position_matrix(sorted_eigenvectors_ta_real_part,sorted_eigenvectors_ta_imaginary_part,number_of_PAPs):
    number_of_rows = sorted_eigenvectors_ta_real_part.shape[0]
    sum_matrix = np.zeros((number_of_rows,number_of_rows), dtype=float)
    for i in range(number_of_PAPs):
        sum_matrix += get_ith_asymmetric_position_matrix(sorted_eigenvectors_ta_real_part,sorted_eigenvectors_ta_imaginary_part,i)
    return sum_matrix/number_of_PAPs

def calculate_sharpe_ratio(returns):
    # Compute excess returns
    
    # Compute average excess return
    average_return = returns.mean()
    
    # Compute standard deviation of returns
    std_dev_returns = returns.std()
    
    # Compute Sharpe Ratio
    sharpe_ratio = average_return / std_dev_returns
    
    return sharpe_ratio

def filter_dataframes_by_common_dates(df1, df2, is_date_index=True):
    # Find common dates (intersection of index values)
    if is_date_index:
        common_dates = df1.index.intersection(df2.index)
        
        # Filter both dataframes to keep only the rows with the common dates
        df1_filtered = df1.loc[common_dates]
        df2_filtered = df2.loc[common_dates]
    
    else:
        common_dates = set(df1['date']).intersection(set(df2['date']))

        # Filter both DataFrames
        df1_filtered = df1[df1['date'].isin(common_dates)]
        df2_filtered = df2[df2['date'].isin(common_dates)]
    
    return df1_filtered, df2_filtered


def regression_results(X, Y):
    # Get the standard deviation of X[1] (which is the first non-constant column of X)
    std_X1 = X.iloc[:, 1].std()  # Assuming X[1] is the first non-constant column

    # Scale Y to have the same standard deviation as X[1]
    std_Y = Y.std()
    Y_scaled = Y * (std_X1 / std_Y)  # Scale Y by the ratio of std_X1 to std_Y

    # Add a constant (intercept) to the independent variables
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(Y_scaled, X).fit()

    # Get the coefficients (including the intercept)
    coefficients = model.params.values


    # Get the t-statistics
    t_stats = model.tvalues.values

    # Calculate the standard deviation of residuals
    residuals = model.resid
    std_residuals = residuals.std(ddof=1)  # Sample standard deviation (ddof=1)

    # Calculate intercept divided by standard deviation of residuals
    intercept_over_std_residuals = (coefficients[0] / std_residuals) * math.sqrt(12) #sqrt(12) is to make the IR annualized. again, to make comparison with Kelly's paper easier.

    # Get the R-squared value
    r_squared = model.rsquared

    coefficients[0] = coefficients[0] * 12 # This is to make the results comparable to Kelly(2023). They report annualized alphas in their tables.

    # Get the names of the independent variables (including the constant)
    variable_names = X.columns.tolist()

    # Return the list as required
    return [coefficients, t_stats, intercept_over_std_residuals, r_squared, variable_names]



def build_PP(input_return_dataset_df, signal_df, number_of_lookback_periods,
            starting_year_to_filter, end_year_to_filter, portfolio_formation_df = None,
            factor_data_monthly=None, number_of_PPs_to_consider = 3,
            number_of_PEPs_to_consider = 3, number_of_PAPs_to_consider = 3,
            use_demeaned_returns = True):
    
    '''
    # Here input_return_dataset_df, signal_df, portfolio_formation_df and factor_data_monthly have a column named 'date'.
    # Other columns of these dataframes are percentage returns.
    # input_return_dataset_df, signal_df and portfolio_formation_df must have the same columns.
    # input_return_dataset_df is used to calculate the prediction matrix. It can be excess returns or simple returns.
    # portfolio_formation_df is used for portfolio formation.
    # One can use excess returns to get portfolio weights, but use simple returns to measure performance.
    # This is why these two variables are seperated. If there is not input for portfolio_formation_df, the code assumes that it is the same as input_return_dataset_df.
    '''
    # If portfolio_formation_df is not provided, use input_return_dataset_df
    if portfolio_formation_df is None:
        portfolio_formation_df = input_return_dataset_df

    input_return_dataset_df,signal_df = filter_dataframes_by_common_dates(input_return_dataset_df.dropna(), signal_df.dropna(), is_date_index=False)

    # I can think of this matrix as $S_{t-1}$.
    normalized_signal_df = rank_and_map(signal_df)
    normalized_signal_df = normalized_signal_df[(normalized_signal_df['date'].dt.year > starting_year_to_filter) & (normalized_signal_df['date'].dt.year < end_year_to_filter)].reset_index(drop=True)
    
    input_return_dataset_df = input_return_dataset_df[(input_return_dataset_df['date'].dt.year > starting_year_to_filter) & (input_return_dataset_df['date'].dt.year < end_year_to_filter)].reset_index(drop=True)
    # This matrix can be denoted as $R_{t-1}$
    if use_demeaned_returns == True:
        return_matrix_df = cross_sectional_demean(input_return_dataset_df)
    else:
        return_matrix_df = input_return_dataset_df.copy()
    
    # This gives: $R_{t}S'_{t}$
    rs_matrix = compute_rs_product(return_matrix_df, normalized_signal_df)
    
    """
    Prediction matrix for date T+1, used returns data up to month T and signals data up to month T-1.
    In the function get_prediction_matrix, I start the calculations from the previous month. 
    Note that although the input date is the current data, but the in the function that month is excluded.
    Note that in calculating realized returns, I am using the current month(the month of rearlized returns) as index. 
    But remember that the matrix was $S_{t-1}$. So, the index actually retreives the value of the previous month. 
    I formed the matrix this way in order to make the calculations easier.
    """

    realized_returns_df = pd.DataFrame(columns=[
        "date",
        "return_of_simple_factor", 
        "realized_return_of_first_n_PP", 
        "expected_return_of_first_n_PP",
        "realized_return_of_first_n_PEP",
        "expected_return_of_first_n_PEP",
        "realized_return_of_first_n_PAP",
        "expected_return_of_first_n_PAP"
    ])

    # I leave out the first 120 (number of lookback periods) observations to compute the prediction matrix.
    for date_index in return_matrix_df.iloc[number_of_lookback_periods:]['date']:
        date_to_consider = pd.Timestamp(date_index)
        
        #for PP's
        prediction_matrix = get_prediction_matrix(date_to_consider, rs_matrix, number_of_lookback_periods)
        U, S, VT = np.linalg.svd(prediction_matrix)

        #for PEP's
        Symmetric_prediction_matrix = (prediction_matrix + prediction_matrix.T)/2
        eigenvalues, eigenvectors = np.linalg.eig(Symmetric_prediction_matrix)
        idx = eigenvalues.argsort()[::-1]  # Sort in descending order
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # for PAP'S
        assymetric_prediction_matrix = 0.5 * (prediction_matrix - prediction_matrix.T)
        transposed_assymetric_prediction_matrix = assymetric_prediction_matrix.T
        eigenvalues_ta, eigenvectors_ta = np.linalg.eig(transposed_assymetric_prediction_matrix)
        sorted_indices_ta = np.argsort(-eigenvalues_ta.imag)
        sorted_eigenvalues_ta = eigenvalues_ta[sorted_indices_ta].imag
        sorted_eigenvectors_ta = eigenvectors_ta[:, sorted_indices_ta] * math.sqrt(2)  #sqrt(2) is to make the size of the vectors equal to 1.
        positive_indices = np.where(sorted_eigenvalues_ta > 0)
        filtered_eigenvalues_ta = sorted_eigenvalues_ta[positive_indices]
        filtered_eigenvectors_ta = sorted_eigenvectors_ta[:, positive_indices].squeeze()
        sorted_eigenvectors_ta_imaginary_part = filtered_eigenvectors_ta.imag
        sorted_eigenvectors_ta_real_part = filtered_eigenvectors_ta.real

        #to calculate realized returns
        signal_vector = normalized_signal_df[normalized_signal_df.date == date_to_consider].values[0, 1:].reshape(1, -1)  # 1*n matrix
        '''
        There is no difference between using demeaned returns or not_demeaned ones.
        I can replace input_return_dataset_df with return_matrix_df.
        This is because I am forming long-short portfolios.
        The size of the long position is the same as the short position. So, the mean will cancell out.
        '''
        return_vector = portfolio_formation_df[portfolio_formation_df.date == date_to_consider].values[0, 1:].reshape(-1, 1)  # n*1 matrix

        # Compute realized returns
        return_of_simple_factor = (signal_vector @ return_vector)[0][0]
        realized_return_of_first_n_PP = (signal_vector @ first_n_PPs_position_matrix(U, VT, number_of_PPs_to_consider) @ return_vector)[0][0]
        expected_return_of_first_n_PP = first_n_PPs_expected_return(S, number_of_PPs_to_consider)
        realized_return_of_first_n_PEP = (signal_vector @ first_n_PEPs_position_matrix(eigenvectors,number_of_PEPs_to_consider) @ return_vector)[0][0]
        expected_return_of_first_n_PEP = first_n_PEPs_expected_return(eigenvalues, number_of_PEPs_to_consider)
        realized_return_of_first_n_PAP = (signal_vector @ first_n_PAPs_position_matrix(sorted_eigenvectors_ta_real_part,sorted_eigenvectors_ta_imaginary_part,number_of_PAPs_to_consider) @ return_vector)[0][0]
        expected_return_of_first_n_PAP = first_n_PAPs_expected_return(filtered_eigenvalues_ta, number_of_PAPs_to_consider)
        

        # Prepare a list for the current row values
        row_values = [
            date_index,
            return_of_simple_factor,  
            realized_return_of_first_n_PP, 
            expected_return_of_first_n_PP,
            realized_return_of_first_n_PEP,
            expected_return_of_first_n_PEP,
            realized_return_of_first_n_PAP,
            expected_return_of_first_n_PAP
        ]

        # Iterate over all Principal Portfolios (up to len(S)) and calculate realized/expected returns for each
        for i in range(len(S)):

            # for PP's
            realized_return_ith_PP = (signal_vector @ get_ith_position_matrix(U, VT, i) @ return_vector)[0][0]
            expected_return_ith_PP = get_ith_PPs_expected_return(S, i)
            # Add the values for realized and expected returns of the ith PP to the row
            row_values.append(realized_return_ith_PP)
            row_values.append(expected_return_ith_PP)

            # for PEP's
            realized_return_ith_PEP = (signal_vector @ get_ith_symmetric_position_matrix(eigenvectors, i) @ return_vector)[0][0]
            expected_return_ith_PEP = get_ith_PEPs_expected_return(eigenvalues, i)
            # Add the values for realized and expected returns of the ith PEP to the row
            row_values.append(realized_return_ith_PEP)
            row_values.append(expected_return_ith_PEP)


            # Dynamically add columns if they don't exist. for PP's.
            realized_col_name_pp = f"realized_return_of_{i+1}_PP"
            expected_col_name_pp = f"expected_return_of_{i+1}_PP"

            # Dynamically add columns if they don't exist. for PEP's.
            realized_col_name_pep = f"realized_return_of_{i+1}_PEP"
            expected_col_name_pep = f"expected_return_of_{i+1}_PEP"
            
            # for PP's
            if realized_col_name_pp not in realized_returns_df.columns:
                realized_returns_df[realized_col_name_pp] = None
            if expected_col_name_pp not in realized_returns_df.columns:
                realized_returns_df[expected_col_name_pp] = None

            #for PEP'S
            if realized_col_name_pep not in realized_returns_df.columns:
                realized_returns_df[realized_col_name_pep] = None
            if expected_col_name_pep not in realized_returns_df.columns:
                realized_returns_df[expected_col_name_pep] = None
                
        # The number of PAP's is different, so another loop is needed.
        for i in range(sorted_eigenvectors_ta_imaginary_part.shape[1]):
            # for PAP's
            realized_return_ith_PAP = (signal_vector @ get_ith_asymmetric_position_matrix(sorted_eigenvectors_ta_real_part,sorted_eigenvectors_ta_imaginary_part,i) @ return_vector)[0][0]
            expected_return_ith_PAP = get_ith_PAPs_expected_return(filtered_eigenvalues_ta,i)
            # Add the values for realized and expected returns of the ith PEP to the row
            row_values.append(realized_return_ith_PAP)
            row_values.append(expected_return_ith_PAP)

            # Dynamically add columns if they don't exist. for PEP's.
            realized_col_name_pap = f"realized_return_of_{i+1}_PAP"
            expected_col_name_pap = f"expected_return_of_{i+1}_PAP"

            #for PAP'S
            if realized_col_name_pap not in realized_returns_df.columns:
                realized_returns_df[realized_col_name_pap] = None
            if expected_col_name_pap not in realized_returns_df.columns:
                realized_returns_df[expected_col_name_pap] = None

        # Append the row to the dataframe
        realized_returns_df.loc[len(realized_returns_df)] = row_values

    realized_returns_df = realized_returns_df.set_index("date")

    pap_std = realized_returns_df['realized_return_of_first_n_PAP'].std()
    pep_std = realized_returns_df['realized_return_of_first_n_PEP'].std()
    realized_returns_df['adjusted_PAP'] = realized_returns_df['realized_return_of_first_n_PAP'] * (pep_std / pap_std)
    # Take the average of the adjusted "PAP" and "PEP"
    realized_returns_df['PEP and PAP 1-n'] = (realized_returns_df['adjusted_PAP'] + realized_returns_df['realized_return_of_first_n_PEP']) / 2
    # drop the adjusted column.
    realized_returns_df.drop(columns='adjusted_PAP', inplace=True)

    sharpe_df = realized_returns_df.drop(realized_returns_df.filter(like="expected").columns, axis=1).apply(lambda col: calculate_sharpe_ratio(col)) * math.sqrt(12)

    pp_columns = realized_returns_df.filter(like="PP")
    pp_realized_mean_df = pp_columns.filter(like="realized").mean(axis=0)
    pp_expected_mean_df = pp_columns.filter(like="expected").mean(axis=0)

    pep_columns = realized_returns_df.filter(like="PEP")
    pep_realized_mean_df = pep_columns.filter(like="realized").mean(axis=0)
    pep_expected_mean_df = pep_columns.filter(like="expected").mean(axis=0)

    pap_columns = realized_returns_df.filter(like="PAP")
    pap_realized_mean_df = pap_columns.filter(like="realized").mean(axis=0)
    pap_expected_mean_df = pap_columns.filter(like="expected").mean(axis=0)

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

    if factor_data_monthly is not None:
        factor_data_monthly = factor_data_monthly.set_index("date")
        realized_returns_df, factor_data_monthly = filter_dataframes_by_common_dates(realized_returns_df,factor_data_monthly)
        X = factor_data_monthly[['Mkt-RF', 'SMB','HML','RMW','CMA']]
        Y = realized_returns_df['return_of_simple_factor']
        output_dict["regression_result_return_of_simple_factor"] = regression_results(X,Y)

        # Following Kelly's paper. For regressions that have PP's as Y, simple factor must be addes as X.
        X = pd.concat([X, realized_returns_df['return_of_simple_factor']], axis=1)
        # List of column names in realized_returns_df for which I want to perform the regression
        columns_to_iterate = ['realized_return_of_first_n_PP',
                        'realized_return_of_first_n_PEP',
                        'realized_return_of_first_n_PAP',
                        'PEP and PAP 1-n'] 

        # Iterate over each column
        for col in columns_to_iterate:
            # Set Y to the column from realized_returns_df
            Y = realized_returns_df[col]
            # Store the results with a variable name based on the column name
            result_name = f'regression_result_{col}'
            output_dict[result_name] = regression_results(X,Y)
    return output_dict  


def singular_values_vs_realized_returns_graph(output_dict,portfolios_key,number_of_portfolios,title):

    x = np.arange(1, number_of_portfolios)
    singular_values = output_dict[portfolios_key]["pp_expected_mean_df"][1:].values
    eigenvalues_symmetric = output_dict[portfolios_key]["pep_expected_mean_df"][1:].values
    eigenvalues_antisymmetric = output_dict[portfolios_key]["pap_expected_mean_df"][1:].values
    pp_returns = output_dict[portfolios_key]["pp_realized_mean_df"][1:].values
    pep_returns = output_dict[portfolios_key]["pep_realized_mean_df"][1:].values
    pap_returns = output_dict[portfolios_key]["pap_realized_mean_df"][1:].values

    plt.rcParams['axes.grid'] = True
    # Create 3x2 subplots
    fig, axs = plt.subplots(3, 2, figsize=(8, 12))  # 3 rows, 2 columns

    # Panel A: Singular values
    axs[0, 0].plot(x, singular_values, 'k.-')
    axs[0, 0].set_title('Panel A. Π Singular Values')
    axs[0, 0].set_xlabel('Eigenvalue Number')
    axs[0, 0].set_ylabel('Singular Value')

    # Panel B: Symmetric eigenvalues
    axs[1, 0].plot(x, eigenvalues_symmetric, 'k.-')
    axs[1, 0].set_title('Panel B. Π^s Eigenvalues')
    axs[1, 0].set_xlabel('Eigenvalue Number')
    axs[1, 0].set_ylabel('Eigenvalue (λ)')

    # Panel C: Antisymmetric eigenvalues
    axs[2, 0].plot(np.arange(1, int(number_of_portfolios/2)), eigenvalues_antisymmetric, 'k.-')
    axs[2, 0].set_title('Panel C. Π^a Eigenvalues')
    axs[2, 0].set_xlabel('Eigenvalue Number')
    axs[2, 0].set_ylabel('Eigenvalue (λ)')

    # Panel D: PP Average Returns
    axs[0, 1].plot(x, pp_returns, 'k.-')
    axs[0, 1].set_title('Panel D. PP Average Returns')
    axs[0, 1].set_xlabel('Eigenvalue Number')
    axs[0, 1].set_ylabel('PP Returns (%)')

    # Panel E: PEP Average Returns
    axs[1, 1].plot(x, pep_returns, 'k.-')
    axs[1, 1].set_title('Panel E. PEP Average Returns')
    axs[1, 1].set_xlabel('Eigenvalue Number')
    axs[1, 1].set_ylabel('PEP Returns (%)')

    # Panel F: PAP Average Returns
    axs[2, 1].plot(np.arange(1, int(number_of_portfolios/2)), pap_returns, 'k.-')
    axs[2, 1].set_title('Panel F. PAP Average Returns')
    axs[2, 1].set_xlabel('Eigenvalue Number')
    axs[2, 1].set_ylabel('PAP Returns (%)')

    for ax in axs.flat:
        ax.yaxis.set_major_locator(MultipleLocator(0.1)) 

    # Add a title for the entire figure
    fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.show()


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