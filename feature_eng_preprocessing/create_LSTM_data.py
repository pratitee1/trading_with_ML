# For a given test date, query previous data for the validation and training set keeping an OFFSET of 3 days to avoid data leakage.
# Create labels for the data considering future 3-day returns. If a stock has a higher future 3-day return than the future 3-day crossectional median of all stocks, 
#     label it as 1 (out-performing stock), else label it as 0 (under-performing stock).
# Create 15-day sequences of technical features as input and the label of the last day as output for the training and validation set. Store it as .npz files
# For the test set, label the data similarly and store the dataframe. The LSTM sequences for the test set are created later, directly during testing.
# (FUTURE WORK: Include the extracted fundamental indicators as features or directly in the trading strategy.)

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from sklearn.preprocessing import StandardScaler
import pickle
import os

features_list = ['vol_adj_rtn_12m', 'd130_min_rtn', 'stochastic_k', 'stochastic_d', 'stochastic_k_20', 'stochastic_d_20',
       'adj_sto_6m_k', 'adj_sto_6m_d', 'std_dev_20', 'stderr_180d', 'ann_vol_1m', 'ann_vol_12m', 'vol_signal_5d', 'vol_signal_50d',
       'm24_res_rtn_var', 'd90_cv', 'd60_cv', 'vol_style', 'pm_1m', 'pm_5d', 'pm_6m', 'pm_9m', 'pm_12m', 'pm_12m1m', 'prc_to_260dl', 
        'prc_to_52wh', 'pratio_15_36w', 'ratio_50_to_200', 'pm_style', 'pa52wl20dlag', 'w39_rtn_lag4w', 'osc_4to52w_prc', 'hl_1m', 
        'hl_52w', 'log_unadj_price', 'rsi_14', 'rsi_26w', 'd10_macd', 'd10_macd_signal', 'slope_52w', 'pslope_serr_26w', 'sma_10',
        'sma_50', 'sma_200', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower', 'adx', 'd5_money_flow', 'd5_money_flow_vol', 
        'amihud', 'chg_1y_amihud', 'obv', 'beta_60m', 'alpha_60m', 'alpha_18m_6mchg', 'alpha_12m_6mchg', 'alpha_36m_6mchg', 
        'rel_pr_str_12m', 'sharpe_60d']
        
def get_train_val_data(db_path, end_date, days):
    conn = sqlite3.connect(db_path)
    query = f"""
    WITH filtered_dates AS (
        SELECT DISTINCT date 
        FROM technical_indicators 
        WHERE date < '{end_date}'
        ORDER BY date DESC
        LIMIT {days} OFFSET 3
    )
    SELECT * 
    FROM technical_indicators
    WHERE date IN (SELECT date FROM filtered_dates)
    ORDER BY date, ticker;
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    min_query = f"""
    WITH filtered_dates AS (
        SELECT DISTINCT date 
        FROM technical_indicators 
        WHERE date < '{end_date}'
        ORDER BY date DESC
        LIMIT {days} OFFSET 3
    )
    SELECT MIN(date) FROM filtered_dates;
    """
    cursor = conn.cursor()
    cursor.execute(min_query)
    min_date = cursor.fetchone()[0]
    min_date = datetime.strptime(min_date, "%Y-%m-%d %H:%M:%S").date()
    conn.close()
    return df, min_date
    
def get_test_data(db_path, start_date, days):
    #DO: HANDLE LAST YEAR IN DATABASE CORRECTLY
    conn = sqlite3.connect(db_path)
    query = f"""
    WITH previous_dates AS (
        SELECT DISTINCT date 
        FROM technical_indicators 
        WHERE date < '{start_date}'
        ORDER BY date DESC
        LIMIT 14
    ),
    filtered_dates AS (
        SELECT DISTINCT date 
        FROM technical_indicators 
        WHERE date >= '{start_date}'
        ORDER BY date ASC
        LIMIT {days}
    ),
    combined_dates AS (
        SELECT date FROM previous_dates
        UNION
        SELECT date FROM filtered_dates
    )
    SELECT * 
    FROM technical_indicators
    WHERE date IN (SELECT date FROM combined_dates)
    ORDER BY date, ticker;
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    max_query = f"""
    WITH filtered_dates AS (
        SELECT DISTINCT date 
        FROM technical_indicators 
        WHERE date >= '{start_date}'
        ORDER BY date ASC
        LIMIT {days}
    )
    SELECT MAX(date) FROM filtered_dates;
    """
    cursor = conn.cursor()
    cursor.execute(max_query)
    max_date = cursor.fetchone()[0]
    next_date_query = f"""
    SELECT MIN(date) FROM technical_indicators 
    WHERE date > '{max_date}';
    """
    cursor.execute(next_date_query)
    next_test_date = cursor.fetchone()[0]
    max_date = datetime.strptime(max_date, "%Y-%m-%d %H:%M:%S").date()
    next_test_date = datetime.strptime(next_test_date, "%Y-%m-%d %H:%M:%S").date()
    conn.close()
    return df, max_date, next_test_date
    
def create_labels(df):
    df['cross_sectional_median'] = df.groupby('date')['future_3D_return'].transform('median')
    df['label'] = (df['future_3D_return'] > df['cross_sectional_median']).astype(int)
    return df
    
def create_lstm_dataset(df, feature_list, sequence_length=15):
    X, y, metadata = [], [], [] 
    for _, stock_data in df.groupby('ticker', group_keys=False):
        stock_data = stock_data.reset_index(drop=True)
        feature_array = stock_data[feature_list].values
        label_array = stock_data['label'].values
        for i in range(len(stock_data) - sequence_length):
            X.append(feature_array[i:i+sequence_length])
            y.append(label_array[i + sequence_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)

if __name__ == "__main__":
    year = 2022
    data_dir = "../data"
    DB_PATH = os.path.join(data_dir, "tech_ind.db")
    year_dir = os.path.join(data_dir, str(year))  
    os.makedirs(year_dir, exist_ok=True)
    test_start_date = datetime(year, 1, 1).date()
    test_end_date_for_year = datetime(year, 12, 30).date()
    counter = 1
    while (test_start_date < test_end_date_for_year) :
        print(counter, test_start_date)
        val_data, val_start_date = get_train_val_data(DB_PATH, test_start_date, 160)
        train_data, train_start_date = get_train_val_data(DB_PATH, val_start_date, 1020)
        test_data, test_end_date, test_start_date = get_test_data(DB_PATH, test_start_date, 20)
        scaler = StandardScaler()
        train_data[features_list] = scaler.fit_transform(train_data[features_list])
        val_data[features_list] = scaler.transform(val_data[features_list])
        test_data[features_list] = scaler.transform(test_data[features_list])
        train_data_labelled = create_labels(train_data)
        val_data_labelled = create_labels(val_data)
        test_data_labelled = create_labels(test_data)
        X_train, y_train = create_lstm_dataset(train_data_labelled, features_list)
        X_val, y_val = create_lstm_dataset(val_data_labelled, features_list)
        train_val_save_path = os.path.join(year_dir, f"LSTM_train_val_{counter}.npz")
        np.savez(train_val_save_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        test_save_path = os.path.join(year_dir, f"LSTM_test_{counter}.pkl")
        test_data_labelled.to_pickle(test_save_path)
        counter+=1
    print (counter, test_start_date)
 
