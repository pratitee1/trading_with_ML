#Compute technical indicators as listed in https://www.nature.com/articles/s41598-023-50783-0 
#Compute and store ticker-by-ticker to avoid memory issues

import sqlite3
import pandas as pd
import json
import os
import numpy as np
import ta

features_list = ['vol_adj_rtn_12m', 'd130_min_rtn', 'stochastic_k', 'stochastic_d', 'stochastic_k_20', 'stochastic_d_20',
       'adj_sto_6m_k', 'adj_sto_6m_d', 'std_dev_20', 'stderr_180d', 'ann_vol_1m', 'ann_vol_12m', 'vol_signal_5d', 'vol_signal_50d',
       'm24_res_rtn_var', 'd90_cv', 'd60_cv', 'vol_style', 'pm_1m', 'pm_5d', 'pm_6m', 'pm_9m', 'pm_12m', 'pm_12m1m', 'prc_to_260dl', 
        'prc_to_52wh', 'pratio_15_36w', 'ratio_50_to_200', 'pm_style', 'pa52wl20dlag', 'w39_rtn_lag4w', 'osc_4to52w_prc', 'hl_1m', 
        'hl_52w', 'log_unadj_price', 'rsi_14', 'rsi_26w', 'd10_macd', 'd10_macd_signal', 'slope_52w', 'pslope_serr_26w', 'sma_10',
        'sma_50', 'sma_200', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower', 'adx', 'd5_money_flow', 'd5_money_flow_vol', 
        'amihud', 'chg_1y_amihud', 'obv', 'beta_60m', 'alpha_60m', 'alpha_18m_6mchg', 'alpha_12m_6mchg', 'alpha_36m_6mchg', 
        'rel_pr_str_12m', 'sharpe_60d']

SECTOR_ETFS = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC"
}

def create_database(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS technical_indicators (
        ticker TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adj_close REAL,
        volume INTEGER,
        sector TEXT,
        sector_close REAL,
        sp500_close REAL, 
        sp500_return REAL,
        risk_free_rate REAL,
        return REAL, 
        return3D REAL,
        future_3D_return REAL,
        future_3D_market_return REAL,
        return_other_strategy REAL,
        vol_adj_rtn_12m REAL, 
        d130_min_rtn REAL, 
        stochastic_k REAL, 
        stochastic_d REAL,
        stochastic_k_20 REAL, 
        stochastic_d_20 REAL, 
        adj_sto_6m_k REAL, 
        adj_sto_6m_d REAL,
        std_dev_20 REAL, 
        stderr_180d REAL, 
        ann_vol_1m REAL, 
        ann_vol_12m REAL,
        vol_signal_5d REAL, 
        vol_signal_50d REAL, 
        m24_res_rtn_var REAL, 
        d90_cv REAL,
        d60_cv REAL, 
        vol_style REAL, 
        pm_1m REAL, 
        pm_5d REAL, 
        pm_6m REAL, 
        pm_9m REAL, 
        pm_12m REAL,
        pm_12m1m REAL, 
        prc_to_260dl REAL, 
        prc_to_52wh REAL, 
        pratio_15_36w REAL,
        ratio_50_to_200 REAL, 
        max_ret_payoff REAL, 
        pm_style REAL, 
        pa52wl20dlag REAL,
        w39_rtn_lag4w REAL, 
        osc_4to52w_prc REAL, 
        hl_1m REAL, 
        hl_52w REAL, 
        indrel_pm1m REAL,
        indrel_pm5d REAL, 
        indrel_pm6m REAL, 
        indrel_pm9m REAL, 
        indrel_pm12m REAL,
        indrel_pm_12m1m REAL, 
        indrel_vol_50d REAL, 
        indrel_max_ret REAL,
        log_unadj_price REAL, 
        rsi_14 REAL, 
        rsi_26w REAL, 
        d10_macd REAL, 
        d10_macd_signal REAL,
        slope_52w REAL, 
        pslope_serr_26w REAL, 
        sma_10 REAL,
        sma_50 REAL, 
        sma_200 REAL,
        bollinger_upper REAL, 
        bollinger_middle REAL, 
        bollinger_lower REAL, 
        adx REAL,
        d5_money_flow REAL, 
        d5_money_flow_vol REAL, 
        amihud REAL, 
        chg_1y_amihud REAL, 
        obv REAL, 
        beta_60m REAL, 
        alpha_60m REAL, 
        alpha_18m_6mchg REAL,
        alpha_12m_6mchg REAL, 
        alpha_36m_6mchg REAL, 
        rel_pr_str_12m REAL, 
        sharpe_60d REAL,
        PRIMARY KEY (ticker, date)
    )
    """)
    conn.commit()
    conn.close()
    
def load_ticker_data(db_path, ticker):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM daily_data WHERE ticker='{ticker}' ORDER BY date"
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    return df 
    
def load_market_data(db_path):
    conn = sqlite3.connect(db_path)
    market_data = pd.read_sql("SELECT * FROM market_index ORDER BY date", conn, parse_dates=['date'])
    risk_free_data = pd.read_sql("SELECT * FROM risk_free_rate ORDER BY date", conn, parse_dates=['date'])
    conn.close()
    return market_data, risk_free_data

def load_industry_ETF_data(db_path, stock_sector):
    conn = sqlite3.connect(db_path)
    sector_table = f"sector_{stock_sector.replace(' ', '_')}" 
    sector_column = f"{stock_sector}_index" 
    sector_query = f"SELECT Date as date, `{sector_column}` FROM {sector_table} ORDER BY date"
    sector_df = pd.read_sql(sector_query, conn, parse_dates=['date']).rename(columns={sector_column: 'sector_close'})
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    conn.close()
    return sector_df
    
def compute_technical_indicators(df, market_data, risk_free_data, sector_df):
    df = df.sort_values(by='date')
    df['date'] = pd.to_datetime(df['date'])
    market_data['date'] = pd.to_datetime(market_data['date'])
    risk_free_data['date'] = pd.to_datetime(risk_free_data['date'])
    df = df.merge(sector_df, on='date', how='left')
    df = df.merge(market_data, on='date', how='left')
    df = df.merge(risk_free_data, on='date', how='left')
    # Returns
    df['return'] = df['adj_close'].pct_change()
    df['return3D'] = df['adj_close'].pct_change(periods=3)
    df['future_3D_return'] = df['return3D'].shift(-3)
    df['future_3D_market_return'] = (df['sp500_close'].shift(-3) - df['sp500_close']) / df['sp500_close']
    df['return_other_strategy'] = (df['open'].shift(-6) - df['open'].shift(-1)) / df['open'].shift(-1)
    df['vol_adj_rtn_12m'] = df['return'].rolling(252).mean() / df['return'].rolling(252).std()
    df['d130_min_rtn'] = df['return'].rolling(130).min()
    #Stochastic Oscillators
    stochastic = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stochastic_k'] = stochastic.stoch()
    df['stochastic_d'] = stochastic.stoch_signal()
    stochastic_20 = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=20)
    df['stochastic_k_20'] = stochastic_20.stoch()
    df['stochastic_d_20'] = stochastic_20.stoch_signal()
    adj_sto_6m = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=126)
    df['adj_sto_6m_k'] = adj_sto_6m.stoch()
    df['adj_sto_6m_d'] = adj_sto_6m.stoch_signal()
    # Volatility Indicators
    df['std_dev_20'] = df['return'].rolling(20).std()
    df['stderr_180d'] = df['return'].rolling(180).std()
    df['ann_vol_1m'] = df['return'].rolling(21).std() * np.sqrt(252)
    df['ann_vol_12m'] = df['return'].rolling(252).std() * np.sqrt(252)
    df['vol_signal_5d'] = df['return'].rolling(5).std()
    df['vol_signal_50d'] = df['return'].rolling(50).std()
    df['m24_res_rtn_var'] = df['return'].rolling(504).var()
    df['d90_cv'] = df['return'].rolling(90).std() / df['adj_close'].rolling(90).mean()
    df['d60_cv'] = df['return'].rolling(60).std() / df['adj_close'].rolling(60).mean()
    df['vol_style'] = df['return'].rolling(252).std() / df['adj_close'].rolling(252).mean()
    # Price Momentum Indicators
    df['pm_1m'] = df['adj_close'].pct_change(periods=21)
    df['pm_5d'] = df['adj_close'].pct_change(periods=5)
    df['pm_6m'] = df['adj_close'].pct_change(periods=126)
    df['pm_9m'] = df['adj_close'].pct_change(periods=189)
    df['pm_12m'] = df['adj_close'].pct_change(periods=252)
    df['pm_12m1m'] = df['pm_12m'] / df['pm_1m']
    df['prc_to_260dl'] = df['adj_close'] / df['low'].rolling(260).min()
    df['prc_to_52wh'] = df['adj_close'] / df['high'].rolling(252).max()
    df['pratio_15_36w'] = df['adj_close'].shift(105) / df['adj_close'].shift(252)
    df['ratio_50_to_200'] = df['adj_close'].rolling(50).mean() / df['adj_close'].rolling(200).mean()
    df['max_ret_payoff'] = df['return'].rolling(30).max() / df['sector_close'].pct_change(periods=30).rolling(30).max()
    df['pm_style'] = df['return'].rolling(252).mean() / df['return'].rolling(252).std()  # Annualized Sharpe ratio approximation
    df['pa52wl20dlag'] = df['adj_close'].shift(20) / df['adj_close'].rolling(252).min()
    df['w39_rtn_lag4w'] = df['adj_close'].pct_change(periods=195).shift(21)
    df['osc_4to52w_prc'] = df['adj_close'].rolling(20).mean() / df['adj_close'].rolling(252).mean()
    df['hl_1m'] = df['high'].rolling(21).max() - df['low'].rolling(21).min()
    df['hl_52w'] = df['high'].rolling(252).max() - df['low'].rolling(252).min()
    df['rsi_14'] = ta.momentum.RSIIndicator(df['adj_close'], window=14).rsi()
    df['rsi_26w'] = ta.momentum.RSIIndicator(df['adj_close'], window=182).rsi()
    macd_10 = ta.trend.MACD(df['adj_close'], window_slow=26, window_fast=10, window_sign=9)
    df['d10_macd'] = macd_10.macd()
    df['d10_macd_signal'] = macd_10.macd_signal()
    df['slope_52w'] = df['return'].rolling(252).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    df['pslope_serr_26w'] = df['return'].rolling(126).apply(lambda x: np.polyfit(range(len(x)), x, 1)[1])
    # Industry-relative indicators
    df['indrel_pm1m'] = df['pm_1m'] / df['sector_close'].pct_change(periods=21)
    df['indrel_pm5d'] = df['pm_5d'] / df['sector_close'].pct_change(periods=5)
    df['indrel_pm6m'] = df['pm_6m'] / df['sector_close'].pct_change(periods=126)
    df['indrel_pm9m'] = df['pm_9m'] / df['sector_close'].pct_change(periods=189)
    df['indrel_pm12m'] = df['pm_12m'] / df['sector_close'].pct_change(periods=252)
    df['indrel_pm_12m1m'] = df['indrel_pm12m'] / df['indrel_pm1m']
    df['indrel_vol_50d'] = df['adj_close'].rolling(50).std() / df['sector_close'].rolling(50).std()
    df['indrel_max_ret'] = df['return'].rolling(30).max() / df['sector_close'].pct_change(periods=30).rolling(30).max()
    df['log_unadj_price'] = np.log(df['close'])
    # Extra indicators
    df['sma_10'] = ta.trend.SMAIndicator(df['adj_close'], window=10).sma_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(df['adj_close'], window=50).sma_indicator()
    df['sma_200'] = ta.trend.SMAIndicator(df['adj_close'], window=200).sma_indicator()
    bollinger = ta.volatility.BollingerBands(df['adj_close'])
    df['bollinger_upper'] = bollinger.bollinger_hband()
    df['bollinger_middle'] = bollinger.bollinger_mavg()
    df['bollinger_lower'] = bollinger.bollinger_lband()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    df['d5_money_flow'] = df['close'] * df['volume']  
    df['d5_money_flow_vol'] = df['d5_money_flow'].rolling(5).std()
    df['amihud'] = df['return'].abs() / df['volume'] 
    df['chg_1y_amihud'] = df['amihud'].pct_change(periods=252, fill_method=None)
    df['amihud'] = df['return'].abs() / df['volume']
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['adj_close'], df['volume']).on_balance_volume()
    # Risk-Adjusted Indicators
    df['cov_stock_market'] = df['return'].rolling(1260).cov(df['sp500_return'])  # 1260 trading days ~ 60 months
    df['market_variance'] = df['sp500_return'].rolling(1260).var()
    df['beta_60m'] = df['cov_stock_market'] / df['market_variance']
    df['risk_free_rate'] = df['risk_free_rate'].ffill()
    df['excess_return'] = df['return'] - df['risk_free_rate']
    df['market_excess_return'] = df['sp500_return'] - df['risk_free_rate']
    df['alpha_60m'] = df['excess_return'].rolling(1260).mean() - df['beta_60m'] * df['market_excess_return'].rolling(1260).mean()
    df['alpha_18m_6mchg'] = df['alpha_60m'].shift(378) - df['alpha_60m'].shift(126) 
    df['alpha_12m_6mchg'] = df['alpha_60m'].shift(252) - df['alpha_60m'].shift(126) 
    df['alpha_36m_6mchg'] = df['alpha_60m'].shift(756) - df['alpha_60m'].shift(126)  # 36M shift - 6M shift
    df['rel_pr_str_12m'] = df['return'].rolling(252).mean() / df['market_excess_return'].rolling(252).mean()
    # Sharpe Ratio
    df['sharpe_60d'] = df['excess_return'].rolling(60).mean() / df['excess_return'].rolling(60).std()
    #pre-processing
    df = df.drop(columns=['cov_stock_market','market_variance','excess_return','market_excess_return'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[features_list] = df[features_list].ffill()
    return df
    
def store_dataframe_in_database(database_path, table_name, dataframe):
    conn = sqlite3.connect(database_path)
    dataframe.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()
    
if __name__ == "__main__":
    data_dir = "../data"
    database_path = os.path.join(data_dir, "historical_data.db")
    json_path = os.path.join(data_dir, "stock_names_list.json")
    with open(json_path, "r") as file:
        tickers_list = json.load(file) 
    market_data, risk_free_data = load_market_data(database_path)
    market_data = market_data.rename(columns={"Date": "date"})
    risk_free_data = risk_free_data.rename(columns={"DATE": "date"})
    database_path_TI = os.path.join(data_dir, "tech_ind.db")
    create_database(database_path_TI) 
    for ticker in tickers_list:
        df = load_ticker_data(database_path, ticker)
        sector = df['sector'][0]
        sector_df = load_industry_ETF_data(database_path, sector)
        if df.empty:
            print(f"Warning: No data found for {ticker}. Skipping...")
            continue
        df_indicators = compute_technical_indicators(df, market_data, risk_free_data, sector_df)
        store_dataframe_in_database(database_path_TI, "technical_indicators", df_indicators)
    
