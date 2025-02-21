# Obtain current S&P500 constituents from web.
# (FUTURE WORK: S&P500 constituents should be obtained for the time period before testing starts.)
# Obtain daily OCHLV of S&P500 constituents. 
# Obtain S&P500 daily data for market index approximation.
# Obtain sector ETFs data for sector index approximation.
# Obtain historical risk-free rate from 10-year US treasury yield (from FRED).
# Store data in SQLite database with date and ticker as primary keys.

#Load modules
import yfinance as yf
import pandas as pd
import sqlite3
import json
import os
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings("ignore")

# Dictionary of sector ETFs 
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

# Fetch S&P500 constituents and their sectors
def fetch_sp500_tickers_and_sectors():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]  # First table on the page
    tickers = sp500_table["Symbol"].tolist()
    sectors = sp500_table["GICS Sector"].tolist()
    return pd.DataFrame({"Ticker": tickers, "Sector": sectors})

def create_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Stock price data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_data (
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adj_close REAL,
        volume INTEGER,
        sector TEXT,
        PRIMARY KEY (ticker, date)
    )""")
    # Market index data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS market_index (
        date DATE PRIMARY KEY,
        sp500_close REAL,
        sp500_return REAL
    )""")
    # Risk-free rate table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS risk_free_rate (
        date DATE PRIMARY KEY,
        risk_free_rate REAL
    )""")
    conn.commit()
    return conn

# Store stock data in database
def store_data_in_database(data, ticker, sector, conn):
    data = data.reset_index()
    data['ticker'] = ticker
    data['sector'] = sector
    data = data.rename(columns={
        'Date': 'date', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
    })
    data[['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'sector']].to_sql(
        "daily_data", conn, if_exists='append', index=False
    )

# Fetch market index data (S&P 500)
def fetch_market_index_data(start_date, end_date):
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    sp500 = sp500[['Close']].rename(columns={'Close': 'sp500_close'})
    sp500['sp500_return'] = sp500['sp500_close'].pct_change() 
    return sp500

# Fetch sector index data (using ETFs)
def fetch_sector_index_data(start_date, end_date):
    sector_prices = {}
    for sector, etf in SECTOR_ETFS.items():
        try:
            sector_data = yf.download(etf, start=start_date, end=end_date, progress=False)
            sector_prices[sector] = sector_data[['Close']].rename(columns={'Close': f'{sector}_index'})
        except Exception as e:
            print(f"Error fetching {sector} ETF ({etf}): {e}")
    return sector_prices

# Fetch risk-free rate ()
def fetch_risk_free_rate(start_date, end_date):
    try:
        rfr = pdr.get_data_fred('DGS10', start_date, end_date) 
        rfr = rfr.rename(columns={'DGS10': 'risk_free_rate'}) / 100 
        return rfr
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return None

# Fetch and store data
def filter_tickers_and_store(ticker_data, start_date, end_date, db_path):
    conn = create_database(db_path)
    valid_tickers = []
    required_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    for _, row in ticker_data.iterrows():
        ticker = row['Ticker']
        sector = row['Sector']
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                available_dates = pd.to_datetime(data.index).normalize()
                missing_dates = set(required_dates) - set(available_dates)
                if len(missing_dates) < 190: #Currently depends on the start date and end date
                    valid_tickers.append(ticker)
                    store_data_in_database(data, ticker, sector, conn)
                else:
                    print(f"Ticker {ticker} is missing data for {len(missing_dates)} days.")
            else:
                print(f"No data for ticker {ticker}.")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    conn.close()
    return valid_tickers

if __name__ == "__main__":
    tickers = fetch_sp500_tickers_and_sectors()
    print(f"Total tickers fetched: {len(tickers)}")
    start_date = "2005-01-01"
    end_date = "2024-12-31"
    data_dir = "../data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    db_path = os.path.join(data_dir, "historical_data.db")
    json_path = os.path.join(data_dir, "stock_names_list.json")
    valid_tickers = filter_tickers_and_store(tickers, start_date, end_date, db_path)
    print(f"Total valid tickers with complete data: {len(valid_tickers)}")
    with open(json_path, "w") as file:
        json.dump(valid_tickers, file)
    conn = sqlite3.connect(db_path)    
    # Fetch & Store Market Index Data
    sp500_data = fetch_market_index_data(start_date, end_date)
    sp500_data.to_sql("market_index", conn, if_exists="replace", index=True)
    ### Fetch & Store Sector Index Data
    sector_data = fetch_sector_index_data(start_date, end_date)
    for sector, df in sector_data.items():
        df.to_sql(f"sector_{sector.replace(' ', '_')}", conn, if_exists="replace", index=True)
    ### Fetch & Store Risk-Free Rate Data
    risk_free_data = fetch_risk_free_rate(start_date, end_date)
    if risk_free_data is not None:
        risk_free_data.to_sql("risk_free_rate", conn, if_exists="replace", index=True)
    conn.close()

