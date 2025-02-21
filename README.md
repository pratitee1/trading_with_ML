# AI-Driven Financial Data Extraction & Trading Strategy

## Overview
This repository provides a **scalable and automated pipeline** for extracting **fundamental** and **technical** financial data from free sources, training an **LSTM-based predictive model**, and implementing a **stock selection trading strategy** that outperforms the S&P 500 index. The key components include:

- **Data Extraction**: Automated retrieval of **daily stock prices**, **sector trends**, **market index data**, and **fundamental financial metrics** from SEC filings.
- **Technical Analysis**: Computation of **50+ technical indicators**, incorporating **sector-based** and **market-relative** factors based on this [article](https://www.nature.com/articles/s41598-023-50783-0).
- **Deep Learning-Based Market Prediction**: Training an **LSTM model** to **predict short-term stock performance**.
- **Trading Strategy Development**: A **stock selection strategy** based on **LSTM model outputs**, which **outperforms the S&P 500 equal allocation index** based on this [article](https://www.nature.com/articles/s41598-023-50783-0).

All extracted data is stored in an **efficient SQLite database**, enabling access for financial analysis and strategy backtesting.

---

## Motivation
- **Free & Scalable Financial Data Extraction**: Avoid reliance on expensive financial services by **leveraging public financial data sources**.
- **AI-Driven Market Prediction**: Utilize **LSTM models** to **identify outperforming stocks**.
- **Technical & Fundamental Integration**: Combine **technical indicators** and **fundamental financial data** to enhance **stock selection**.
- **High-Frequency Trading Viability**: Achieve **prediction accuracy above 50%**, which is **profitable in high-frequency trading environments**.
- **Efficient & Scalable Processing**: Store data in **lightweight SQLite databases**, making **historical financial analysis scalable**.

---

## Features

### 1. **Automated Daily Stock & Market Data Extraction**
- **S&P 500 Constituents** dynamically sourced from **Wikipedia**.
- **Daily OHLCV Stock Data** retrieved via **Yahoo Finance (yfinance)**.
- **Sector ETF Data** used as a **proxy for sector-based analysis**.
- **Market Index Benchmarking** using **S&P 500 index** as a market proxy.
- **Risk-Free Rate Calculation** sourced from **10-Year US Treasury Yield (FRED)** via **Pandas-DataReader**.
- **Optimized Storage** in **SQLite**, ensuring **efficient access & scalability**.

📌 **Implementation:** [`get_daily_data_yfinance.py`](data_extraction/get_daily_data_yfinance.py)

---

### 2. **Automated SEC Fundamental Data Extraction**
- **Extracts financial data from SEC 10-Q filings**.
- **Overcomes SEC Label Variability** during parsing:
  - **Sentence Transformers** pre-filter top **3 financial labels** (reducing token usage by **80%**).
  - **LangChain + Groq API** dynamically extracts **correct labels** (**97% accuracy vs. 91% Regex**).
- **Handles Large-Scale Extraction**:
  - **Processes 3000+ SEC filings in 3 days**.
  - **Batch processing** with a **time-delay mechanism** ensures free-tier API efficiency.
  - **Multi-account API strategy** enhances **request throughput**.

📌 **Implementation:** [`get_fundamental_data_sec.py`](data_extraction/get_fundamental_data_sec.py)

---

### 3. **Technical Indicator Computation**
- **Extracts 50+ Technical Indicators** based on the [article](https://www.nature.com/articles/s41598-023-50783-0), including:
  - **Stock-Specific Metrics** (RSI, MACD, Bollinger Bands).
  - **Sector-Relative Features** (industry-adjusted momentum, volatility).
  - **Market-Relative Features** (correlation with S&P 500).
- **Prevents Memory Overload** by processing **stock-by-stock** using **SQLite database**.
- **Scalable for Large Datasets**.

📌 **Implementation:** [`compute_technical_indicators.py`](data_extraction/compute_technical_indicators.py)

---

### 4. **LSTM Data Preparation for Deep Learning Models**
- **Creates Rolling Window Sequences**:
  - **15-day time-series sequences sequences** for LSTM model training.
  - **Avoids data leakage** using a **3-day offset**.
- **Binary Classification Labels** based on future 3-day returns:
  - **Stocks outperforming the cross-sectional median** → **Label 1**.
  - **Underperforming stocks** → **Label 0**.
- **Data Efficiently Stored**:
  - **Compressed `.npz` and `.pkl` files** for fast training & retrieval.

📌 **Implementation:** [`create_LSTM_data.py`](data_extraction/create_LSTM_data.py)

---

### 5. **Deep Learning: LSTM-Based Stock Movement Prediction**
- **Binary Classification Task**:
  - Predicts whether a stock will **overperform or underperform**.
- **Rolling Window Training Framework**:
  - **1020-day training period**.
  - **160-day validation period**.
  - **20-day testing period**.
- **Accuracy > 50% is profitable** in **high-frequency trading** due to market volatility.
- **Ongoing Retraining Required**:
  - **Model must be retrained periodically** using a rolling window **to adapt to market trends**.

📌 **Implementation:** [`LSTM_model_training.ipynb`](ML_trading_strategy/LSTM_model_training.ipynb)

---

### 6. **Stock Selection Trading Strategy**
- **Stock Ranking Methodology**:
  - Uses **LSTM model's predicted probabilities** to **rank stocks**.### 4. **LSTM Data Preparation for Deep Learning Models**
- **Prepares data for training LSTM models** for stock movement prediction.
- **Avoids data leakage** by maintaining a **3-day offset** in training data.
- **Creates rolling time-series sequences** of **15-day windows** for LSTM input.
- **Generates labels based on future 3-day returns**:
  - **Stocks outperforming the cross-sectional median are labeled 1.**
  - **Underperforming stocks are labeled 0.**
- **Data stored as compressed `.npz` and `.pkl` files** for efficient access.
  - Selects **top k stocks** for portfolio allocation.
- **Backtesting shows higher return** compared to **equal allocation S&P 500 index**.
- Requires **continuous model retraining** to **maintain profitability**.

📌 **Implementation:** [`trading_strategy_with_LSTM.ipynb`](ML_trading_strategy/trading_strategy_with_LSTM.ipynb)

---

## Additional Details:
- Install requirements.py
- Save name and email in .env for verification by SEC Edgar API
- Generate Groq-API key and save in .env.
