# Extract historical fundamental financial data from SEC 10-Q filings.
# Utilize similarity score from sentence transformers to select top 3 labels matching the required fundamental data.
# Utilize LLMs and prompt engineering to obtain the correct label (necessary since quite frequently the highest similarity score corresponds to a wrong label).
# Write error outputs to a file to monitor the accuracy of the data obtained. 
# Store the fundamental data in a database filing-by-filing to avoid memory issues.

import sqlite3
import pandas as pd
import numpy as np
import json
import time
from dotenv import load_dotenv
import os
from datetime import datetime, date
from edgar import Company, set_identity
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
load_dotenv(dotenv_path="../.env")
name = os.getenv("NAME")
email = os.getenv("EMAIL")
set_identity(name + ' ' + email)
groq_api_key = os.getenv("GROQ_API_KEY")                           
llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")


def create_database(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fundamental_data (
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        total_debt INTEGER,
        total_assets INTEGER,
        total_liabilities INTEGER,
        total_equity INTEGER,
        book_value_of_equity INTEGER,
        financial_leverage REAL,
        shares_outstanding INTEGER,
        EPS_basic REAL,
        EPS_diluted REAL,
        ROIC REAL,
        PRIMARY KEY (ticker, date)
    )
    """)
    conn.commit()
    conn.close()
    
def get_most_similar_indices(description_list, predefined_label, number_of_indices):
    dict_sim_score ={}
    for item in description_list:
        predefined_embedding = sentence_model.encode(predefined_label, convert_to_tensor=True)
        description_embedding = sentence_model.encode(item, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(predefined_embedding, description_embedding).item()
        dict_sim_score[item] = similarity_score
    top_sim_score_indices = sorted(range(len(dict_sim_score)), key=lambda i: list(dict_sim_score.values())[i], reverse=True)[:number_of_indices]
    return top_sim_score_indices
    
def get_sheet_details(sheet_data, question_keywords, prompt_template, is_shares_outstanding):
    retriever_chain = prompt_template | llm
    row_label_list =[]
    sheet_data.index = [f"(index={j}, label={label})" for j, label in enumerate(sheet_data.index)]
    for row_label in sheet_data.index:
        row_label_list.append(row_label)
    description_list =[]
    column_headings_list = sheet_data.columns.tolist()
    top_indices = get_most_similar_indices(column_headings_list, "concept or details or description", 1)
    description_index = next(iter(top_indices))
    for row_label in sheet_data.index:
        description_list.append(row_label+'; Description='+sheet_data.loc[row_label][description_index])
    if is_shares_outstanding:
        top_indices = get_most_similar_indices(description_list, "Shares outstanding or issued or common stock", 3)
        time.sleep(5)
        question = question_keywords[0]
        context="Here are some financial details: \n"
        for index in top_indices:
            context+=f"description = {description_list[index]}; Value = {sheet_data.iloc[index][0]}. \n"
        try:
            response = retriever_chain.invoke({"question": question,"context":context})
        except Exception as e:
            with open("Error_list.txt", "a") as file: file.write("Issue with LLM invoke.\n")
            return 0
        try: 
            return int(response.content)
        except ValueError:
            with open("Error_list.txt", "a") as file: file.write("Value Error with outstanding shares\n")
            return 0
    else:
        answers_list=[0] * len(question_keywords)
        for k, keyword in enumerate(question_keywords):
            time.sleep(5)
            top_indices = get_most_similar_indices(description_list, keyword, 3)
            question = "What is the index of the "+ keyword + "?"
            context="Here are some financial details in the format: index -> description: \n"
            for index in top_indices:
                context+=f"index = {index} -> description = {description_list[index]}; Value = {sheet_data.iloc[index][0]}. \n"
            try:
                response = retriever_chain.invoke({"question": question,"context":context})
            except Exception as e:
                with open("Error_list.txt", "a") as file: file.write("Issue with LLM invoke\n")
                answers_list[k] = 0
                continue
            try:
                matching_index=int(response.content)
                answers_list[k] = float(sheet_data.iloc[matching_index][0])
            except ValueError:
                with open("Error_list.txt", "a") as file: file.write(f"Value Error with {keyword}.\n")
                answers_list[k] = 0
            except Exception as e:
                with open("Error_list.txt", "a") as file: file.write(f"Unexpected Error with {keyword}.\n")
                answers_list[k] = 0
        return answers_list
    return 0
    
def get_fundamental_data(ticker_name,start_date,j):
    ticker = Company(ticker_name)
    filings = ticker.get_filings(form="10-Q")
    data_list = []
    i=0
    for filing in filings:
        i+=1
        #if(i!=j): continue #For debugging
        try:
            date = filing.filing_date
            if date < datetime.strptime(start_date, "%Y-%m-%d").date(): continue
            time.sleep(20)
            with open("Error_list.txt", "a") as file: file.write(f"Ticker={ticker_name}, date={date}, access_index={i}\n")
            filing_object = filing.obj()
            balance_sheet_data = filing_object.balance_sheet.data
            income_statement_data = filing_object.income_statement.data
            prompt_template1 = ChatPromptTemplate.from_template("""You are a helpful assistant. Please answer the question {question} as truthfully as 
                            possible in one integer value based on the given text:{context}, no preamble, no other characters except the integer such that 
                            your answer can easily be converted to an integer data. """)
            prompt_template2 = ChatPromptTemplate.from_template("""You are a helpful assistant. Please answer the question {question} as truthfully as 
                            possible in one integer value based on the given text:{context}. Remember: no preamble, no other characters except the integer such that 
                            your answer can easily be converted to an integer data. Calculate if necessary from shares issued""")
        except Exception as e:
            with open("Error_list.txt", "a") as file: file.write(f"Error processing filing or prompt: {e}\n")
            continue
        question_keywords=["long term debt current or short term borrowings or if these are not available then current liabilities", 
                            "Not current long term debt", 
                            "Total assets", 
                            "Total liabilities", 
                            "Total equity", 
                            "marketable securities short term or current or Available For Sale Securities Current", 
                            "marketable securities long term or noncurrent or or Available For Sale Securities noncurrent", 
                            "Cash and cash equivalents"]
        balance_sheet_answers = [0.0] * len(question_keywords)
        try: 
            balance_sheet_answers = get_sheet_details(balance_sheet_data, question_keywords, prompt_template1, False)
        except Exception as e:
            with open("Error_list.txt", "a") as file: file.write("Error in balance sheet. \n")
        question_keywords = ["what is the total shares outstanding or common stock?"]
        shares_outstanding_answer = 0
        try: 
            shares_outstanding_answer = get_sheet_details(balance_sheet_data, question_keywords, prompt_template2, True)
        except Exception as e:
            with open("Error_list.txt", "a") as file: file.write("Error in shares outstanding.\n")
        question_keywords=["Basic earnings per share (in dollars per share) or EPS basic",
                            "Diluted earnings per share (in dollars per share) or EPS diluted",
                            "Provision for income taxes",
                            "Income before provision for income taxes",
                            "Operating income"]
        income_statement_answers = [0.0] * len(question_keywords)
        try: 
            income_statement_answers = get_sheet_details(income_statement_data, question_keywords, prompt_template1, False)
        except Exception as e:
            with open("Error_list.txt", "a") as file: file.write("Error in income sheet.\n")
        total_debt=balance_sheet_answers[0]+balance_sheet_answers[1]
        if balance_sheet_answers[4]!=0: 
            fin_lev = float(total_debt)/float(balance_sheet_answers[4])
        else: 
            fin_lev = 0
        non_operating_assets = balance_sheet_answers[5]+balance_sheet_answers[6]+balance_sheet_answers[7]
        if income_statement_answers[3]!=0: 
            tax_rate = float(income_statement_answers[2])/float(income_statement_answers[3])
        else: 
            tax_rate = 0
        NOPAT = income_statement_answers[4] * (1 - tax_rate)
        invested_capital = total_debt + balance_sheet_answers[4] - non_operating_assets
        if invested_capital!=0: 
            ROIC = NOPAT / invested_capital
        else: 
            ROIC = 0
        data_list.append({
            "ticker": ticker_name,
            "date": pd.to_datetime(date),
            "total_debt": total_debt,
            "total_assets": balance_sheet_answers[2],
            "total_liabilities": balance_sheet_answers[3],
            "total_equity": balance_sheet_answers[4],
            "book_value_of_equity": balance_sheet_answers[2] - balance_sheet_answers[3],
            "financial_leverage": fin_lev,
            "shares_outstanding": shares_outstanding_answer,
            "EPS_basic": income_statement_answers[0],
            "EPS_diluted": income_statement_answers[1],
            "ROIC": ROIC
        })
    return pd.DataFrame(data_list)  
    
def fill_missing_dates(df, start_date, end_date):
    df['date'] = pd.to_datetime(df['date'])
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    result = []
    for current_date in date_range:
        current_row = {"date": current_date}
        current_row["ticker"] = df['ticker'][0]
        for column in df.columns:
            if column == 'date' or column == 'ticker':
                continue
            value = None
            preceding_rows = df[df['date'] < current_date].sort_values('date', ascending=False)
            for _, row in preceding_rows.iterrows():
                if row[column] != 0:
                    value = row[column]
                    break
            if value is None:
                succeeding_rows = df[df['date'] > current_date].sort_values('date', ascending=True)
                for _, row in succeeding_rows.iterrows():
                    if row[column] != 0:
                        value = row[column]
                        break
            if value is None:
                current_row[column] = 0
                with open("Error_list.txt", "a") as file: 
                    file.write(f"Ticker={df['ticker'][0]} has no non-zero values for {column} \n")
            else:
                current_row[column] = value
        result.append(current_row)
    return pd.DataFrame(result)
    
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
    create_database(database_path) 
    for idx, ticker in enumerate(tickers_list, start=1):
        start_time = time.time()
        start_date_for_files = "2018-01-01"
        fundamentals_df = get_fundamental_data(ticker, start_date_for_files, 0)
        start_date = "2019-01-01"
        end_date = "2024-12-31"
        fundamentals_df_with_all_dates = fill_missing_dates(fundamentals_df, start_date, end_date)
        store_dataframe_in_database(database_path, "fundamental_data", fundamentals_df_with_all_dates)
        end_time = time.time()
        print(f"Time taken for {idx}.{ticker}: {end_time - start_time:.2f} seconds")
        if idx%5==0:
            print("Waiting 24 hrs for GROQ API")
            time.sleep(86400)
            print("Waiting period over")

