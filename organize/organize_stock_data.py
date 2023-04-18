# # Organize Stock Dataset Stock data is store in an Excel file but the structure isn't what we ideally want. In this
# notebook we aim to do the following 1. make two Excel files one contain all 上市股票 other contain all 上櫃股票 2. separate
# each stock to a sheet and order by time 3. rename column to english so will be easy to use in the future
from typing import Dict
import dotenv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

dotenv.load_dotenv()

# change to project root directory
folder_name = "organize"
if folder_name in os.getcwd():
    os.chdir(os.path.abspath(os.pardir))

# # Configuration
DATA_DIR = os.getenv("RAW_DATA_DIR")  # "./bda2023_mid_dataset"
STOCKS_DIR = Path(os.getenv("ORGANIZED_DATASET_DIR"), "stocks")  # "./organized_data/stocks"
METADATA_PATH = Path(os.getenv("ORGANIZED_DATASET_DIR"), "stock_meta.csv")  # "./organized_data/stock_metadata.csv"
RAW_DATASET_NAME = 'stock_data_2019-2023.xlsx'
RENAMED_COLUMNS = [
    'name', 'date', 'open', 'high', 'low', 'close', 'volume(k)', 'turnover(k)', 'transaction', 'outstanding(k)', 'pe',
    'pb'
]
raw_stocks_path = Path(DATA_DIR, RAW_DATASET_NAME)


# # Utility functions
def get_raw_stocks_dfs() -> Dict[str, pd.DataFrame]:
    """
    get raw stocks dfs, this function will load the excel file and return a dict of dfs
    """
    # Load the Excel file
    excel_file = pd.ExcelFile(raw_stocks_path)

    # Get the sheet names
    sheet_names = excel_file.sheet_names

    # Load all sheets
    dfs = {sheet_name: excel_file.parse(sheet_name, na_values=['-']) for sheet_name in sheet_names if
           sheet_name != '摘要'}
    return dfs


# # Research & Preprocessing
# I just simply open excel and look around a bit, here we check about null value and see
# if there is any error in the data
raw_stock_dfs = get_raw_stocks_dfs()

# Look like there is some stock have missing prices value, we will mark them as null value and handle it later
# first, let change column into english, and cast data type to what we want
for sheet_name, df in raw_stock_dfs.items():
    # rename columns
    df.columns = RENAMED_COLUMNS
    # cast data
    df['date'] = pd.to_datetime(df['date'])
    # if name have char * replace with _
    df['name'] = df['name'].str.replace('*', '_')

# make each stock a df
stock_dfs = dict()
for sheet_name, df in raw_stock_dfs.items():
    # make each stock a sheet
    for stock_id in tqdm(df['name'].unique()):
        stock_df = df[df['name'] == stock_id]
        # if stock_dfs have stock_id, stack it
        if stock_id in stock_dfs:
            stock_dfs[stock_id] = pd.concat([stock_dfs[stock_id], stock_df])
        else:
            stock_dfs[stock_id] = stock_df

# order by time
for stock_id, df in stock_dfs.items():
    stock_dfs[stock_id] = df.sort_values(by='date')

# Now we make metadata for all stock, this should include row_cnt, have_null_price, missing_rows_cnt
# make metadata for all stock
stock_metadata = dict()
for stock_id, df in stock_dfs.items():
    # get row count
    row_cnt = df.shape[0]
    # check if there is any null price
    have_null_price = df['close'].isnull().any()
    stock_metadata[stock_id] = {
        "id": stock_id.split(' ')[0],
        "name": stock_id.split(' ')[1],
        'row_cnt': row_cnt,
        'have_null_price': have_null_price,
        "history_path": "",
    }
# check type 上市 or 上櫃
for sheet_name, df in raw_stock_dfs.items():
    stock_type = "上市" if sheet_name == "上市股票" else "上櫃"
    for stock_id in df['name'].unique():
        stock_metadata[stock_id]['stock_type'] = stock_type
# convert to df
stock_metadata_df = pd.DataFrame(stock_metadata).T

for stock_id, df in stock_dfs.items():
    stock_dfs[stock_id] = stock_dfs[stock_id].drop(columns=['name'])

# Save metadata and stock dfs csv files
# save to csv
os.makedirs(STOCKS_DIR, exist_ok=True)
for stock_id, df in tqdm(stock_dfs.items()):
    csv_file_path = Path(STOCKS_DIR, f"{stock_id}.csv")
    stock_metadata_df.loc[stock_id, 'history_path'] = str(csv_file_path.absolute())
    df.to_csv(csv_file_path, index=False)
metadata_csv_path = Path(METADATA_PATH)
stock_metadata_df.to_csv(metadata_csv_path, index=False)
