# # Organize Article Dataset
# The raw dataset is a little messy, in this notebook, we aim to process dataset to became more structured, here is main steps:
# 1. Remove unused columns
# 2. covert data types( exp: post_time to datetime)
# 3. for each year/kind we make a new csv file
# 4. each year stock data separate to a csv file


import pandas as pd
from pathlib import Path
from typing import Literal
import os
from dotenv import load_dotenv;

load_dotenv()

# change to project root directory
folder_name = "organize"
if folder_name in os.getcwd():
    os.chdir(os.path.abspath(os.pardir))

# configurations
DATA_DIR = os.getenv("RAW_DATA_DIR")  # "./bda2023_mid_dataset"
ORGANIZED_DATASET_DIR = os.getenv("ORGANIZED_DATASET_DIR")  # "./organized_data"
ORGANIZED_DATASET_NAME = os.getenv("DOCS_NAME")
RAW_DATASET_NAMES = {
    'bbs_2019_2021': "bda2023_mid_bbs_2019-2021.csv",
    'bbs_2022_2023': "bda2023_mid_bbs_2022-2023.csv",
    'forum_2019': "bda2023_mid_forum_2019.csv",
    'forum_2020': "bda2023_mid_forum_2020.csv",
    'forum_2021': "bda2023_mid_forum_2021.csv",
    'forum_2022_2023': "bda2023_mid_forum_2022-2023.csv",
    'news_2019': "bda2023_mid_news_2019.csv",
    'news_2020': "bda2023_mid_news_2020.csv",
    'news_2021': "bda2023_mid_news_2021.csv",
    'news_2022': "bda2023_mid_news_2022.csv",
    'news_2022_2023': "bda2023_mid_news_2022-2023.csv",
    'news_2023': "bda2023_mid_news_2023.csv",
}


# # Utility functions
def have_same_columns(*dfs):
    """check df have same columns or not"""
    for i in range(len(dfs)):
        for j in range(i + 1, len(dfs)):
            if not dfs[i].columns.equals(dfs[j].columns):
                print(f"columns of {i} and {j} are not equal")
                return False
    return True


def get_article_dfs_by_type(article_type: Literal['bbs', 'forum', 'news']) -> pd.DataFrame:
    """
    get article dfs by type, type can be bbs, forum, news
    return a dataframe that contains all data of that type.
    exp: get_article_dfs_by_type('bbs') will return a dataframe that contains all bbs data from 2019 to 2023
    """
    raw_dataset_paths = {k: Path(DATA_DIR, v) for k, v in RAW_DATASET_NAMES.items()}
    dfs = [pd.read_csv(file_path) for k, file_path in raw_dataset_paths.items() if k.startswith(article_type)]
    assert have_same_columns(*dfs)
    df = pd.concat(dfs)
    return df


# load all article df
bbs_df = get_article_dfs_by_type('bbs')
forum_df = get_article_dfs_by_type('forum')
news_df = get_article_dfs_by_type('news')
article_df = pd.concat([bbs_df, forum_df, news_df])

# After some understanding of the dataset, we decided to do the following
# 1. as three types of articles df is pretty much the same, so I decided to stack them together
# 2. remove duplicates (by id)
# 3. remove useless columns, we don't need id and page url
# 4. sort by post_time (first converted to datetime object)
# 
# NOTE: we don't handle Null values here, because the way we handle null values depends on the algorithm we want to use.

# # Preprocessing
# remove duplicated rows by id
original_rows = article_df.shape[0]
article_df = article_df.drop_duplicates(subset=['id'])
dropped_rows = original_rows - article_df.shape[0]
print(f"{dropped_rows} rows were dropped due to duplicates.")

# drop id and page url
print("drop id and page url")
article_df = article_df.drop(columns=['id', 'page_url'])

# convert post_time to datetime object and sort by post_time
print("convert post_time to datetime object and sort by post_time")
article_df['post_time'] = pd.to_datetime(article_df['post_time'])
article_df = article_df.sort_values(by='post_time')

# if author or title or content is NaN, convert to ""
article_df['author'].fillna('""', inplace=True)
article_df['title'].fillna('""', inplace=True)
article_df['content'].fillna('""', inplace=True)

print("set post_time as index and drop duplicates from index")
article_df.set_index('post_time', inplace=True)
# check for duplicates in the index
if article_df.index.duplicated().any():
    # drop duplicates from the index
    article_df = article_df[~article_df.index.duplicated()]

# the preprocessed data seem good, store it to csv
print("saving...")
article_csv_path = Path(ORGANIZED_DATASET_DIR, ORGANIZED_DATASET_NAME)
# create preprocessed_dataset dir if not exists
if not Path(ORGANIZED_DATASET_DIR).exists():
    Path(ORGANIZED_DATASET_DIR).mkdir()
# save to csv file
article_df.to_csv(article_csv_path)
print(f"article df is saved to {article_csv_path}")
