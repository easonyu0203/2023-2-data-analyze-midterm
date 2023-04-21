import os

import pymongo
import dotenv
from pathlib import Path

# env
dotenv.load_dotenv()
docs_path = Path(os.getenv("ORGANIZED_DATASET_DIR"), os.getenv("DOCS_NAME"))
db_url = os.getenv("DB_URL")
db_name = os.getenv("DB_NAME")
docs_collection_name = os.getenv("DOCS_COLLECTION_NAME")
stock2doc_collection_name = os.getenv("STOCK_2_DOC_COLLECTION_NAME")


def get_docs_collection() -> pymongo.collection.Collection:
    """Connect to MongoDB and return the collection"""
    client = pymongo.MongoClient(db_url)
    db = client[db_name]
    collection = db[docs_collection_name]
    return collection


def get_stock2doc_collection() -> pymongo.collection.Collection:
    """Connect to MongoDB and return the collection"""
    client = pymongo.MongoClient(db_url)
    db = client[db_name]
    collection = db[stock2doc_collection_name]
    return collection
