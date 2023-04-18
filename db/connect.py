import os

import pymongo
import dotenv
from pathlib import Path

# env
dotenv.load_dotenv()
docs_path = Path(os.getenv("ORGANIZED_DATASET_DIR"), os.getenv("DOCS_NAME"))
db_url = os.getenv("DB_URL")
db_name = os.getenv("DB_NAME")
db_collection_name = os.getenv("DB_COLLECTION_NAME")


def connect_db() -> pymongo.collection.Collection:
    """Connect to MongoDB and return the collection"""
    client = pymongo.MongoClient(db_url)
    db = client[db_name]
    collection = db[db_collection_name]
    return collection


