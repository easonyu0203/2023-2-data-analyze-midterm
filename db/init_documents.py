import os
from pathlib import Path

import dotenv
import pandas as pd

from db.connect import connect_db

# env
dotenv.load_dotenv()
docs_path = Path(os.getenv("ORGANIZED_DATASET_DIR"), os.getenv("DOCS_NAME"))


def init_db(collection):
    """Initialize the MongoDB collection with documents from the CSV file
    if already initialized, reset the collection"""

    # Read the CSV file using pandas
    data = pd.read_csv(docs_path)
    # Add a new column for keywords, initialized as empty lists
    data['keywords'] = data.apply(lambda _: [], axis=1)
    # Convert the DataFrame to a list of dictionaries
    documents = data.to_dict(orient='records')
    # Insert documents into the MongoDB collection
    collection.insert_many(documents)


if __name__ == '__main__':
    connection = connect_db()
    init_db(connection)
