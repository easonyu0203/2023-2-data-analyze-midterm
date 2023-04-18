import pickle
import time

from tqdm import tqdm

from db._generate_keywords_worker import generate_keywords
from db.connect import connect_db
import concurrent.futures

import concurrent.futures
from pymongo import UpdateOne

import itertools


def batch(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


def store_generated_keywords(verbose=True, num_workers=4, batch_size=10000):
    """Using jieba to generate keywords for each document and store them in the database."""
    collection = connect_db()

    if verbose: print("loading documents from database...")
    update_operations = []
    docs_to_process = collection.find({'keywords': []})
    total_count = docs_to_process.count()
    if verbose: print(f"found {total_count} documents to process, start processing...")
    pbar = tqdm(docs_to_process, total=total_count, desc="generating keywords", disable=not verbose)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        update_operations = []
        for batch_docs in batch(docs_to_process, batch_size):
            future_to_docId = {executor.submit(generate_keywords, doc): doc['_id'] for doc in batch_docs}
            for future in concurrent.futures.as_completed(future_to_docId):
                pbar.update(1)
                doc_id = future_to_docId[future]
                try:
                    doc_id, keywords = future.result()
                    update_operations.append(UpdateOne({'_id': doc_id}, {'$set': {'keywords': keywords}}))

                except Exception as exc:
                    print(f"Exception while processing document {doc_id}: {exc}")

            # update the remaining entries, if any
            if update_operations:
                collection.bulk_write(update_operations)


if __name__ == '__main__':
    # store_generated_keywords_single_process(verbose=True)
    store_generated_keywords(verbose=True, num_workers=10, batch_size=10000)
