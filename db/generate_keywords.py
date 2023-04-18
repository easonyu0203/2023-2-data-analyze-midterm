import jieba.analyse
from tqdm import tqdm

from db.connect import connect_db


def store_generated_keywords(verbose=True):
    """using jieba to generate keywords for each document and store them in the database"""
    collection = connect_db()

    pbar = tqdm(collection.find(), desc="generating keywords", disable=not verbose)
    # for each document, extract the keywords and store them in the database
    for doc in pbar:
        if not doc['keywords']:
            # if the keywords are not generated yet, generate them
            # extract the keywords
            doc_str = ";".join([doc['title'], doc['author'], doc['content']])
            keywords = jieba.analyse.textrank(doc_str, topK=None, withWeight=True)
            # store the keywords
            collection.update_one({'_id': doc['_id']}, {'$set': {'keywords': keywords}})


if __name__ == '__main__':
    store_generated_keywords(verbose=True)
