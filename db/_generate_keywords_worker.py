# import jieba.analyse
import jieba_fast.analyse


def generate_keywords(doc):
    """Generates keywords for the given document using jieba's textrank algorithm."""
    try:
        doc_str = ";".join([doc['title'], doc['author'], doc['content']])
        keywords = jieba_fast.analyse.textrank(doc_str, topK=None, withWeight=True)
    except Exception as e:
        print(f"Exception while processing document {doc['_id']}: {e}")
        keywords = []
    return doc['_id'], keywords
