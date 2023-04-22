from typing import Protocol, List
from datasets.docs_dataset import IDocsDataset, DocsDataset
from datasets.stock_dataset import Stock
from tqdm import tqdm
import jieba_fast
from gensim.models import Word2Vec
import numpy as np
from utils.cacher import Cacher


class IDocsFilterer(Protocol):
    """Filter documents that aren't relevant to the stock we are interested in."""

    def fit(self, documents: IDocsDataset, stock: Stock, verbose=True):
        """fit the filterer"""
        ...

    def filter_documents(self, documents: IDocsDataset, stock: Stock, verbose=True) -> IDocsDataset:
        """filter documents by relevant to the stock we are interested in."""
        ...


class StockNameFilterer(IDocsFilterer):
    """Use whether doc title or content contains the stock name to filter documents"""

    def __init__(self, max_docs: int = None):
        """
        Filter documents by whether doc title or content contains the stock name
        :param max_docs: maximum number of documents to keep after filtering
        """
        self.max_docs = max_docs

    def fit(self, documents: IDocsDataset, stock: Stock, verbose=True):
        """no fit needed"""
        ...

    def filter_documents(self, documents: IDocsDataset, stock: Stock, verbose=True) -> IDocsDataset:
        if verbose:
            print("[StockNameFilterer] filtering documents by whether doc title or content contains the stock name")

        # perform filtering
        stock_name = stock.name
        filtered_documents = []
        p_bar = tqdm(documents, desc="filtering documents", disable=not verbose)
        for document in p_bar:
            if self.max_docs is not None and len(filtered_documents) >= self.max_docs:
                break

            title_words = set(document.title.split())
            content_words = set(document.content.split())

            if stock_name in title_words or stock_name in content_words:
                filtered_documents.append(document)

        if verbose:
            # print remaining count
            print(f"left with {len(filtered_documents)} documents after filtering")

        return DocsDataset(document_list=filtered_documents)


class Word2VecSimilarFilterer(IDocsFilterer):
    """
    Use word2vec similarity to filter documents
    we first find the top most similar words to the stock name
    then we filter documents that its keywords contain the top most similar words
    """

    def __init__(self, topn: int = 10, white_noise_ratio: float = 0.1):
        """
        Filter documents by does content/title/author contain the top most similar words to the stock name
        :param topn: number of top most similar words to the stock name to use
        :param white_noise_ratio: for som percentage, we will just add document into dataset without filtering
        """
        self.topn = topn
        self.white_noise_ratio = white_noise_ratio
        self.model = None

    def fit(self, documents: IDocsDataset, stock: Stock, verbose=True):
        """fit the filterer"""
        if verbose:
            print("[Word2VecSimilarFilterer] fitting the filterer")

        # load from cache
        if Cacher.exits("word2vec_100"):
            if verbose:
                print("load word2vec model from cache")
            self.model = Cacher.load("word2vec_100")
        else:
            keyword_docs: List[List[str]] = [[k for k, w in doc.keywords] for doc in documents]
            self.model = Word2Vec(keyword_docs, vector_size=100, window=5, min_count=5, workers=8)
            if verbose:
                print("save word2vec model to cache")
            Cacher.cache(self.model, "word2vec_100")

    def filter_documents(self, documents: IDocsDataset, stock: Stock, verbose=True) -> IDocsDataset:
        if verbose:
            print("[Word2VecSimilarFilterer] Filter documents by does content/title/author contain the top most "
                  "similar words to the stock name")

        # perform filtering
        stock_name = stock.name
        stock_keywords = jieba_fast.cut(stock_name)
        similar_words = set()
        for stock_k in stock_keywords:
            # Get the most similar words for the current stock keyword
            most_similar = {k for k, w in self.model.wv.most_similar(stock_k, topn=self.topn)}
            # Add the most similar words for this stock keyword to the set
            similar_words.update(most_similar)

        filtered_documents = []
        p_bar = tqdm(documents, desc="filtering documents", disable=not verbose)
        for document in p_bar:
            doc_words = set(k for k, w in document.keywords)
            if similar_words.intersection(doc_words):
                filtered_documents.append(document)
        
        white_noise_cnt = int(len(filtered_documents) * self.white_noise_ratio)
        # random select some documents to add white noise, use documents.documents_df
        indices = np.random.choice(len(documents), white_noise_cnt)
        for i in indices:
            filtered_documents.append(documents[i])
            
            

        if verbose:
            # print remaining count
            print(f"left with {len(filtered_documents)} documents after filtering")

        return DocsDataset(document_list=filtered_documents)