from typing import Protocol
from datasets.docs_dataset import IDocsDataset, DocsDataset
from datasets.stock_dataset import Stock
from tqdm import tqdm
from utils.cacher import Cacher


class IDocsFilterer(Protocol):
    """Filter documents that aren't relevant to the stock we are interested in."""

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

    def filter_documents(self, documents: IDocsDataset, stock: Stock, verbose=True) -> IDocsDataset:
        if verbose:
            print("[StockNameFilterer] filtering documents by whether doc title or content contains the stock name")

        # load from cache if exists
        cache_name = f'stock_name_filterer__{stock.name}.pkl'
        if self.max_docs is None and Cacher.exits(cache_name):
            if verbose: print(f"load from cache: {cache_name}")
            filtered_documents = Cacher.load(cache_name)
            return DocsDataset(document_list=filtered_documents)

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

        # save to cache using pickle
        if self.max_docs is None:
            cache_name = f'stock_name_filterer__{stock.name}.pkl'
            if verbose: print(f"save to cache: {cache_name}")
            Cacher.cache(cache_name, filtered_documents)

        if verbose:
            # print remaining count
            print(f"left with {len(filtered_documents)} documents after filtering")

        return DocsDataset(document_list=filtered_documents)
