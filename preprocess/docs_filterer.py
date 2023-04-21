import os
import pickle
from pathlib import Path
from typing import Protocol
from datasets.docs_dataset import IDocsDataset, DocsDataset
from datasets.stock_dataset import Stock
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

default_filt_cache_dir = Path(os.getenv("CACHE_DIR")) / "default_filterer"

class IDocsFilterer(Protocol):
    """Filter documents that aren't relevant to the stock we are interested in."""

    def filter_documents(self, documents: IDocsDataset, stock: Stock, verbose=True) -> IDocsDataset:
        """filter documents by relevant to the stock we are interested in."""
        ...


class DefaultFilterer(IDocsFilterer):
    """use whether doc title or content contains the stock name to filter documents"""

    def __init__(self, max_docs: int = None):
        """
        filter documents by whether doc title or content contains the stock name
        :param max_docs: maximum number of documents to keep after filtering
        """
        self.max_docs = max_docs

    def filter_documents(self, documents: IDocsDataset, stock: Stock, max_docs=None, verbose=True) -> IDocsDataset:
        if verbose:
            print("[DefaultFilterer] filtering documents by whether doc title or content contains the stock name")

        # load from cache if exists
        if self.max_docs is None and default_filt_cache_dir.exists():
            if verbose: print(f"load from cache: {default_filt_cache_dir / f'{stock.name}.pkl'}")
            filtered_documents = pickle.load(open(default_filt_cache_dir / f"{stock.name}.pkl", "rb"))
            return DocsDataset(document_list=filtered_documents)

        # perform filtering
        stock_name = stock.name
        filtered_documents = []
        p_bar = tqdm(documents, desc="filtering documents", disable=not verbose)
        for document in p_bar:
            if self.max_docs is not None and len(filtered_documents) >= self.max_docs:
                break
            if stock_name in document.title or stock_name in document.content:
                filtered_documents.append(document)

        # save to cache using pickle
        if self.max_docs is None:
            if verbose: print(f"save to cache: {default_filt_cache_dir / f'{stock.name}.pkl'}")
            default_filt_cache_dir.mkdir(parents=True, exist_ok=True)
            pickle.dump(filtered_documents, open(default_filt_cache_dir / f"{stock.name}.pkl", "wb"))

        if verbose:
            # print remaining count
            print(f"left with {len(filtered_documents)} documents after filtering")

        return DocsDataset(document_list=filtered_documents)
