from typing import Protocol
from datasets.docs_dataset import IDocsDataset, DocsDataset
from datasets.stock_dataset import Stock
from tqdm import tqdm


class IDocsFilterer(Protocol):
    """Filter documents that aren't relevant to the stock we are interested in."""

    def filter_documents(self, documents: IDocsDataset, stock: Stock, verbose=True) -> IDocsDataset:
        """filter documents by relevant to the stock we are interested in."""
        ...


class DefaultFilterer(IDocsFilterer):
    """use whether doc title or content contains the stock name to filter documents"""

    def filter_documents(self, documents: IDocsDataset, stock: Stock, verbose=True) -> IDocsDataset:
        if verbose:
            print("[DefaultFilterer] filtering documents by whether doc title or content contains the stock name")
        stock_name = stock.name
        filtered_documents = []
        p_bar = tqdm(documents, desc="filtering documents", disable=not verbose)
        for document in p_bar:
            filtered_documents.append(document)
            if stock_name in document.title or stock_name in document.content:
                filtered_documents.append(document)


        if verbose:
            # print remaining count
            print(f"left with {len(filtered_documents)} documents after filtering")

        return DocsDataset(document_list=filtered_documents)


