from typing import Protocol

from tqdm import tqdm

from datasets.labeled_docs_dataset import ILabeledDataset, LabelDataset
from datasets.stock_dataset import Stock


class ILabeledDocsFilterer(Protocol):
    """Filter labeled documents that aren't relevant to the stock we are interested in."""

    def filter_documents(self, labeled_documents: ILabeledDataset, stock: Stock, verbose=True) -> ILabeledDataset:
        """filter labeled documents by relevant to the stock we are interested in."""
        ...


class Near0returnFilterer(ILabeledDocsFilterer):
    """filter out y (future return %) is close to 0"""

    def __init__(self, threshold: float = 5):
        self.threshold = threshold

    def filter_documents(self, labeled_documents: ILabeledDataset, stock: Stock, verbose=True) -> ILabeledDataset:
        stock_name = stock.name
        filtered_documents = []
        labels = []
        pbar = tqdm(labeled_documents, desc="filtering documents", disable=not verbose)
        for (document, label) in pbar:
            if abs(document.label) > self.threshold:
                filtered_documents.append(document)
                labels.append(label)

        return LabelDataset(features=filtered_documents, labels=labels)