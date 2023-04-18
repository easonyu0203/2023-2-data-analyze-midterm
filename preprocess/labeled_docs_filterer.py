from typing import Protocol

from tqdm import tqdm

from datasets.labeled_docs_dataset import ILabeledDataset, LabelDataset


class ILabeledDocsFilterer(Protocol):
    """filter docs by the label"""

    def filter_documents(self, labeled_documents: ILabeledDataset, verbose=True) -> ILabeledDataset:
        """filter docs by the label"""
        ...


class Near0returnFilterer(ILabeledDocsFilterer):
    """filter out y (future return %) is close to 0"""

    def __init__(self, threshold: float = 5):
        self.threshold = threshold

    def filter_documents(self, labeled_documents: ILabeledDataset, verbose=True) -> ILabeledDataset:
        """filter out y (future return %) is close to 0"""
        filtered_documents = []
        labels = []
        pbar = tqdm(labeled_documents, desc="filtering documents", disable=not verbose)
        for (document, label) in pbar:
            if abs(document.label) > self.threshold:
                filtered_documents.append(document)
                labels.append(label)

        if verbose:
            print(f"left with {len(filtered_documents)} documents after filtering")

        return LabelDataset(features=filtered_documents, labels=labels)