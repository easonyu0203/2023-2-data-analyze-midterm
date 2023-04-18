from typing import Protocol, List, Tuple
from tqdm import tqdm
from datasets.document import Document
from datasets.labeled_docs_dataset import ILabeledDataset, LabelDataset


class IKeywordExtractor(Protocol):
    """
    Extract keywords from documents. for each doc, we extract a list of keywords and their weights.
    this return a labeled dataset contain (x, y) where x is a list of pair(keyword, weight) and y is a float value.
    """

    def extract_keywords(self, labeled_docs: ILabeledDataset, verbose=True) -> ILabeledDataset:
        """extract keywords from documents."""
        ...

    def transform(self, doc: Document) -> List[Tuple[str, float]]:
        """transform a document to a list of keywords and their weights"""
        ...


class DefaultKeywordExtractor(IKeywordExtractor):
    """
    by default, we use jieba to extract keywords from documents.
    you can set the max number of keywords to extract by setting the topK parameter.
    this return a labeled dataset contain (x, y) where x is a list of pair(keyword, weight) and y is a float value.
    set topK to None to extract all keywords.
    """

    def __init__(self, topK: int = None):
        self.topK = topK

    def extract_keywords(self, labeled_docs: ILabeledDataset, verbose=True) -> ILabeledDataset:
        doc: Document
        doc_keywords_list: List[List[Tuple[str, float]]] = []

        p_bar = tqdm(labeled_docs, desc="extracting keywords", disable=not verbose)
        for doc, label in p_bar:
            # since we have already extracted keywords from documents, we can skip this step.
            keywords = self.transform(doc)
            doc_keywords_list.append(keywords)

        # remove doc if keywords == []
        doc_keywords_list = [doc_keywords for doc_keywords in doc_keywords_list if doc_keywords != []]
        if verbose:
            print(f"remove {len(labeled_docs) - len(doc_keywords_list)} docs because of empty keywords")
            print("left with {} docs".format(len(doc_keywords_list)))

        return LabelDataset(doc_keywords_list, labeled_docs.labels)

    def transform(self, doc: Document) -> List[Tuple[str, float]]:
        """transform a document to a list of keywords and their weights"""
        assert doc.keywords is not None
        return doc.keywords


if __name__ == "__main__":
    DefaultKeywordExtractor(topK=10)
