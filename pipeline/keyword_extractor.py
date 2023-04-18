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
            assert doc.keywords is not None
            doc_keywords_list.append(doc.keywords)

        return LabelDataset(doc_keywords_list, labeled_docs.labels)


if __name__ == "__main__":
    DefaultKeywordExtractor(topK=10)
