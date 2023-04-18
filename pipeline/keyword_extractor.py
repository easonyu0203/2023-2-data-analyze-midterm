from typing import Protocol, List, Tuple
import jieba.analyse

from datasets.docs_dataset import IDocsDataset
from datasets.document import Document
from datasets.labeled_docs_dataset import ILabeledDataset, LabelDataset


class IKeywordExtractor(Protocol):
    """
    Extract keywords from documents. for each doc, we extract a list of keywords and their weights.
    this return a labeled dataset contain (x, y) where x is a list of pair(keyword, weight) and y is a float value.
    """
    def extract_keywords(self, labeled_docs: ILabeledDataset) -> ILabeledDataset:
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

    def extract_keywords(self, labeled_docs: ILabeledDataset) -> ILabeledDataset:

        doc: Document
        doc_keywords: List[Tuple[str, float]] = []
        for doc, label in labeled_docs:
            doc_str = ";".join([doc.title, doc.author, doc.content])
            # extract the keywords
            keywords = jieba.analyse.textrank(doc_str, topK=self.topK, withWeight=True)
            doc_keywords.append(keywords)

        return LabelDataset(doc_keywords, labeled_docs.labels)



if __name__ == "__main__":
    DefaultKeywordExtractor(topK=10)