from typing import Protocol, Tuple, List

from sklearn.feature_extraction.text import TfidfVectorizer

from datasets.labeled_docs_dataset import ILabeledDataset, LabelDataset


class IVectorlizer(Protocol):
    """IVectorlizer convert keywords to vector"""

    def convert(self, labeled_docs_keywords: ILabeledDataset, verbose=True) -> ILabeledDataset:
        ...


class TFIDFVectorlizer(IVectorlizer):
    """
    TFIDFVectorlizer convert keywords to vector
    we use keyword as the token and weight as the weight of the token to generate the vector.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def convert(self, labeled_docs_keywords: ILabeledDataset, verbose=True) -> ILabeledDataset:
        """
        TFIDFVectorlizer convert keywords to vector
        we use keyword as the token and weight as the weight of the token to generate the vector.
        """
        # Convert the input list of keywords with weights into a list of strings
        docs = []
        for keywords, label in labeled_docs_keywords:
            # convert keywords to string with weights by joining them with space
            doc = " ".join([' '.join([k] * int(w * 100)) for k, w in keywords])
            docs.append(doc)

        # fit the vectorizer
        tfidf_matrix = self.vectorizer.fit_transform(docs)
        vectors = tfidf_matrix.toarray()

        return LabelDataset(vectors, labeled_docs_keywords.labels)
