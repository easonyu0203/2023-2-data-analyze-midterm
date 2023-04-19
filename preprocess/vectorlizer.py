from typing import Protocol, Tuple, List

from sklearn.feature_extraction.text import TfidfVectorizer

from datasets.labeled_docs_dataset import ILabeledDataset, LabelDataset


class IVectorlizer(Protocol):
    """IVectorlizer convert keywords to vector"""

    def transform(self, keywords: List[Tuple[str, float]]) -> List[float]:
        ...

    def convert(self, labeled_docs_keywords: ILabeledDataset, verbose=True) -> ILabeledDataset:
        ...


class TFIDFVectorlizer(IVectorlizer):
    """
    TFIDFVectorlizer convert keywords to vector
    we use keyword as the token and weight as the weight of the token to generate the vector.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def transform(self, keywords: List[Tuple[str, float]]) -> List[float]:
        # convert keywords to string with weights by joining them with space
        doc = " ".join([' '.join([k] * int(w * 100)) for k, w in keywords])
        return self.vectorizer.transform([doc]).toarray()[0]

    def convert(self, labeled_docs_keywords: ILabeledDataset, verbose=True) -> ILabeledDataset:
        """
        TFIDFVectorlizer convert keywords to vector
        we use keyword as the token and weight as the weight of the token to generate the vector.
        """
        if verbose:
            print(f"[TFIDFVectorlizer] converting {len(labeled_docs_keywords)} docs to vectors using tf-idf")
        # Convert the input list of keywords with weights into a list of strings
        docs = []
        for keywords, label in labeled_docs_keywords:
            # convert keywords to string with weights by joining them with space
            doc = " ".join([' '.join([k] * int(w * 100)) for k, w in keywords])
            docs.append(doc)

        if verbose:
            print("tfidf vectorizer fitting...")
        # fit the vectorizer
        tfidf_matrix = self.vectorizer.fit_transform(docs)
        vectors = tfidf_matrix.toarray()
        if verbose:
            print("tfidf vectorizer fitted.")

        return LabelDataset(vectors, labeled_docs_keywords.labels)
