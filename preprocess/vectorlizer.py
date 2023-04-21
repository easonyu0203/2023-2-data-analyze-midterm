from typing import Protocol, Tuple, List

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from datasets.labeled_docs_dataset import ILabeledDataset, LabelDataset


class IVectorlizer(Protocol):
    """IVectorlizer convert keywords to vector"""

    def convert(self, labeled_docs_keywords: ILabeledDataset, verbose=True) -> ILabeledDataset:
        """convert keywords data set to vector dataset"""
        ...

    def transform(self, keywords: List[Tuple[str, float]]) -> List[float]:
        """transform a list of keywords and their weights to a vector"""
        ...


class BaselineVectorlizer(IVectorlizer):
    """
    BaselineVectorlizer convert keywords to vector
    1. we use count vectorizer to convert keywords to vector
    2. use PCA to reduce the dimension of the vector
    3. feed in to tf-idf transformer to get the final vector
    """

    def __init__(self, count_features=1000, pca_components=100):
        self.count_vectorizer = CountVectorizer(max_features=count_features)
        self.pca = PCA(n_components=pca_components)
        self.tfidf_transformer = TfidfTransformer()

    def convert(self, labeled_docs_keywords: ILabeledDataset, verbose=True) -> ILabeledDataset:
        """
        convert keywords to vector by count vectorizer, pca and tf-idf transformer
        we use keyword as the token and weight as the weight of the token to generate the vector.
        """
        if verbose:
            print(f"[BaselineVectorlizer] converting {len(labeled_docs_keywords)} docs to vectors")

        # Convert the input list of keywords with weights into a list of strings
        docs = []
        for keywords, label in labeled_docs_keywords:
            # convert keywords to string with weights by joining them with space
            doc = " ".join([' '.join([k] * int(w * 100)) for k, w in keywords])
            docs.append(doc)

        if verbose:
            print("transformers fitting...")

        # fit and transform the CountVectorizer object to the list of documents
        X_counts = self.count_vectorizer.fit_transform(docs)

        # use pca to reduce the dimension of the vector
        X_counts = self.pca.fit_transform(X_counts.toarray())

        # transform the CountVectorizer output to a TF-IDF weighted matrix
        X_tfidf = self.tfidf_transformer.fit_transform(X_counts)

        vectors = X_tfidf.toarray()
        if verbose:
            print("vectorizer fitted.")

        return LabelDataset(vectors, labeled_docs_keywords.labels)

    def transform(self, keywords: List[Tuple[str, float]]) -> List[float]:
        # convert keywords to string with weights by joining them with space
        doc = " ".join([' '.join([k] * int(w * 100)) for k, w in keywords])
        x_counts = self.count_vectorizer.transform([doc])
        x_tfidf = self.tfidf_transformer.transform(x_counts)
        vec = x_tfidf.toarray()[0]

        return vec
