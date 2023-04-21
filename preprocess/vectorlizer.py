from typing import Protocol, Tuple, List

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from datasets.labeled_docs_dataset import ILabeledDataset, LabelDataset


class IVectorlizer(Protocol):
    """IVectorlizer convert keywords to vector"""

    def fit(self, labeled_dataset: ILabeledDataset, verbose=True):
        """fit the vectorizer"""
        ...

    def transform(self, labeled_dataset: ILabeledDataset, verbose=True) -> ILabeledDataset:
        """transform the labeled dataset to vector's labeled dataset"""
        ...


class KeywordsTfIdfVectorlizer(IVectorlizer):
    """
    KeywordsTfIdfVectorlizer convert keywords to vector with respect to keywords' weights and using tf-idf.
    the specific steps are:
    1. we use count vectorizer to convert keywords to vector
    2. use PCA to reduce the dimension of the vector
    3. feed in to tf-idf transformer to get the final vector
    """

    def __init__(self, count_features=1000, pca_components=100):
        self.count_vectorizer = CountVectorizer(max_features=count_features)
        self.pca = PCA(n_components=pca_components)
        self.tfidf_transformer = TfidfTransformer()

    def fit(self, labeled_dataset: ILabeledDataset, verbose=True):
        """fit the vectorizer"""
        if verbose:
            print(f"[KeywordsTfIdfVectorlizer] fitting {len(labeled_dataset)} docs")

        # Convert the input list of keywords with weights into a list of strings
        keywords_docs = self._labeled_dataset_to_weighted_keywords_list(labeled_dataset)

        # fit and transform the CountVectorizer object to the list of documents
        X_counts = self.count_vectorizer.fit_transform(keywords_docs)
        X_counts_pca = self.pca.fit_transform(X_counts.toarray())
        self.tfidf_transformer.fit(X_counts_pca)
        if verbose:
            print("vectorizer fitted.")

    def transform(self, labeled_dataset: ILabeledDataset, verbose=True) -> ILabeledDataset:
        """transform the labeled dataset to vector's labeled dataset"""
        if verbose:
            print(f"[KeywordsTfIdfVectorlizer] transforming {len(labeled_dataset)} docs to vectors")

        # Convert the input list of keywords with weights into a list of strings
        keywords_docs = self._labeled_dataset_to_weighted_keywords_list(labeled_dataset)

        # transform the CountVectorizer output to a TF-IDF weighted matrix
        X_counts = self.count_vectorizer.transform(keywords_docs)
        X_counts_pca = self.pca.transform(X_counts.toarray())
        X_tfidf = self.tfidf_transformer.transform(X_counts)
        vectors = X_tfidf.toarray()

        return LabelDataset(vectors, labeled_dataset.labels)

    def _labeled_dataset_to_weighted_keywords_list(self, labeled_dataset: ILabeledDataset) -> List[str]:
        """convert labeled dataset to list of weighted keywords"""
        docs = []
        for keywords, label in labeled_dataset:
            # convert keywords to string with weights by joining them with space
            doc = " ".join([' '.join([k] * int(w * 100)) for k, w in keywords])
            docs.append(doc)
        return docs
