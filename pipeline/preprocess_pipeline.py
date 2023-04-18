"""
This module contains the problem spec pipeline class. which follow our problem specification.
1. filter documents: remove documents that aren't relevant to the stock we are interested in.
2. label documents: label documents by the future return percentage of the stock.
3. extract keywords: extract keywords from documents.
4. document's keywords to vectors: convert document's keywords to vectors.
5. split data: split the data into training and testing sets.
6. train model: train a supervised learning model to predict the future return percentage of the stock.
7. evaluate model: evaluate the model's performance.
8. backtest: backtest the model's performance.
"""
from dataclasses import dataclass

from datasets.docs_dataset import IDocsDataset
from datasets.labeled_docs_dataset import ILabeledDataset
from datasets.stock_dataset import Stock
from pipeline.docs_filterer import IDocsFilterer
from pipeline.docs_labeler import IDocsLabeler
from pipeline.keyword_extractor import IKeywordExtractor
from pipeline.labeled_docs_filterer import ILabeledDocsFilterer
from pipeline.vectorlizer import IVectorlizer


@dataclass
class PreprocessPipeLineConfig:
    docs_dataset: IDocsDataset
    stock: Stock
    docs_filterer: IDocsFilterer
    docs_labeler: IDocsLabeler
    labeled_docs_filterer: ILabeledDocsFilterer
    keywords_extractor: IKeywordExtractor
    vectorizer: IVectorlizer


class PreprocessPipeline:

    def __init__(self, config: PreprocessPipeLineConfig):
        self.config = config

    def preprocess(self, verbose=True) -> ILabeledDataset:
        # filter documents
        filtered_docs = self.config.docs_filterer.filter_documents(self.config.docs_dataset, self.config.stock,
                                                                   verbose=verbose)

        # label documents
        labeled_docs = self.config.docs_labeler.label_documents(filtered_docs, self.config.stock, verbose=verbose)

        # filter labeled documents
        filtered_labeled_docs = self.config.labeled_docs_filterer.filter_documents(labeled_docs, verbose=verbose)

        # extract keywords
        labeled_docs_keywords = self.config.keywords_extractor.extract_keywords(filtered_labeled_docs, verbose=verbose)

        # convert keywords to vectors
        labeled_docs_vectors = self.config.vectorizer.convert(labeled_docs_keywords, verbose=verbose)

        return labeled_docs_vectors
