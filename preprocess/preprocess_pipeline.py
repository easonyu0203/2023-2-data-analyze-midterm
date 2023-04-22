"""
This module contains the problem spec preprocess class. which follow our problem specification.
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
from typing import List

from datasets.docs_dataset import IDocsDataset
from datasets.labeled_docs_dataset import ILabeledDataset
from datasets.stock_dataset import Stock
from preprocess.docs_filterer import IDocsFilterer
from preprocess.docs_labeler import IDocsLabeler
from preprocess.keyword_extractor import IKeywordExtractor
from preprocess.vectorlizer import IVectorlizer


class PreprocessPipeline:

    def __init__(self,
                 docs_filterer: IDocsFilterer,
                 docs_labeler: IDocsLabeler,
                 keywords_extractor: IKeywordExtractor,
                 vectorizer: IVectorlizer,
                 ):
        self.docs_filterer = docs_filterer
        self.docs_labeler = docs_labeler
        self.keywords_extractor = keywords_extractor
        self.vectorizer = vectorizer

    def preprocess(self, docs_dataset: IDocsDataset, stock: Stock, do_fit: bool = True,
                   verbose=True) -> ILabeledDataset:
        """main preprocess pipe line function"""

        self.print_line(verbose=verbose)
        # filter documents
        if do_fit:
            self.docs_filterer.fit(docs_dataset, stock, verbose=verbose)
        filtered_docs = self.docs_filterer.filter_documents(docs_dataset, stock, verbose=verbose)
        self.print_line(verbose=verbose, space_down=2)

        self.print_line(verbose=verbose)
        # label documents
        docs_dataset = self.docs_labeler.label_documents(filtered_docs, stock, verbose=verbose)
        self.print_line(verbose=verbose, space_down=2)

        self.print_line(verbose=verbose)
        # extract keywords
        keyword_docs_dataset = self.keywords_extractor.extract_keywords(docs_dataset, verbose=verbose)
        self.print_line(verbose=verbose, space_down=2)

        self.print_line(verbose=verbose)
        # fit vectorizer
        if do_fit:
            self.vectorizer.fit(keyword_docs_dataset, verbose=verbose)
        # convert keywords to vectors
        vector_dataset = self.vectorizer.transform(keyword_docs_dataset, verbose=verbose)
        self.print_line(verbose=verbose, space_down=2)

        return vector_dataset

    def print_line(self, verbose=True, space_down=0):
        if verbose:
            print("=" * 50)

            for _ in range(space_down):
                print()
