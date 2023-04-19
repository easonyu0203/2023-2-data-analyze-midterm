from datetime import timedelta
from typing import Protocol, List, Tuple, Hashable
import pandas as pd
from tqdm import tqdm
from datasets.docs_dataset import IDocsDataset, DocsDataset
from datasets.stock_dataset import Stock
from datasets.labeled_docs_dataset import ILabeledDataset, LabelDataset


class IDocsLabeler(Protocol):
    """Label documents to corresponding to the impact of future return percentage of the stock."""
    def label_documents(self, documents: IDocsDataset, stock: Stock, verbose=True) -> ILabeledDataset:
        """label documents to corresponding to the impact of future return percentage of the stock."""
        ...


class DefaultDocsLabeler(IDocsLabeler):
    """simply label document by the s day future return percentage of the stock."""

    def __init__(self, s: int):
        """s is the number of days in the future."""
        self.s = s

    def label_documents(self, documents: IDocsDataset, stock: Stock, verbose=True) -> ILabeledDataset:
        """simply label document by the s day future return percentage of the stock."""
        if verbose:
            print("[DefaultDocsLabeler] labeling documents by the s day future return percentage of the stock...")

        # we don't want to alter original stock data
        stock_history = stock.history_df.copy()
        stock_history['future_return%'] = stock_history['close'].pct_change(self.s).shift(-self.s) * 100

        # label documents by the future return percentage of the stock
        labels = []
        documents_list = []
        date: pd.Timestamp
        p_bar = tqdm(stock_history.iterrows(), desc="labeling documents", disable=not verbose)
        for date, stock_row in p_bar:
            # query documents that are posted within the same day as the stock
            queried_docs = documents.query_by_time(date, date + timedelta(days=1))
            future_return = stock_row['future_return%']
            # label documents by the future return percentage of the stock
            for document in queried_docs:
                labels.append(future_return)
                documents_list.append(document)

        return LabelDataset(documents_list, labels)
