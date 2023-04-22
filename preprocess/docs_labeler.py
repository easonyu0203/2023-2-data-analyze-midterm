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


class FutureReturnDocsLabeler(IDocsLabeler):
    """
    FutureReturnDocsLabeler is a class that labels documents based on the future return percentage
    of the stock within a specified number of days, compared to a threshold value.
    """

    def __init__(self, s: int = 3, threshold: float = 0.1):
        """
        :param s: days into the future to look at the return percentage of the stock
        :param threshold: the threshold of the future return percentage of the stock to label a document as positive/negative
        """
        self.s = s
        self.threshold = threshold

    def label_documents(self, documents: IDocsDataset, stock: Stock, verbose=True) -> ILabeledDataset:
        """simply label document by the s day future return percentage of the stock."""
        if verbose:
            print("[FutureReturnDocsLabeler] labeling documents by the s day future return percentage of the stock...")

        # we don't want to alter original stock data
        stock_history = stock.history_df.copy()
        stock_history['future_return%'] = stock_history['close'].pct_change(self.s).shift(-self.s) * 100

        # label documents by the future return percentage of the stock
        labels = []
        documents_list = []
        date: pd.Timestamp
        p_bar = tqdm(stock_history.iterrows(), total=len(stock_history), desc="labeling documents", disable=not verbose)
        for date, stock_row in p_bar:
            # if the future return percentage of the stock is not greater than the threshold, we don't label it
            if abs(stock_row['future_return%']) <= self.threshold:
                continue
            # query documents that are posted within the same day as the stock
            queried_docs = documents.query_by_time(date, date + timedelta(days=1))
            future_return = stock_row['future_return%']
            # label documents by the future return percentage of the stock
            for document in queried_docs:
                labels.append(1 if future_return > 0 else 0)
                documents_list.append(document)
                
        if verbose:
            # print remaining count
            print(f"left with {len(labels)} documents after labeling")

        return LabelDataset(documents_list, labels)
