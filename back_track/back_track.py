from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import Any, List

import pandas as pd

from datasets.docs_dataset import IDocsDataset
from datasets.stock_dataset import Stock
from preprocess.preprocess_pipeline import PreprocessPipeline


@dataclass
class BackTrackConfig:
    s: int  # the number of days in the future to calculate future return percentage
    docs_dataset: IDocsDataset  # the documents dataset
    start_date: datetime  # the start date of the backtest
    end_date: datetime  # the end date of the backtest
    train_span: timedelta  # the time span of the training set (day)
    inference_span: timedelta  # the time span of the inference set (day)
    preprocess_pipeline: PreprocessPipeline  # the preprocessing pipeline config
    take_shot_threshold: float  # the threshold of the take shot action (percentage) ex. if 6 pos 4 neg => (6 - 4) / 10 = 0.2
    return_no_move_threshold: float  # the threshold that below this we consider the return is no move (percentage)


@dataclass
class BackTrackResult:
    """the result of the backtest"""
    shot_count: int
    hit_count: int
    opportunity_count: int


class BackTrack:

    def __init__(self, config: BackTrackConfig):
        self.config = config

    def run(self, stock: Stock, model: Any):
        """run back-test for given stock and model"""

        future_returnes = self.get_future_return(stock)

        shot_count = 0
        hit_count = 0
        opportunity_count = 0
        step_size = self.config.inference_span
        train_start_date = self.config.start_date
        while True:
            train_end_date = train_start_date + self.config.train_span
            inference_start_date = train_end_date
            inference_end_date = train_end_date + self.config.inference_span

            # check if the end date of the inference set is greater than the end date of the backtest
            if inference_end_date > self.config.end_date:
                break

            # get the training dataset
            train_docs = self.config.docs_dataset.query_by_time(train_start_date, train_end_date)
            val_docs = self.config.docs_dataset.query_by_time(inference_start_date, inference_end_date)
            # preprocess dataset
            processed_train_dataset = self.config.preprocess_pipeline.preprocess(train_docs, stock, do_fit=True,
                                                                                 verbose=False)
            processed_val_dataset = self.config.preprocess_pipeline.preprocess(val_docs, stock, do_fit=False,
                                                                               verbose=False)

            # train the model
            X_train, y_train = zip(*processed_train_dataset)
            model.fit(X_train, y_train)

            # validate the model
            test_date = inference_start_date
            while test_date <= inference_end_date:
                # check have future return percentage of the stock
                if test_date not in future_returnes:
                    # move to next day
                    test_date += timedelta(days=1)
                # get the documents of the day
                single_date_docs = self.config.docs_dataset.query_by_time(test_date, test_date + timedelta(days=1))
                # preprocess the documents
                processed_val_dataset = self.config.preprocess_pipeline.preprocess(val_docs, stock, do_fit=False,
                                                                                   verbose=False)
                # predict the future return percentage of the stock
                X_val, y_val = zip(*processed_val_dataset)
                predictions = model.predict(X_val)
                pred_probas = model.predict_proba(X_val)
                # take shot
                shot = self.take_shot(predictions, pred_probas, future_returnes[test_date])

                # record the result
                shot_count += 1 if shot != 0 else 0
                hit_count += 1 if shot == 1 else 0
                opportunity_count += 1 if abs(future_returnes[test_date]) > self.config.return_no_move_threshold or shot else 0

                # move to next day
                test_date += timedelta(days=1)
            # move to next training set
            train_start_date += step_size

        return BackTrackResult(shot_count, hit_count, opportunity_count)

    def get_future_return(self, stock: Stock) -> pd.Series:
        return stock.history_df['close'].pct_change(self.config.s).shift(-self.config.s) * 100

    def take_shot(self, pred_future_return: List[float], pred_probas: List[float], ground_truth: float) -> int:
        """consider whether to take shot or not
            return 1 if take correct shot, -1 if wrong, 0 if no action"""
        if abs(sum(pred_future_return)/len(pred_future_return)) <= self.config.take_shot_threshold:
            return 0

        # pred correct
        if sum(pred_future_return) * ground_truth > 0:
            return 1
        # pred wrong
        else:
            return -1


from datasets.docs_dataset import DocsDataset
from datasets.stock_dataset import StockMeta

docs_dataset = DocsDataset()
stock_meta = StockMeta("./organized_data/stock_metadata.csv")
from preprocess.preprocess_pipeline import PreprocessPipeline
from preprocess.docs_filterer import IDocsFilterer, StockNameFilterer, Word2VecSimilarFilterer
from preprocess.docs_labeler import IDocsLabeler, FutureReturnDocsLabeler
from preprocess.keyword_extractor import IKeywordExtractor, JiebaKeywordExtractor
from preprocess.vectorlizer import IVectorlizer, KeywordsTfIdfVectorlizer
from utils.data import random_split_train_val
from sklearn.linear_model import LogisticRegression
from sklearn_model_process.train_eval_model import train_eval_model, display_evaluation_result

# set up config
stock = stock_meta.get_stock_by_name("台積電")
clf = LogisticRegression()
preprocess_pipeline = PreprocessPipeline(
    docs_filterer=Word2VecSimilarFilterer(topn=5, white_noise_ratio=0),
    docs_labeler=FutureReturnDocsLabeler(s=3, threshold=5),
    keywords_extractor=JiebaKeywordExtractor(),
    vectorizer=KeywordsTfIdfVectorlizer(count_features=1000, pca_components=100)
)

config = BackTrackConfig(
    s=3,
    docs_dataset=docs_dataset,
    start_date=datetime(2019, 1, 10),
    end_date=datetime(2023, 1, 10),
    train_span=timedelta(days=90),
    inference_span=timedelta(days=30),
    take_shot_threshold=0.1,
    return_no_move_threshold=0.1,
    preprocess_pipeline=preprocess_pipeline
)

# run backtest
backtrack = BackTrack(config)
result = backtrack.run(stock, clf)
print(result)