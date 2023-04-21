from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import Any

import pandas as pd

from datasets.docs_dataset import IDocsDataset
from datasets.stock_dataset import Stock
from preprocess.preprocess_pipeline import PreprocessPipeline, PreprocessPipeLineConfig


@dataclass
class BackTrackConfig:
    s: int  # the number of days in the future to calculate future return percentage
    docs_dataset: IDocsDataset  # the documents dataset
    start_date: datetime  # the start date of the backtest
    end_date: datetime  # the end date of the backtest
    train_span: timedelta  # the time span of the training set (day)
    inference_span: timedelta  # the time span of the inference set (day)
    preprocessConfig: PreprocessPipeLineConfig  # the preprocessing pipeline config
    take_shot_threshold: float  # the threshold of the take shot action (percentage)


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
            train_preprocess = PreprocessPipeline(self.config.preprocessConfig)
            train_dataset = train_preprocess.preprocess(train_docs, stock, verbose=False)

            # train the model
            X_train, y_train = zip(*train_dataset)
            model.fit(X_train, y_train)

            # get the inference dataset
            test_docs = self.config.docs_dataset.query_by_time(inference_start_date, inference_end_date)
            test_date = inference_start_date
            while test_date <= inference_end_date:
                # get the documents of the day
                single_date_docs = self.config.docs_dataset.query_by_time(test_date, test_date + timedelta(days=1))
                # transform to dataset by training preprocess pipeline
                single_date_dataset = [train_preprocess.single_doc_transform(doc) for doc in single_date_docs]
                # predict the future return percentage of the stock
                predictions = model.predict(single_date_dataset)
                # sum the predictions to get the predicted future return percentage of the stock
                pred_future_return = sum(predictions)
                # take shot
                shot = self.take_shot(pred_future_return, future_returnes[test_date])

                # record the result
                shot_count += 1 if shot != 0 else 0
                hit_count += 1 if shot == 1 else 0
                opportunity_count += 1 if abs(pred_future_return) > self.config.take_shot_threshold else 0

                # move to next day
                test_date += timedelta(days=1)
            # move to next training set
            train_start_date += step_size

        return BackTrackResult(shot_count, hit_count, opportunity_count)

    def get_future_return(self, stock: Stock) -> pd.Series:
        return stock.history_df['close'].pct_change(self.config.s).shift(-self.config.s) * 100

    def take_shot(self, pred_future_return: float, ground_truth: float) -> int:
        """consider whether to take shot or not
            return 1 if take correct shot, -1 if wrong, 0 if no action"""
        if abs(pred_future_return) <= self.config.take_shot_threshold:
            return 0

        # pred correct
        if pred_future_return * ground_truth > 0:
            return 1
        # pred wrong
        else:
            return -1
