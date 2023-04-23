from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import Any, List

import pandas as pd
from tqdm import tqdm

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


@dataclass
class BackTrackResult:
    """the result of the backtest"""
    shot_count: int
    hit_count: int
    opportunity_count: int
    test_date_to_action: pd.DataFrame  # the date to action mapping

class BackTrack:

    def __init__(self, config: BackTrackConfig):
        self.config = config



    def run(self, stock: Stock, model: Any, verbose=True):
        """run back-test for given stock and model"""

        future_returnes = self.get_future_return(stock)

        shot_count = 0
        hit_count = 0
        opportunity_count = 0
        step_size = self.config.inference_span
        train_start_date = self.config.start_date
        test_date_to_action = {}
        if verbose:
            print(f"backtest start date: {train_start_date}")
            print(f"backtest end date: {self.config.end_date}")
            print(f"train span: {self.config.train_span}")
            print(f"inference span: {self.config.inference_span}")
            print("=" * 50)

        while True:

            train_end_date = train_start_date + self.config.train_span
            inference_start_date = train_end_date
            inference_end_date = train_end_date + self.config.inference_span

            if verbose:
                print(f"\n\ntrain span: {train_start_date} => {train_end_date}")

            # check if the end date of the inference set is greater than the end date of the backtest
            if inference_end_date > self.config.end_date:
                print('backtest end')
                break

            # get the training dataset
            train_docs = self.config.docs_dataset.query_by_time(train_start_date, train_end_date)
            # preprocess dataset
            processed_train_dataset = self.config.preprocess_pipeline.preprocess(train_docs, stock, do_fit=True,
                                                                                 verbose=False)

            # train the model
            try:
                if len(processed_train_dataset) > 0:
                    X_train, y_train = zip(*processed_train_dataset)
                    model.fit(X_train, y_train)
            except ValueError:
                pass

            # validate the model
            test_date = inference_start_date
            if verbose:
                pass
                print(f"\ninference span: {test_date} => {inference_end_date}")

            pbar = tqdm(total=(inference_end_date - test_date).days)
            while test_date <= inference_end_date:
                # check have future return percentage of the stock
                if test_date not in future_returnes:
                    # move to next day
                    test_date += timedelta(days=1)
                    pbar.update(1)
                    continue
                if verbose:
                    pbar.desc = f"test date: {test_date}"
                # get the documents of the day
                single_date_docs = self.config.docs_dataset.query_by_time(test_date, test_date + timedelta(days=1))
                # preprocess the documents
                processed_val_dataset = self.config.preprocess_pipeline.preprocess(single_date_docs, stock, do_fit=False,
                                                                                   verbose=False)
                # predict the future return percentage of the stock
                try:
                    X_val, y_val = zip(*processed_val_dataset)
                    predictions = model.predict(X_val)
                    pred_probas = model.predict_proba(X_val)
                    # take shot
                    shot = self.take_shot(predictions, pred_probas, future_returnes[test_date])
                except ValueError:
                    # print(f"no doc for {test_date}")
                    shot = 0

                # record the result
                shot_count += 1 if shot != 0 else 0
                hit_count += 1 if shot == 1 else 0
                opportunity_count += 1


                if shot == 1:
                    test_date_to_action[test_date] = 'hit'
                elif shot == -1:
                    test_date_to_action[test_date] = 'miss'
                else:
                    test_date_to_action[test_date] = 'no action'

                # move to next day
                pbar.update(1)
                test_date += timedelta(days=1)
            # move to next training set
            train_start_date += step_size
            del pbar

        # convert test_date_to_action to dataframe
        test_date_to_action = pd.DataFrame.from_dict(test_date_to_action, orient='index', columns=['action'])
        return BackTrackResult(shot_count, hit_count, opportunity_count, test_date_to_action)

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

