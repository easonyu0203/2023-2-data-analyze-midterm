from typing import List
import pandas as pd


class StockMeta:
    """StockMeta is a class that contains all the stock metadata.
    It can be used to get a stock by its name or id."""
    stock_names: List[str]
    stock_ids: List[str]

    def __init__(self, stock_meta_path: str):
        stock_meta_df = pd.read_csv(stock_meta_path)
        self.stock_ids = list(stock_meta_df['id'])
        self.stock_names = list(stock_meta_df['name'])
        self._stock_meta_df = stock_meta_df

    def get_stock_by_name(self, stock_name: str):
        stock_row = self._stock_meta_df[self._stock_meta_df['name'] == stock_name]
        stock_id = stock_row['id'].values[0]
        stock_history_path = stock_row['history_path'].values[0]
        return Stock(stock_name, stock_id, stock_history_path)

    def get_stock_by_id(self, stock_id: str):
        stock_row = self._stock_meta_df[self._stock_meta_df['id'] == stock_id]
        stock_name = stock_row['name'].values[0]
        stock_history_path = stock_row['history_path'].values[0]
        return Stock(stock_name, stock_id, stock_history_path)

    def __str__(self):
        return f"StockMeta(stock_count={len(self.stock_names)})"

    def __repr__(self):
        return f"StockMeta(stock_count={len(self.stock_names)})"


class Stock:
    """Stock is a class that contains a stock's name, id, and history."""
    name: str
    id: str
    history_df: pd.DataFrame

    def __init__(self, stock_name: str, stock_id: str, history_path: str):
        self.name = stock_name
        self.id = stock_id
        self.history_df = pd.read_csv(history_path)
        self.history_df['date'] = pd.to_datetime(self.history_df['date'])
        self.history_df.set_index('date', inplace=True)

    def __repr__(self):
        return f"Stock(name={self.name}, id={self.id})"


if __name__ == "__main__":
    stock_meta = StockMeta("../organized_data/stock_metadata.csv")
    stock = stock_meta.get_stock_by_name("富邦科技")
    print(stock)
