from data.process.bucketizer import Bucketizer
from typing import Dict
from typing import List
import pandas as pd
from abc import ABC, abstractmethod


class Percentage(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DailyOHLCPercentage(Percentage):

    @staticmethod
    def process_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").copy()
        group["close_pct"] = (group["close"] - group["close"].shift(1)) / group[
            "close"
        ].shift(1)
        group["open_pct"] = (group["open"] - group["close"].shift(1)) / group[
            "close"
        ].shift(1)
        group["high_pct"] = (group["high"] - group["close"].shift(1)) / group[
            "close"
        ].shift(1)
        group["low_pct"] = (group["low"] - group["close"].shift(1)) / group[
            "close"
        ].shift(1)
        group["volume_pct"] = group["volume"] / group["volume"].shift(1)

        return group.iloc[1:][
            ["id", "date", "open_pct", "high_pct", "low_pct", "close_pct", "volume_pct"]
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process daily stock data:
        - Open, High, Low, Close relative to the previous day's closing price percentage (truncate to 2 decimal places)
        - Keep id and Date
        """
        result = df.groupby("id", group_keys=False).apply(self.process_group)
        return result.reset_index(drop=True)
