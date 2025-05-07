from data.process.bucketizer import Bucketizer
from typing import Dict


class FinancialTokenEncoder:
    def __init__(self, bucketizer: Bucketizer):
        self.bucketizer = bucketizer

    def encode_ohlc(self, open_pct, high_pct, low_pct, close_pct) -> str:
        return "".join(
            [
                format(self.bucketizer.encode(open_pct), "X"),
                format(self.bucketizer.encode(high_pct), "X"),
                format(self.bucketizer.encode(low_pct), "X"),
                format(self.bucketizer.encode(close_pct), "X"),
            ]
        )

    def label_ohlc(self, open_pct, high_pct, low_pct, close_pct) -> Dict[str, int]:
        return {
            "open": self.bucketizer.encode(open_pct),
            "high": self.bucketizer.encode(high_pct),
            "low": self.bucketizer.encode(low_pct),
            "close": self.bucketizer.encode(close_pct),
        }
