from data.process.bucketizer import Bucketizer
from typing import Dict
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data.process.bucketizer import ExponentialBucketizer


class FinancialTokenEncoder:
    def __init__(self, bucketizer: Bucketizer = None):

        self.encoded_df = None
        self.bucketizer = bucketizer

    def set_bucketizer(self, bucketizer: Bucketizer):
        self.bucketizer = bucketizer

    def encode_ohlc(self, open_pct, high_pct, low_pct, close_pct) -> List[int]:
        self.encoded_df = pd.DataFrame(
            {
                "open_pct": self.bucketizer.encode(open_pct),
                "high_pct": self.bucketizer.encode(high_pct),
                "low_pct": self.bucketizer.encode(low_pct),
                "close_pct": self.bucketizer.encode(close_pct),
            }
        )
        return self.encoded_df

    def encode_df(self, df):

        self.encoded_df = pd.DataFrame(
            {
                "open_pct": self.bucketizer.encode(df["open_pct"]),
                "high_pct": self.bucketizer.encode(df["high_pct"]),
                "low_pct": self.bucketizer.encode(df["low_pct"]),
                "close_pct": self.bucketizer.encode(df["close_pct"]),
            }
        )
        return self.encoded_df

    def encode_csv(self, csv_file: str) -> List[List[int]]:
        df = pd.read_csv(csv_file)
        return self.encode_df(df)

    def plot_bucket_distribution(self, subplot=True):

        all_buckets = self.encoded_df.values.flatten()
        bucket_counts = pd.Series(all_buckets).value_counts().sort_index()
        ax = plt.gca() if subplot else plt.figure().gca()
        bucket_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Bucket Index")
        ax.set_ylabel("Count")
        ax.set_title("Bucket Distribution")

        if subplot:
            return ax
        else:
            plt.tight_layout()
            plt.show()

    def plot_zero_e_grid(self, df: pd.DataFrame = None):
        k = 10
        zeros = np.linspace(1e-5, 1e-4, k)
        es = np.linspace(1.1, 1.2, k)

        fig, axes = plt.subplots(k, k, figsize=(20, 20))
        for i, zero in enumerate(zeros):
            for j, e in enumerate(es):
                self.bucketizer.set_arg(zero=zero, e=e)
                df = self.encode_df(df)
                ax = axes[i, j]
                all_buckets = df.values.flatten()
                bucket_counts = pd.Series(all_buckets).value_counts().sort_index()
                bucket_counts.plot(kind="bar", ax=ax)
                ax.set_title(f"zero={zero:.1e}\ne={e:.2f}")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.label_outer()
        plt.tight_layout()
        plt.show()
