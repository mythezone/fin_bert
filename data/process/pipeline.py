import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from data.process.bucketizer import Bucketizer
from data.process.corpus_builder import CorpusBuilder
from data.process.token_encoder import FinancialTokenEncoder


class CorpusPipeline:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        bucketizer: Bucketizer,
        verbose: bool = True,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.encoder = FinancialTokenEncoder(bucketizer)
        self.builder = CorpusBuilder(self.encoder)
        self.verbose = verbose

    def run(self):
        if self.verbose:
            print(f"📥 Loading raw data from {self.input_path}")

        df = (
            pd.read_pickle(self.input_path)
            if self.input_path.endswith(".pkl")
            else pd.read_csv(self.input_path)
        )
        processed = self.builder.process_dataframe(df)

        if self.verbose:
            print(
                f"✅ Processed {len(processed)} records. Saving to {self.output_path}"
            )

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        processed.to_pickle(self.output_path)

    def visualize_token_distribution(self, processed_df: Optional[pd.DataFrame] = None):
        if processed_df is None:
            processed_df = pd.read_pickle(self.output_path)
        token_counts = processed_df["token"].value_counts().head(50)
        token_counts.plot(
            kind="barh", title="Top 50 Token Frequencies", figsize=(10, 8)
        )
        plt.xlabel("Frequency")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def log_sample(self, n: int = 5):
        df = pd.read_pickle(self.output_path)
        print(df.head(n).to_string(index=False))
