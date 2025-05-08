import pandas as pd
from data.process.discretizer import Discretizer
import os


class CorpusBuilder:
    def __init__(self, out_dir: str = "dataset/pickled", *args, **kwags):
        self.discretizer = None
        self.grouped = None
        self.out_dir = out_dir

    def build_corpus_from_df(
        self, df: pd.DataFrame, segment_length: int = 22, stride: int = 22
    ) -> pd.DataFrame:
        # Encode the OHLC columns to get token columns
        df = df[["id", "open_pct", "high_pct", "low_pct", "close_pct"]].copy()

        # Create token tuples from the encoded columns
        # tokens = list(
        #     zip(
        #         df["open_pct"],
        #         df["high_pct"],
        #         df["low_pct"],
        #         df["close_pct"],
        #     )
        # )
        df["token"] = list(
            zip(df["open_pct"], df["high_pct"], df["low_pct"], df["close_pct"])
        )

        # # Add tokens as a new column to the original df
        # df = df.copy()
        # df["token"] = tokens

        # # Group tokens by 'id' and aggregate tokens as list
        # grouped = df.groupby("id")["token"].apply(list).reset_index()

        # # Rename the tokens list column to 'text'
        # self.grouped = grouped.rename(columns={"token": "text"})

        # return self.grouped
        grouped = df.groupby("id")
        corpus = []

        for stock_id, group in grouped:
            tokens = group["token"].tolist()
            for i in range(0, len(tokens), stride):
                segment = tokens[i : i + segment_length]
                if len(segment) == segment_length:
                    corpus.append({"id": f"{stock_id}_{i//stride}", "text": segment})

        self.grouped = pd.DataFrame(corpus)

    def save_corpus(self, file_name: str):
        """
        Save the corpus to a pickled file.
        :param file_name: The name of the file to save the corpus to.
        """
        if self.grouped is not None:
            self.grouped.to_pickle(os.path.join(self.out_dir, file_name))
        else:
            raise ValueError(
                "Corpus has not been built yet. Please call build_corpus_from_df() first."
            )
