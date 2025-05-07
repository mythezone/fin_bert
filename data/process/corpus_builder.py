import pandas as pd
from data.process.token_encoder import FinancialTokenEncoder


class CorpusBuilder:
    def __init__(self, encoder: FinancialTokenEncoder):
        self.encoder = encoder

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in df.iterrows():
            try:
                token = self.encoder.encode_ohlc(
                    row["Open"], row["High"], row["Low"], row["Close"]
                )
                label = self.encoder.label_ohlc(
                    row["Open"], row["High"], row["Low"], row["Close"]
                )
                records.append(
                    {
                        "id": row["id"],
                        "token": token,
                        "label": label,
                        "date": row["Date"],
                    }
                )
            except Exception as e:
                continue  # optional logging here
        return pd.DataFrame(records)
