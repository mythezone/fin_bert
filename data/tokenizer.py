import pandas as pd
import numpy as np


def truncate_2(x):
    return np.floor(x * 10000) / 100  # 转为百分比后截断两位小数


def process_daily_stock_diff(input_path: str, output_path: str):
    """
    处理股票日频数据：
    - 开高低相对于前一日收盘价百分比（截断2位小数，乘100）
    - 收盘价为收盘环比（乘100）
    - 保留id和Date
    - Volume不变
    """
    df = pd.read_pickle(input_path)

    def process_group(group):
        group = group.sort_values("Date").copy()
        # group["Close_pct"] = group["Close"].pct_change() * 100
        group["Close_pct"] = (group["Close"] - group["Close"].shift(1)) / group[
            "Close"
        ].shift(1)
        group["Open_pct"] = (group["Open"] - group["Close"].shift(1)) / group[
            "Close"
        ].shift(1)
        group["High_pct"] = (group["High"] - group["Close"].shift(1)) / group[
            "Close"
        ].shift(1)
        group["Low_pct"] = (group["Low"] - group["Close"].shift(1)) / group[
            "Close"
        ].shift(1)
        group["Volume_pct"] = group["Volume"] / group["Volume"].shift(1)

        group["Open_pct"] = truncate_2(group["Open_pct"])
        group["High_pct"] = truncate_2(group["High_pct"])
        group["Low_pct"] = truncate_2(group["Low_pct"])
        group["Close_pct"] = truncate_2(group["Close_pct"])

        return group.iloc[1:][
            ["id", "Date", "Open_pct", "High_pct", "Low_pct", "Close_pct", "Volume_pct"]
        ]

    result = df.groupby("id", group_keys=False).apply(process_group)
    result.to_pickle(output_path)
    print(f"Saved processed data to {output_path}")


def tokenize_processed_data(input_path: str, output_path: str, k: int = 16):
    df = pd.read_pickle(input_path).copy()
    pct_cols = ["Open_pct", "High_pct", "Low_pct", "Close_pct"]

    edges = np.linspace(-10, 10, k + 1)  # k buckets => k+1 edges

    def tokenize_pct(val):
        return min(max(int(np.digitize(val, edges) - 1), 0), k - 1)

    for col in pct_cols:
        df[col] = df[col].apply(tokenize_pct)

    df["Volume_tok"] = df["Volume_pct"].apply(
        lambda x: int(np.round(np.log2(x))) if x > 0 else 0
    )
    df = df[["id", "Date"] + pct_cols + ["Volume_tok"]]
    df.to_pickle(output_path)
    print(f"Saved tokenized data to {output_path}")


def generate_financial_corpus(input_path: str, output_path: str):
    df = pd.read_pickle(input_path)
    words = df.apply(
        lambda row: "".join(
            [
                format(int(row[col]), "X")
                for col in ["Open_pct", "High_pct", "Low_pct", "Close_pct"]
            ]
        ),
        axis=1,
    )
    df["word"] = words
    corpus = df.groupby("id")["word"].apply(list).reset_index()
    corpus.rename(columns={"word": "text"}, inplace=True)
    corpus.to_pickle(output_path)
    print(f"Saved financial corpus to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/daily_df.pkl",
        help="Path to the input pkl file.",
    )
    parser.add_argument(
        "--processed_path",
        type=str,
        default="data/daily_df_processed.pkl",
        help="Path to save the processed pkl file.",
    )

    parser.add_argument(
        "--tokenized_path",
        type=str,
        default="data/tokenized/daily_df_tokenized_16.pkl",
        help="Path to save the tokenized pkl file.",
    )

    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="Whether to tokenize the processed data.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Number of buckets for percentage tokenization.",
    )
    parser.add_argument(
        "--build_corpus",
        action="store_true",
        help="Whether to generate financial corpus.",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data/tokenized/financial_corpus.pkl",
        help="Path to save corpus pkl.",
    )

    args = parser.parse_args()
    if args.tokenize:
        tokenize_processed_data(args.processed_path, args.tokenized_path, k=args.k)
        exit()
    if args.build_corpus:
        generate_financial_corpus(args.tokenized_path, args.corpus_path)
        exit()
    process_daily_stock_diff(args.input_path, args.processed_path)
