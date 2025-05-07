import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm  # 添加 import
from datetime import datetime, timedelta
from multiprocessing import Pool


def list_stocks_for_date(root_dir: str, date: str) -> List[str]:
    year = date[:4]
    month = date[:6]
    day = date
    folder = Path(root_dir) / year / month / day
    if not folder.exists():
        return []
    codes = set()
    for file in folder.glob("*.CSV"):
        parts = file.stem.split(".")
        if len(parts) != 2:
            continue
        code = parts[0]
        market = parts[1].split("-")[0]
        codes.add(f"{code}.{market}")
    return sorted(codes)


def tokenize_stock_data(
    root_dir: str, k: int = 8, output_file: str = "tokenized_data.csv"
):
    bucket_edges = np.linspace(-0.1, 0.1, 2 * k + 1)  # 生成2k个区间边界，中心为0

    def discretize(pct: float) -> int:
        """将百分比映射到离散整数桶编号，范围为[-k, k]"""
        if pct <= bucket_edges[0]:
            return -k
        elif pct >= bucket_edges[-1]:
            return k
        return int(np.digitize(pct, bucket_edges)) - k

    results = []
    files_to_process = []

    # 构造 2014 年所有日期
    start_date = datetime(2014, 1, 1)
    end_date = datetime(2014, 12, 31)

    while start_date <= end_date:
        date_str = start_date.strftime("%Y%m%d")
        year = date_str[:4]
        month = date_str[:6]
        day = date_str
        file_path = (
            Path(root_dir) / year / month / day / f"000001.SZ1-1M-{date_str}-L1.CSV"
        )
        if file_path.exists():
            files_to_process.append(file_path)
        start_date += timedelta(days=1)

    # 处理文件
    for file in tqdm(sorted(files_to_process)):
        code = file.stem.split(".")[0]
        market = file.stem.split(".")[1].split("-")[0]
        date = file.stem.split("-")[2]
        df = pd.read_csv(file)

        df = df[["Open", "High", "Low", "Close"]].copy()
        close_pct = df["Close"].pct_change().fillna(0)
        open_pct = (df["Open"] - df["Close"]) / df["Close"]
        high_pct = (df["High"] - df["Close"]) / df["Close"]
        low_pct = (df["Low"] - df["Close"]) / df["Close"]

        discretized = []
        for o, h, l, c in zip(open_pct, high_pct, low_pct, close_pct):
            tokens = [discretize(p) for p in [o, h, l, c]]
            hex_tokens = "".join([format((t + k) % (2 * k + 1), "X") for t in tokens])
            discretized.append(hex_tokens)

        seq_id = f"{code}.{market}-1M-{date}"
        results.append((seq_id, discretized))

    df_out = pd.DataFrame(results, columns=["id", "tokens"])
    df_out["tokens"] = df_out["tokens"].apply(lambda x: ",".join(x))
    df_out.to_csv(output_file, index=False)

    print(f"Processed {len(results)} sequences. Saved to {output_file}")


def process_one_stock_year(args: Tuple[str, str, str, str]):
    code_market, root_dir, year, output_dir = args
    k = 8
    bucket_edges = np.linspace(-0.1, 0.1, 2 * k + 1)

    def discretize(pct: float) -> int:
        if pct <= bucket_edges[0]:
            return -k
        elif pct >= bucket_edges[-1]:
            return k
        return int(np.digitize(pct, bucket_edges)) - k

    results = []
    start_date = datetime(int(year), 1, 1)
    end_date = datetime(int(year), 12, 31)
    code, market = code_market.split(".")

    while start_date <= end_date:
        date_str = start_date.strftime("%Y%m%d")
        month = date_str[:6]
        path = (
            Path(root_dir)
            / year
            / month
            / date_str
            / f"{code}.{market}-1M-{date_str}-L1.CSV"
        )
        if path.exists():
            try:
                df = pd.read_csv(path)
                df = df[["Open", "High", "Low", "Close"]].copy()
                close_pct = df["Close"].pct_change().fillna(0)
                open_pct = (df["Open"] - df["Close"]) / df["Close"]
                high_pct = (df["High"] - df["Close"]) / df["Close"]
                low_pct = (df["Low"] - df["Close"]) / df["Close"]

                discretized = []
                for o, h, l, c in zip(open_pct, high_pct, low_pct, close_pct):
                    tokens = [discretize(p) for p in [o, h, l, c]]
                    hex_tokens = "".join(
                        [format((t + k) % (2 * k + 1), "X") for t in tokens]
                    )
                    discretized.append(hex_tokens)

                seq_id = f"{code}.{market}-1M-{date_str}"
                results.append((seq_id, discretized))
            except Exception as e:
                print(f"Failed to process {path}: {e}")
        start_date += timedelta(days=1)

    if results:
        df_out = pd.DataFrame(results, columns=["id", "tokens"])
        df_out["tokens"] = df_out["tokens"].apply(lambda x: ",".join(x))
        out_path = Path(output_dir) / f"{code}.{market}-{year}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Saved {len(results)} entries to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multi_stock",
        action="store_true",
        help="Enable multi-stock yearly processing.",
    )
    parser.add_argument("--year", type=str, default="2014", help="Year to process.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="stock_tokens",
        help="Directory to store per-stock output.",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default="/Volumes/ashare/market_data/jydata_unzip",
        help="Root directory for the data files.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of buckets for discretization.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="tokenized_data.csv",
        help="Output file for tokenized data.",
    )
    parser.add_argument(
        "--scan_only",
        action="store_true",
        help="Only scan and list unique stock codes for 2014.",
    )
    args = parser.parse_args()

    if args.multi_stock:
        stock_list = pd.read_csv("data/sample/all_stocks_20140102.csv")[
            "code.market"
        ].tolist()
        os.makedirs(args.output_dir, exist_ok=True)
        args_list = [
            (code, args.root_dir, args.year, args.output_dir) for code in stock_list
        ]
        with Pool(processes=os.cpu_count()) as pool:
            pool.map(process_one_stock_year, args_list)
        exit()

    if args.scan_only:
        codes = list_stocks_for_date(args.root_dir, "20140102")
        pd.DataFrame(sorted(codes), columns=["code.market"]).to_csv(
            "all_stocks_20140102.csv", index=False
        )
        print("Saved unique stock codes for 20140102 to all_stocks_20140102.csv")
        exit()

    tokenize_stock_data(args.root_dir, args.k, args.output_file)
