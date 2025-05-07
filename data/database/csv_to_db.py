import pandas as pd
from data.mysql import create_mysql_engine


def process_custom_csv(file_path):
    df = pd.read_csv(file_path)
    db = create_mysql_engine()

    for code in df["Code"].unique():
        sub_df = df[df["Code"] == code].copy()

        # 构建时间索引
        sub_df["datetime"] = pd.to_datetime(
            sub_df["Date"].astype(str) + sub_df["Time"].astype(str).str.zfill(6),
            format="%Y%m%d%H%M%S",
        )
        sub_df.set_index("datetime", inplace=True)

        # 保留字段
        sub_df = sub_df[
            ["Open", "High", "Low", "Close", "Volume", "Turnover", "MatchItems"]
        ]

        # 写入数据库，表名为 code，如 '000001.SH1'
        sub_df.to_sql(name=code.lower(), con=db, if_exists="append", index=True)
