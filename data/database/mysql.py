import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import pandas as pd


def create_mysql_engine() -> Engine:
    load_dotenv()
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db = os.getenv("DB_NAME")

    if not all([user, password, host, port, db]):
        raise ValueError("数据库配置不完整，请检查 .env 文件")

    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, pool_recycle=3600)


class FileProcessor:
    def __init__(self):
        self.engine = create_mysql_engine()

    def process_csv(self, file_path: str):
        df = pd.read_csv(file_path)
        code = df["Code"].unique()[0]
        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str) + df["Time"].astype(str).str.zfill(6),
            format="%Y%m%d%H%M%S",
        )
        df.set_index("datetime", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume", "Turnover", "MatchItems"]]

        # 表名建议替换掉点

        table_name = code.lower().replace(".", "_")
        df.to_sql(
            name=table_name,
            con=self.engine,
            if_exists="append",
            index=True,
            index_label="datetime",
        )

        return f"[OK] {file_path}"


if __name__ == "__main__":
    fp = FileProcessor()

    fp.process_csv("data/ohlc/1m/000001.SH1-1M-20140102-L1.CSV")
