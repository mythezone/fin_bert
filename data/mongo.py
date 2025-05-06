import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import pandas as pd
from pymongo import MongoClient
import re

load_dotenv()


class MongoFileProcessor:
    def __init__(self):

        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.client = MongoClient(mongo_uri)

    def process_csv(self, file_path: str):
        # Parse filename
        filename = os.path.basename(file_path)
        match = re.match(
            r"(?P<code>\d{6})\.(?P<exchange>SH1|SZ1)-(?P<dtype>Tick|1M|10M|30M|60M)-(?P<date>\d{8})-L1\.CSV",
            filename,
        )
        if not match:
            raise ValueError(f"Invalid file name format: {filename}")

        code = match.group("code")
        exchange = match.group("exchange")
        dtype = match.group("dtype")
        date_str = match.group("date")

        dbname = f"jydata_{date_str[:4]}"
        month = date_str[:6]

        db = self.client[dbname]
        collection = db[month]
        if collection.find_one({"_id": filename}):
            return f"[SKIPPED] {file_path} already exists"

        df = pd.read_csv(file_path)
        records = df.to_dict(orient="records")
        document = {
            "_id": filename,
            "code": code,
            "exchange": exchange,
            "dtype": dtype,
            "date": date_str,
            "data": records,
        }

        if records:
            collection.insert_one(document)

        return f"[OK] {file_path}"


if __name__ == "__main__":
    fp = MongoFileProcessor()

    fp.process_csv("data/ohlc/1m/000001.SH1-1M-20140102-L1.CSV")
