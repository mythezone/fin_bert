import os
import pandas as pd
import pytest
from sqlalchemy import inspect
from data.csv_to_db import process_custom_csv
from data.mysql import create_mysql_engine

TEST_CSV_FILE = "test_sample.csv"
TEST_CODE = "test001.sh1"  # 自定义 Code 字段以避免污染正式表


@pytest.fixture(scope="module")
def sample_csv_file():
    data = """Code,Date,Time,Open,High,Low,Close,Volume,Turnover,MatchItems,Interest
test001.sh1,20230101,93000,100,101,99,100.5,1000,100000.0,50,0
test001.sh1,20230101,93100,100.5,102,100,101,1100,110000.0,60,0
"""
    with open(TEST_CSV_FILE, "w") as f:
        f.write(data)
    yield TEST_CSV_FILE
    os.remove(TEST_CSV_FILE)


def test_process_custom_csv_inserts_data(sample_csv_file):
    process_custom_csv(sample_csv_file)

    engine = create_mysql_engine()
    inspector = inspect(engine)
    assert TEST_CODE in inspector.get_table_names()

    df = pd.read_sql_table(TEST_CODE, engine)
    assert len(df) == 8
    assert "Open" in df.columns
    assert "Turnover" in df.columns
    assert "MatchItems" in df.columns
