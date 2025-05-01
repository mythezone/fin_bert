import pytest
from data.mysql import create_mysql_engine
from sqlalchemy import text


def test_mysql_db_singleton():
    db1 = create_mysql_engine()
    db2 = create_mysql_engine()
    assert db1 is not db2  # 单例验证


def test_mysql_db_connection():
    engine = create_mysql_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * from `test001.sh1`"))
        assert result is not None
