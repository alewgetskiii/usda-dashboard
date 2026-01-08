import sqlite3
from contextlib import contextmanager
from .config import DB_PATH


@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def execute_script(sql: str):
    from textwrap import dedent
    sql = dedent(sql)
    with get_connection() as conn:
        conn.executescript(sql)
