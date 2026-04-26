from memo_app.db import init_db


def test_init_db_creates_memos_table():
    conn = init_db(":memory:")
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memos'"
    )
    assert cursor.fetchone() is not None


def test_memos_table_schema():
    conn = init_db(":memory:")
    cursor = conn.execute("PRAGMA table_info(memos)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    assert "id" in columns
    assert "title" in columns
    assert "content" in columns
    assert "created_at" in columns
