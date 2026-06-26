import sqlite3
import viewer_app


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    return c


def test_migrate_creates_table():
    conn = _conn()
    viewer_app.migrate_annotations_table(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(comparison_annotations)")}
    assert {'id', 'user_id', 'image_data', 'note_text', 'results_json', 'created_at'} <= cols


def test_migrate_is_idempotent():
    conn = _conn()
    viewer_app.migrate_annotations_table(conn)
    viewer_app.migrate_annotations_table(conn)  # no error second time
    cols = {r[1] for r in conn.execute("PRAGMA table_info(comparison_annotations)")}
    assert 'image_data' in cols
