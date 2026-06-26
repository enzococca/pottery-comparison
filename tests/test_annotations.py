import json
import sqlite3
import viewer_app


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _migrated():
    conn = _conn()
    viewer_app.migrate_annotations_table(conn)
    return conn


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


def test_create_and_get_round_trip():
    conn = _migrated()
    rid = viewer_app.create_annotation(conn, 1, "data:image/png;base64,AAA", "la mia nota",
                                       json.dumps([{"id": "x", "similarity": 88}]))
    row = viewer_app.get_annotation(conn, rid, 1)
    assert row is not None
    assert row["image_data"] == "data:image/png;base64,AAA"
    assert row["note_text"] == "la mia nota"
    assert json.loads(row["results_json"])[0]["id"] == "x"


def test_list_omits_image_and_counts_results():
    conn = _migrated()
    viewer_app.create_annotation(conn, 1, "img", "n1", json.dumps([{"id": "a"}, {"id": "b"}]))
    rows = viewer_app.list_annotations(conn, 1)
    assert len(rows) == 1
    assert "image_data" not in rows[0]
    assert rows[0]["result_count"] == 2


def test_owner_scoping_get_and_delete():
    conn = _migrated()
    rid = viewer_app.create_annotation(conn, 1, "img", "mine", None)
    # another user cannot read or delete it
    assert viewer_app.get_annotation(conn, rid, 2) is None
    assert viewer_app.delete_annotation(conn, rid, 2) is False
    # owner can
    assert viewer_app.get_annotation(conn, rid, 1) is not None
    assert viewer_app.delete_annotation(conn, rid, 1) is True
    assert viewer_app.get_annotation(conn, rid, 1) is None


def test_delete_missing_returns_false():
    conn = _migrated()
    assert viewer_app.delete_annotation(conn, 999, 1) is False
