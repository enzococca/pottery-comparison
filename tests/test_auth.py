import sqlite3

import pytest

import viewer_app


def test_hash_round_trip():
    stored = viewer_app.hash_password_salted("correct horse")
    assert viewer_app.verify_password_salted("correct horse", stored) is True


def test_hash_wrong_password():
    stored = viewer_app.hash_password_salted("secret123")
    assert viewer_app.verify_password_salted("secret124", stored) is False


def test_hash_format():
    stored = viewer_app.hash_password_salted("pw")
    parts = stored.split("$")
    assert parts[0] == "pbkdf2_sha256"
    assert parts[1] == "200000"
    assert len(parts) == 4


def test_hash_unique_salt():
    a = viewer_app.hash_password_salted("same")
    b = viewer_app.hash_password_salted("same")
    assert a != b  # salt casuale diverso


def test_verify_malformed_is_false():
    assert viewer_app.verify_password_salted("pw", "garbage") is False
    assert viewer_app.verify_password_salted("pw", "") is False


def test_role_allows_hierarchy():
    assert viewer_app.role_allows('iscritto', 'iscritto') is True
    assert viewer_app.role_allows('iscritto', 'editor') is True
    assert viewer_app.role_allows('iscritto', 'admin') is True
    assert viewer_app.role_allows('editor', 'iscritto') is False
    assert viewer_app.role_allows('editor', 'editor') is True
    assert viewer_app.role_allows('admin', 'editor') is False
    assert viewer_app.role_allows('admin', 'admin') is True


def test_role_allows_anonymous():
    assert viewer_app.role_allows('iscritto', None) is False
    assert viewer_app.role_allows('iscritto', 'sconosciuto') is False


def _fresh_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_migrate_users_table_adds_columns():
    conn = _fresh_conn()
    # vecchia forma della tabella (come nel DB committato)
    conn.execute("""CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'viewer',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    viewer_app.migrate_users_table(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
    assert {'email', 'status', 'approved_at', 'approved_by', 'last_login'} <= cols


def test_migrate_users_table_idempotent():
    conn = _fresh_conn()
    viewer_app.migrate_users_table(conn)  # crea da zero
    viewer_app.migrate_users_table(conn)  # secondo giro: nessun errore
    cols = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
    assert 'email' in cols


def _migrated_conn():
    conn = _fresh_conn()
    viewer_app.migrate_users_table(conn)
    return conn


def test_create_and_get_user():
    conn = _migrated_conn()
    uid = viewer_app.create_user(conn, "  Mario@Example.com ", "pw12345678")
    row = viewer_app.get_user_by_email(conn, "mario@example.com")
    assert row is not None
    assert row["id"] == uid
    assert row["role"] == "iscritto"
    assert row["status"] == "pending"
    assert viewer_app.verify_password_salted("pw12345678", row["password_hash"]) is True


def test_create_user_duplicate_email():
    conn = _migrated_conn()
    viewer_app.create_user(conn, "dup@example.com", "pw12345678")
    with pytest.raises(sqlite3.IntegrityError):
        viewer_app.create_user(conn, "dup@example.com", "other12345")


def test_set_status_and_role():
    conn = _migrated_conn()
    uid = viewer_app.create_user(conn, "u@example.com", "pw12345678")
    assert viewer_app.set_user_status(conn, uid, "approved", approved_by="admin") is True
    assert viewer_app.set_user_role(conn, uid, "editor") is True
    row = viewer_app.get_user_by_id(conn, uid)
    assert row["status"] == "approved"
    assert row["role"] == "editor"
    assert row["approved_by"] == "admin"


def test_list_and_delete_user():
    conn = _migrated_conn()
    uid = viewer_app.create_user(conn, "a@example.com", "pw12345678")
    assert any(u["email"] == "a@example.com" for u in viewer_app.list_users(conn))
    assert viewer_app.delete_user(conn, uid) is True
    assert viewer_app.get_user_by_id(conn, uid) is None


def test_mutators_return_false_for_missing_user():
    conn = _migrated_conn()
    assert viewer_app.set_user_status(conn, 99999, "approved") is False
    assert viewer_app.set_user_role(conn, 99999, "editor") is False
    assert viewer_app.delete_user(conn, 99999) is False


def test_verify_credentials_admin_only(monkeypatch):
    # hash di 'topsecret'
    h = viewer_app.hash_password("topsecret")
    monkeypatch.setattr(viewer_app, "ADMIN_HASH", h)
    assert viewer_app.verify_credentials("topsecret") == "admin"
    assert viewer_app.verify_credentials("wrong") is None


def test_create_session_carries_user(monkeypatch):
    token = viewer_app.create_session("iscritto", user_id=42, email="x@example.com")
    sess = viewer_app.SESSIONS[token]
    assert sess["role"] == "iscritto"
    assert sess["user_id"] == 42
    assert sess["email"] == "x@example.com"
