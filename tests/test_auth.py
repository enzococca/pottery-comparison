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
