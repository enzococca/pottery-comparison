# Account iscritti, ruoli e approvazione admin — Implementation Plan (Fase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sostituire l'auth a due password condivise di `viewer_app.py` con account per-utente (email+password), registrazione self-service, approvazione admin e ruoli (iscritto/editor/admin), ri-gateando le funzioni esistenti.

**Architecture:** Estensione in-place del layer auth stdlib del monolite `viewer_app.py`. Account persistiti nella tabella `users` (attivata via migrazione idempotente), password salate con `hashlib.pbkdf2_hmac`, sessioni in-memory invariate come storage. La logica testabile (hashing, ordinamento ruoli, helper DB sugli utenti) è scritta come funzioni che accettano una connessione, così è coperta da unit test; le rotte HTTP (thin wrapper) sono verificate con `curl`/browser, coerentemente con l'approccio di test del repo.

**Tech Stack:** Python 3 stdlib (`http.server`, `sqlite3`, `hashlib`, `hmac`, `secrets`), pytest. Nessuna nuova dipendenza.

**Spec:** `docs/superpowers/specs/2026-06-26-iscritti-accounts-roles-design.md`

## Global Constraints

- **Zero nuove dipendenze.** Solo stdlib + pytest (già in `requirements.txt`).
- **Eseguire pytest dalla root del repo** (`python -m pytest -q`) — nessun file di config pytest; gli import top-level (`import viewer_app`, `from preprocess import ...`) si risolvono solo dalla root.
- **Il frontend è una f-string Python**: ogni graffa letterale JS/CSS va raddoppiata `{{ }}` o il server crasha al `.format`/f-string. Vale per `WELCOME_PAGE` e `get_viewer_html`.
- **Valori di ruolo (stringhe DB):** `'iscritto'`, `'editor'`, `'admin'`. **Stati account:** `'pending'`, `'approved'`, `'rejected'`, `'suspended'`.
- **Formato hash password:** `pbkdf2_sha256$200000$<salt_hex>$<hash_hex>` (200_000 iterazioni, salt 16 byte).
- **La password admin condivisa** (`ADMIN_HASH`, SHA256 in env) resta come bootstrap; **la `VIEWER_HASH` viene rimossa**. Non riusare `hash_password()` (SHA256 nudo) per i nuovi account.
- **Migrazione**: aggiungere solo colonne mancanti con il pattern `PRAGMA table_info` già usato in `run_auto_migrations()` (sorgente di verità in-app). Non aggiornare lo script standalone `migrate_add_decoration_fields.py`.
- **Commit** dopo ogni task. Niente trailer di AI-attribution nei messaggi di commit.

---

## File Structure

- **Modify `viewer_app.py`** (monolite — tutte le modifiche backend e i due template HTML vivono qui):
  - import: aggiungere `import hmac` (Task 1)
  - sezione password (`hash_password` ~riga 1702): nuove `hash_password_salted` / `verify_password_salted` (Task 1); `verify_credentials` semplificata (Task 5)
  - nuove costanti/funzioni ruolo: `ROLE_RANK`, `role_allows` (Task 2)
  - `run_auto_migrations` (~1320): chiamata a nuova `migrate_users_table(conn)` (Task 3)
  - nuove helper utenti DB (Task 4): `create_user`, `get_user_by_email`, `get_user_by_id`, `list_users`, `set_user_status`, `set_user_role`, `delete_user`, `touch_last_login`
  - sessioni (`create_session` ~1717): firma estesa; metodo handler `require_role` + `require_admin` come alias (Task 5)
  - rotte `do_POST` (~2120): `/api/login` esteso, `/api/logout` fix, nuova `/api/register` (Task 6,7), nuove `/api/admin/users*` (Task 8); re-gating call-site catalogo (Task 9)
  - rotte `do_GET` / `do_DELETE`: `/viewer`, `/api/data`, `/api/config` resi pubblici; ML/3D gateati; delete-image → editor (Task 9)
  - `WELCOME_PAGE` (~2486): form login + registrazione (Task 10)
  - `get_viewer_html(role)` (~2768): UI condizionale per ruolo + pannello "Gestione utenti" (Task 11)
- **Create `tests/test_auth.py`**: unit test per le funzioni pure e gli helper DB (Task 1–5).

---

### Task 1: Hashing password salato (pbkdf2)

**Files:**
- Modify: `viewer_app.py` (import `hmac` riga ~23; nuove funzioni dopo `hash_password`, ~riga 1704)
- Test: `tests/test_auth.py` (create)

**Interfaces:**
- Produces:
  - `hash_password_salted(password: str) -> str` → stringa `pbkdf2_sha256$200000$<salt_hex>$<hash_hex>`
  - `verify_password_salted(password: str, stored: str) -> bool` → confronto costante; `False` su formato malformato

- [ ] **Step 1: Scrivere i test che falliscono**

Create `tests/test_auth.py`:

```python
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
```

- [ ] **Step 2: Eseguire i test per verificarne il fallimento**

Run: `python -m pytest tests/test_auth.py -q`
Expected: FAIL — `AttributeError: module 'viewer_app' has no attribute 'hash_password_salted'`

- [ ] **Step 3: Implementare**

In `viewer_app.py`, aggiungere `import hmac` alla riga ~23 (dopo `import secrets`). Poi, subito dopo `hash_password` (~riga 1704):

```python
PBKDF2_ITERATIONS = 200_000


def hash_password_salted(password):
    """Salted PBKDF2-SHA256 hash for per-user accounts. Returns 'pbkdf2_sha256$iter$salt$hash'."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password_salted(password, stored):
    """Constant-time verify against a 'pbkdf2_sha256$iter$salt$hash' string."""
    try:
        algo, iters, salt_hex, hash_hex = stored.split('$')
        if algo != 'pbkdf2_sha256':
            return False
        dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False
```

- [ ] **Step 4: Eseguire i test (verde)**

Run: `python -m pytest tests/test_auth.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add viewer_app.py tests/test_auth.py
git commit -m "feat(auth): salted pbkdf2 password hashing"
```

---

### Task 2: Logica di ordinamento ruoli (pura)

**Files:**
- Modify: `viewer_app.py` (dopo le costanti hash, ~riga 1715)
- Test: `tests/test_auth.py`

**Interfaces:**
- Produces:
  - `ROLE_RANK: dict` → `{'iscritto': 1, 'editor': 2, 'admin': 3}`
  - `role_allows(min_role: str, current_role: str | None) -> bool` → `True` se `current_role` ha rango ≥ `min_role`; `False` se `current_role` è `None`/sconosciuto.

- [ ] **Step 1: Scrivere i test che falliscono**

Append a `tests/test_auth.py`:

```python
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
```

- [ ] **Step 2: Eseguire per verificare il fallimento**

Run: `python -m pytest tests/test_auth.py -q`
Expected: FAIL — `AttributeError: ... 'role_allows'`

- [ ] **Step 3: Implementare**

In `viewer_app.py` (~riga 1715, dopo `PBKDF2_ITERATIONS`/hash):

```python
ROLE_RANK = {'iscritto': 1, 'editor': 2, 'admin': 3}


def role_allows(min_role, current_role):
    """True se current_role ha rango >= min_role. None/sconosciuto -> False."""
    return ROLE_RANK.get(current_role, 0) >= ROLE_RANK.get(min_role, 99)
```

- [ ] **Step 4: Eseguire i test (verde)**

Run: `python -m pytest tests/test_auth.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add viewer_app.py tests/test_auth.py
git commit -m "feat(auth): role ranking + role_allows helper"
```

---

### Task 3: Migrazione tabella `users`

**Files:**
- Modify: `viewer_app.py` (nuova `migrate_users_table` prima di `run_auto_migrations` ~riga 1320; chiamarla dentro `run_auto_migrations`)
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: nessuno (usa una connessione passata)
- Produces: `migrate_users_table(conn) -> None` — aggiunge colonne mancanti alla tabella `users`: `email`, `status`, `approved_at`, `approved_by`, `last_login`. Idempotente. Crea la tabella `users` se non esiste (per DB di test).

- [ ] **Step 1: Scrivere il test che fallisce**

Append a `tests/test_auth.py`:

```python
import sqlite3


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
```

- [ ] **Step 2: Eseguire per verificare il fallimento**

Run: `python -m pytest tests/test_auth.py -q`
Expected: FAIL — `AttributeError: ... 'migrate_users_table'`

- [ ] **Step 3: Implementare**

In `viewer_app.py`, subito prima di `def run_auto_migrations():` (~riga 1320):

```python
def migrate_users_table(conn):
    """Idempotently bring the users table up to the per-account schema."""
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'iscritto',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    cursor.execute("PRAGMA table_info(users)")
    existing = {row[1] for row in cursor.fetchall()}
    new_columns = [
        ("email", "TEXT"),
        ("status", "TEXT DEFAULT 'pending'"),
        ("approved_at", "TIMESTAMP"),
        ("approved_by", "TEXT"),
        ("last_login", "TIMESTAMP"),
    ]
    for col_name, col_def in new_columns:
        if col_name not in existing:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_def}")
    # indice unico su email (parziale: ignora i NULL legacy)
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL")
    conn.commit()
```

Poi, dentro `run_auto_migrations()`, dopo il blocco di migrazione di `items` (dopo le `ALTER TABLE items ...`, prima del `conn.commit()` finale della funzione), aggiungere:

```python
    # Migrazione tabella users (account per-utente)
    migrate_users_table(conn)
```

- [ ] **Step 4: Eseguire i test (verde)**

Run: `python -m pytest tests/test_auth.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add viewer_app.py tests/test_auth.py
git commit -m "feat(auth): users table migration (email/status/approval columns)"
```

---

### Task 4: Helper DB per gli utenti

**Files:**
- Modify: `viewer_app.py` (nuove funzioni dopo `migrate_users_table`, ~riga 1320)
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: `hash_password_salted`, `verify_password_salted` (Task 1); `migrate_users_table` (Task 3)
- Produces (tutte prendono `conn` come primo argomento):
  - `create_user(conn, email, password) -> int` → id del nuovo utente (`role='iscritto'`, `status='pending'`); solleva `sqlite3.IntegrityError` se email duplicata. Email normalizzata lowercase+strip.
  - `get_user_by_email(conn, email) -> sqlite3.Row | None`
  - `get_user_by_id(conn, user_id) -> sqlite3.Row | None`
  - `list_users(conn) -> list[dict]` → campi: id, email, role, status, created_at, last_login (ordinati: pending prima, poi per created_at desc)
  - `set_user_status(conn, user_id, status, approved_by=None) -> bool`
  - `set_user_role(conn, user_id, role) -> bool`
  - `delete_user(conn, user_id) -> bool`
  - `touch_last_login(conn, user_id) -> None`

- [ ] **Step 1: Scrivere i test che falliscono**

Append a `tests/test_auth.py`:

```python
import pytest


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
```

- [ ] **Step 2: Eseguire per verificare il fallimento**

Run: `python -m pytest tests/test_auth.py -q`
Expected: FAIL — `AttributeError: ... 'create_user'`

- [ ] **Step 3: Implementare**

In `viewer_app.py`, dopo `migrate_users_table`:

```python
def create_user(conn, email, password):
    """Crea un iscritto pending. Solleva sqlite3.IntegrityError su email duplicata."""
    email = email.strip().lower()
    cur = conn.execute(
        "INSERT INTO users (email, password_hash, role, status) VALUES (?, ?, 'iscritto', 'pending')",
        (email, hash_password_salted(password)),
    )
    conn.commit()
    return cur.lastrowid


def get_user_by_email(conn, email):
    return conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()


def get_user_by_id(conn, user_id):
    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def list_users(conn):
    rows = conn.execute(
        """SELECT id, email, role, status, created_at, last_login FROM users
           ORDER BY (status = 'pending') DESC, created_at DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def set_user_status(conn, user_id, status, approved_by=None):
    if status == 'approved':
        conn.execute(
            "UPDATE users SET status = ?, approved_at = CURRENT_TIMESTAMP, approved_by = ? WHERE id = ?",
            (status, approved_by, user_id),
        )
    else:
        conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
    conn.commit()
    return conn.total_changes > 0


def set_user_role(conn, user_id, role):
    conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    conn.commit()
    return conn.total_changes > 0


def delete_user(conn, user_id):
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    return conn.total_changes > 0


def touch_last_login(conn, user_id):
    conn.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))
    conn.commit()
```

> Nota: `conn.total_changes` è cumulativo sulla connessione; nei test ogni connessione è fresca quindi va bene. Nelle rotte la connessione è creata per-richiesta da `get_db()`, quindi anche lì il valore riflette la singola operazione.

- [ ] **Step 4: Eseguire i test (verde)**

Run: `python -m pytest tests/test_auth.py -q`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add viewer_app.py tests/test_auth.py
git commit -m "feat(auth): user CRUD helpers (create/get/list/status/role/delete)"
```

---

### Task 5: Sessioni, verify_credentials e require_role

**Files:**
- Modify: `viewer_app.py` (`create_session` ~1717, `verify_credentials` ~1707; metodo handler `require_admin` ~1870 + nuovo `require_role`)
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: `role_allows` (Task 2)
- Produces:
  - `create_session(role, user_id=None, email=None) -> str`
  - `verify_credentials(password) -> 'admin' | None` (solo admin condiviso)
  - metodo `ViewerHandler.require_role(self, min_role) -> bool` (403 JSON se insufficiente)
  - `ViewerHandler.require_admin(self) -> bool` reimplementato come `return self.require_role('admin')`

- [ ] **Step 1: Scrivere il test che fallisce (verify_credentials)**

Append a `tests/test_auth.py`:

```python
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
```

- [ ] **Step 2: Eseguire per verificare il fallimento**

Run: `python -m pytest tests/test_auth.py -q`
Expected: FAIL — `create_session() got an unexpected keyword argument 'user_id'` (e/o assenza del ramo admin-only)

- [ ] **Step 3: Implementare**

In `viewer_app.py` sostituire `verify_credentials` (~1707):

```python
def verify_credentials(password):
    """Bootstrap super-admin: verifica solo la password admin condivisa."""
    if hash_password(password) == ADMIN_HASH:
        return 'admin'
    return None
```

Sostituire `create_session` (~1717):

```python
def create_session(role, user_id=None, email=None):
    """Create session token"""
    token = secrets.token_hex(32)
    SESSIONS[token] = {
        'role': role,
        'user_id': user_id,
        'email': email,
        'created': datetime.now().isoformat(),
    }
    return token
```

Sostituire il metodo `require_admin` (~1870) e aggiungere `require_role` subito sopra di esso:

```python
    def require_role(self, min_role):
        """Gate API: 403 JSON se il ruolo corrente non raggiunge min_role."""
        if not role_allows(min_role, self.get_role()):
            self.send_json({'error': 'Accesso non autorizzato', 'required': min_role}, 403)
            return False
        return True

    def require_admin(self):
        """Alias storico: richiede ruolo admin."""
        return self.require_role('admin')
```

- [ ] **Step 4: Eseguire i test (verde)**

Run: `python -m pytest tests/test_auth.py -q`
Expected: PASS (15 passed)

- [ ] **Step 5: Commit**

```bash
git add viewer_app.py tests/test_auth.py
git commit -m "feat(auth): session carries user; admin-only verify_credentials; require_role"
```

---

### Task 6: Endpoint di registrazione `/api/register`

**Files:**
- Modify: `viewer_app.py` (`do_POST`, subito dopo il blocco `/api/login` ~riga 2139)

**Interfaces:**
- Consumes: `get_db`, `create_user`, `get_user_by_email`
- Produces: rotta pubblica `POST /api/register`, body `{email, password}`.

- [ ] **Step 1: Implementare la rotta**

In `do_POST`, dopo il `return` del blocco `/api/login` (~riga 2139):

```python
        # Registrazione iscritto (pubblica)
        if parsed.path == '/api/register':
            email = (post_data.get('email') or '').strip().lower()
            password = post_data.get('password') or ''
            if '@' not in email or '.' not in email.split('@')[-1]:
                self.send_json({'success': False, 'error': 'Email non valida'}, 400)
                return
            if len(password) < 8:
                self.send_json({'success': False, 'error': 'La password deve avere almeno 8 caratteri'}, 400)
                return
            conn = get_db()
            try:
                if get_user_by_email(conn, email):
                    self.send_json({'success': False, 'error': 'Email già registrata'}, 409)
                    return
                create_user(conn, email, password)
                self.send_json({'success': True, 'message': 'Registrazione ricevuta. In attesa di approvazione.'})
            except sqlite3.IntegrityError:
                self.send_json({'success': False, 'error': 'Email già registrata'}, 409)
            finally:
                conn.close()
            return
```

- [ ] **Step 2: Verifica unit (helper già coperti)**

Run: `python -m pytest tests/test_auth.py -q`
Expected: PASS (15 passed) — nessuna regressione.

- [ ] **Step 3: Verifica manuale (curl)**

Avviare in un terminale separato: `python viewer_app.py` (porta 8080). Poi:

```bash
curl -s -X POST localhost:8080/api/register -d '{"email":"test@example.com","password":"pw12345678"}'
# atteso: {"success": true, "message": "Registrazione ricevuta. In attesa di approvazione."}
curl -s -X POST localhost:8080/api/register -d '{"email":"test@example.com","password":"pw12345678"}'
# atteso: 409 {"success": false, "error": "Email già registrata"}
curl -s -X POST localhost:8080/api/register -d '{"email":"bad","password":"short"}'
# atteso: 400 {"success": false, "error": "Email non valida"}
```

- [ ] **Step 4: Commit**

```bash
git add viewer_app.py
git commit -m "feat(auth): public /api/register endpoint (pending member)"
```

---

### Task 7: Login esteso + logout che pulisce la sessione

**Files:**
- Modify: `viewer_app.py` (`do_POST` blocchi `/api/login` ~2126 e `/api/logout` ~2142)

**Interfaces:**
- Consumes: `get_db`, `get_user_by_email`, `verify_password_salted`, `touch_last_login`, `verify_credentials`, `create_session`

- [ ] **Step 1: Sostituire il blocco `/api/login`**

```python
        # Login
        if parsed.path == '/api/login':
            email = (post_data.get('email') or '').strip().lower()
            password = post_data.get('password', '')

            if email:
                conn = get_db()
                try:
                    user = get_user_by_email(conn, email)
                    if not user or not verify_password_salted(password, user['password_hash']):
                        self.send_json({'success': False, 'error': 'Credenziali non valide'}, 401)
                        return
                    if user['status'] == 'pending':
                        self.send_json({'success': False, 'error': 'Account in attesa di approvazione', 'status': 'pending'}, 403)
                        return
                    if user['status'] != 'approved':
                        self.send_json({'success': False, 'error': 'Account non attivo', 'status': user['status']}, 403)
                        return
                    touch_last_login(conn, user['id'])
                    token = create_session(user['role'], user_id=user['id'], email=user['email'])
                finally:
                    conn.close()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Set-Cookie', f'session={token}; Path=/; HttpOnly; SameSite=Strict')
                self.end_headers()
                self.wfile.write(json.dumps({'success': True, 'role': user['role']}).encode())
                return

            # bootstrap admin (email vuota, solo password condivisa)
            role = verify_credentials(password)
            if role:
                token = create_session(role)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Set-Cookie', f'session={token}; Path=/; HttpOnly; SameSite=Strict')
                self.end_headers()
                self.wfile.write(json.dumps({'success': True, 'role': role}).encode())
            else:
                self.send_json({'success': False, 'error': 'Credenziali non valide'}, 401)
            return
```

- [ ] **Step 2: Sostituire il blocco `/api/logout`** (rimuove la sessione lato server)

```python
        # Logout
        if parsed.path == '/api/logout':
            session_data = get_session(self.headers.get('Cookie'))
            cookies = http.cookies.SimpleCookie()
            try:
                cookies.load(self.headers.get('Cookie') or '')
                if 'session' in cookies:
                    SESSIONS.pop(cookies['session'].value, None)
            except Exception:
                pass
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Set-Cookie', 'session=; Path=/; Max-Age=0')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True}).encode())
            return
```

- [ ] **Step 3: Verifica manuale (curl) — flusso completo dopo l'approvazione (vedi Task 8)**

Per ora, con l'utente `test@example.com` ancora `pending`:

```bash
curl -s -X POST localhost:8080/api/login -d '{"email":"test@example.com","password":"pw12345678"}'
# atteso: 403 {"success": false, "error": "Account in attesa di approvazione", "status": "pending"}
curl -s -X POST localhost:8080/api/login -d '{"password":"<ADMIN_PASSWORD>"}'
# atteso: 200 {"success": true, "role": "admin"}  (default admin2024 se ADMIN_HASH non impostata)
```

- [ ] **Step 4: Eseguire i test unit (nessuna regressione)**

Run: `python -m pytest tests/test_auth.py -q`
Expected: PASS (15 passed)

- [ ] **Step 5: Commit**

```bash
git add viewer_app.py
git commit -m "feat(auth): email/password login + admin bootstrap; logout clears session"
```

---

### Task 8: API admin di gestione utenti `/api/admin/users*`

**Files:**
- Modify: `viewer_app.py` (`do_POST` nuovi blocchi dopo `/api/register`; `do_GET` per la lista; `do_DELETE` per l'eliminazione)

**Interfaces:**
- Consumes: `require_role`, `get_db`, `list_users`, `set_user_status`, `set_user_role`, `delete_user`, `get_role`/sessione (per `approved_by`)

- [ ] **Step 1: Aggiungere la lista utenti in `do_GET`**

In `do_GET`, nella zona delle API protette (dopo `/api/data`, ~riga 1956):

```python
        # Lista utenti (admin)
        if parsed.path == '/api/admin/users':
            if not self.require_role('admin'):
                return
            conn = get_db()
            try:
                users = list_users(conn)
            finally:
                conn.close()
            self.send_json({'users': users})
            return
```

- [ ] **Step 2: Aggiungere le azioni in `do_POST`** (dopo il blocco `/api/register`)

```python
        # Azioni admin sugli utenti
        if parsed.path in ('/api/admin/users/approve', '/api/admin/users/reject',
                           '/api/admin/users/role', '/api/admin/users/suspend',
                           '/api/admin/users/reset-password'):
            if not self.require_role('admin'):
                return
            user_id = post_data.get('user_id')
            if not user_id:
                self.send_json({'error': 'user_id mancante'}, 400)
                return
            conn = get_db()
            try:
                actor = (get_session(self.headers.get('Cookie')) or {}).get('email') or 'admin'
                if parsed.path.endswith('/approve'):
                    ok = set_user_status(conn, user_id, 'approved', approved_by=actor)
                elif parsed.path.endswith('/reject'):
                    ok = set_user_status(conn, user_id, 'rejected')
                elif parsed.path.endswith('/suspend'):
                    suspend = bool(post_data.get('suspend', True))
                    ok = set_user_status(conn, user_id, 'suspended' if suspend else 'approved',
                                         approved_by=None if suspend else actor)
                elif parsed.path.endswith('/role'):
                    role = post_data.get('role')
                    if role not in ('iscritto', 'editor'):
                        self.send_json({'error': 'Ruolo non valido'}, 400)
                        return
                    ok = set_user_role(conn, user_id, role)
                elif parsed.path.endswith('/reset-password'):
                    new_pw = post_data.get('new_password') or ''
                    if len(new_pw) < 8:
                        self.send_json({'error': 'Password troppo corta'}, 400)
                        return
                    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                                 (hash_password_salted(new_pw), user_id))
                    conn.commit()
                    ok = conn.total_changes > 0
            finally:
                conn.close()
            if not ok:
                self.send_json({'error': 'Utente non trovato'}, 404)
            else:
                self.send_json({'success': True})
            return
```

- [ ] **Step 3: Aggiungere l'eliminazione in `do_DELETE`**

In `do_DELETE`, dopo il `require_admin()` iniziale (~riga 2453), il gate admin copre già tutto l'handler. Aggiungere il ramo:

```python
        # Elimina utente (admin)
        if parsed.path == '/api/admin/users':
            user_id = query.get('user_id', [None])[0] or (post_data.get('user_id') if 'post_data' in dir() else None)
            if not user_id:
                self.send_json({'error': 'user_id mancante'}, 400)
                return
            conn = get_db()
            try:
                ok = delete_user(conn, user_id)
            finally:
                conn.close()
            self.send_json({'success': True} if ok else {'error': 'Utente non trovato'}, 200 if ok else 404)
            return
```

> Nota: verificare come `do_DELETE` legge i parametri (query vs body) e usare lo stesso pattern degli altri rami delete già presenti; usare `query` se l'handler fa `urlparse`/`parse_qs` in cima (come `do_GET`).

- [ ] **Step 4: Verifica manuale (curl) — ciclo completo**

```bash
# login admin -> salva il cookie
curl -s -c /tmp/cj.txt -X POST localhost:8080/api/login -d '{"password":"<ADMIN_PASSWORD>"}' >/dev/null
# lista utenti (vedi test@example.com pending)
curl -s -b /tmp/cj.txt localhost:8080/api/admin/users
# approva (usa l'id reale dalla lista, es. 1)
curl -s -b /tmp/cj.txt -X POST localhost:8080/api/admin/users/approve -d '{"user_id":1}'
# ora il login dell'iscritto funziona
curl -s -X POST localhost:8080/api/login -d '{"email":"test@example.com","password":"pw12345678"}'
# atteso: 200 {"success": true, "role": "iscritto"}
# promozione a editor
curl -s -b /tmp/cj.txt -X POST localhost:8080/api/admin/users/role -d '{"user_id":1,"role":"editor"}'
# atteso: {"success": true}
```

- [ ] **Step 5: Commit**

```bash
git add viewer_app.py
git commit -m "feat(auth): admin user management endpoints (list/approve/reject/role/suspend/delete)"
```

---

### Task 9: Re-gating endpoint + lettura pubblica del catalogo

**Files:**
- Modify: `viewer_app.py` — call-site `require_admin()` del catalogo → `require_role('editor')`; gate `iscritto` su ML/3D; `/viewer`, `/api/data`, `/api/config` pubblici.

**Interfaces:**
- Consumes: `require_role` (Task 5)

- [ ] **Step 1: Catalogo write → editor**

Sostituire `if not self.require_admin():` con `if not self.require_role('editor'):` nei blocchi:
`/api/update-item` (~2154), `/api/update-batch` (~2168), `/api/rotate-image` (~2184), `/api/flip-image` (~2201), `/api/vocabulary` (~2218). 

In `do_DELETE`: il gate iniziale `require_admin()` (~2453) gestisce sia `delete-image` sia (Task 8) `admin/users`. Separare: per `/api/delete-image` usare `require_role('editor')`, per `/api/admin/users` lasciare `require_role('admin')`. Cambiare il gate iniziale dell'handler da unico a per-ramo (spostare il check dentro ciascun ramo).

- [ ] **Step 2: ML/3D → iscritto approvato**

All'inizio della gestione di ogni endpoint ML/3D in `do_POST` (`/api/ml/classify`, `/api/ml/explain`, `/api/ml/similar`, `/api/ml/preprocess`, `/api/ml/combine-drawing`, `/api/ml/similarity-heatmap`, `/api/ml/all-images`, `/api/3d/reconstruct`), aggiungere come prima riga del blocco:

```python
            if not self.require_role('iscritto'):
                return
```

> `/api/ml/status` può restare pubblico (non espone dati sensibili). Localizzare i blocchi con: `grep -n "/api/ml/\|/api/3d/reconstruct" viewer_app.py`.

- [ ] **Step 3: Lettura pubblica**

`/viewer` (~1935): rimuovere il gate redirect, servire a tutti col ruolo corrente:

```python
        if parsed.path == '/viewer':
            self.send_html(get_viewer_html(self.get_role()))
            return
```

`/api/data` (~1953) e `/api/config` (~1945): rimuovere ogni dipendenza da auth (sono già di sola lettura). `/api/config` continua a impostare `config['user_role'] = self.get_role()` (può essere `None`).

- [ ] **Step 4: Verifica manuale (curl)**

```bash
# catalogo pubblico senza login
curl -s -o /dev/null -w "%{http_code}\n" localhost:8080/viewer            # 200
curl -s -o /dev/null -w "%{http_code}\n" localhost:8080/api/data           # 200
# ML senza login -> 403
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8080/api/ml/similar -d '{}'   # 403
# edit senza login -> 403
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8080/api/update-item -d '{}'  # 403
# come iscritto approvato (cookie da login iscritto) -> ML 200/4xx applicativo, non 403
curl -s -c /tmp/ij.txt -X POST localhost:8080/api/login -d '{"email":"test@example.com","password":"pw12345678"}' >/dev/null
curl -s -o /dev/null -w "%{http_code}\n" -b /tmp/ij.txt -X POST localhost:8080/api/ml/status   # 200
```

- [ ] **Step 5: Eseguire i test unit + commit**

Run: `python -m pytest -q` (intera suite, incl. test_preprocess) → atteso PASS.

```bash
git add viewer_app.py
git commit -m "feat(auth): re-gate routes (public read, iscritto for ML/3D, editor for catalog writes)"
```

---

### Task 10: UI — pagina login + registrazione (`WELCOME_PAGE`)

**Files:**
- Modify: `viewer_app.py` (`WELCOME_PAGE` ~riga 2486)

**Interfaces:** consuma `/api/login`, `/api/register`.

- [ ] **Step 1: Aggiornare il markup**

Sostituire il form della `WELCOME_PAGE` con due schede/sezioni — **Login** (email + password + bottone) e **Registrati** (email + password + conferma) — più un link/sezione "Accesso amministratore" (solo password). Ricordare il **raddoppio delle graffe** `{{ }}` in tutto il CSS/JS della f-string. Comportamento JS richiesto:
- Login: `fetch('/api/login', {method:'POST', body: JSON.stringify({email, password})})`; su 200 → `window.location='/viewer'`; su 403 con `status==='pending'` → messaggio "Account in attesa di approvazione"; su 401 → "Credenziali non valide".
- Registrazione: valida che le due password coincidano e ≥8 char lato client; `fetch('/api/register', ...)`; su success → messaggio "Registrazione ricevuta, in attesa di approvazione" e torna al login.
- Accesso admin: `fetch('/api/login', {body: JSON.stringify({password})})` (email vuota).

- [ ] **Step 2: Verifica manuale (browser)**

Aprire `http://localhost:8080/`:
- Registrare un nuovo utente → messaggio "in attesa di approvazione".
- Tentare il login da non approvato → messaggio pending.
- Login admin (solo password) → redirect a `/viewer`.

- [ ] **Step 3: Commit**

```bash
git add viewer_app.py
git commit -m "feat(ui): login + registration on welcome page"
```

---

### Task 11: UI — viewer condizionale per ruolo + pannello "Gestione utenti"

**Files:**
- Modify: `viewer_app.py` (`get_viewer_html(role)` ~riga 2768)

**Interfaces:** consuma `/api/admin/users` (GET) e `/api/admin/users/*` (POST), `/api/config` (per `user_role`).

- [ ] **Step 1: Gating UI per ruolo**

In `get_viewer_html(role)`, mostrare/nascondere i controlli in base a `role` (può essere `None`):
- Pulsanti edit/delete/rotate/flip/vocabolario: solo se `role in ('editor','admin')`.
- Controlli comparazione/3D/AI: solo se `role in ('iscritto','editor','admin')`; per l'anonimo, al click mostrare un invito a registrarsi/loggarsi.
- Voce di menu "Gestione utenti": solo se `role == 'admin'`.

Usare il `role` passato (server-side) per non renderizzare affatto il markup riservato (difesa in profondità: il backend impone comunque con `require_role`).

- [ ] **Step 2: Pannello "Gestione utenti" (solo admin)**

Aggiungere un modale/sezione che:
- `GET /api/admin/users` al caricamento e popola due liste: **In attesa** (con [Approva]/[Rifiuta]) e **Attivi** (email, selettore ruolo Iscritto/Editor → `POST /api/admin/users/role`, [Sospendi]/[Riattiva] → `/suspend`, [Elimina] → `DELETE /api/admin/users?user_id=`).
- Dopo ogni azione, ricarica la lista.
- Raddoppiare le graffe `{{ }}` nel JS/CSS della f-string.

- [ ] **Step 3: Verifica manuale (browser)**

- Login admin → vedere "Gestione utenti", approvare l'iscritto pending, promuoverlo a editor, sospenderlo/riattivarlo.
- Login come iscritto approvato → vedere comparazione/3D/AI ma **non** edit/delete né gestione utenti.
- Anonimo → solo browse/ricerca; il tentativo di comparazione invita al login.

- [ ] **Step 4: Eseguire l'intera suite + commit**

Run: `python -m pytest -q` → PASS.

```bash
git add viewer_app.py
git commit -m "feat(ui): role-conditional viewer controls + admin user management panel"
```

---

## Self-Review

**Spec coverage:**
- Registrazione email/password → Task 6 ✓
- Approvazione/rifiuto/promozione/sospensione admin → Task 8, 11 ✓
- 4 ruoli + ordinamento → Task 2, 5 ✓
- Gating (pubblico lettura / iscritto ML-3D / editor catalogo / admin utenti) → Task 9 ✓
- Admin condiviso bootstrap, viewer rimosso → Task 5, 7 ✓
- Hashing salato + logout fix → Task 1, 7 ✓
- Migrazione users → Task 3 ✓
- UI login/registrazione + pannello admin → Task 10, 11 ✓
- Annotazioni → **fuori scope (Fase 2)**, coerente con la spec ✓

**Placeholder scan:** nessun "TBD"/"handle edge cases" generico; ogni step di codice mostra il codice; le verifiche di rotta hanno comandi `curl` con output atteso. Le due note "verificare come do_DELETE legge i parametri" e "localizzare i blocchi ML con grep" sono istruzioni di localizzazione precise, non placeholder di logica.

**Type consistency:** firme coerenti tra task — `create_session(role, user_id, email)`, `role_allows(min_role, current_role)`, helper utenti sempre `(conn, ...)`, valori ruolo/stato esatti come da Global Constraints. `require_role(min_role)` usato in Task 8/9 è definito in Task 5.

---

## Note di esecuzione

- I Task 1–5 sono TDD puro (pytest verde prima di proseguire). I Task 6–11 sono rotte/UI del monolite stdlib: si verificano con `curl`/browser (coerente con `tests/` del repo che testa solo funzioni pure), mantenendo comunque i test unit verdi a ogni commit.
- Tenere un'istanza `python viewer_app.py` in un terminale separato per le verifiche manuali; ricordare che le sessioni in-memory si azzerano a ogni riavvio.
- Branch di lavoro: `feat/iscritti-accounts-roles` (già creato; la spec è committata).
