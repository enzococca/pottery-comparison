# Annotazioni personali di comparazione — Implementation Plan (Fase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permettere a un iscritto approvato di salvare annotazioni personali (immagine di comparazione + nota + snapshot dei match), rivederle nel pannello "Le mie annotazioni" e riusarle per un nuovo confronto.

**Architecture:** Estensione in-place del monolite `viewer_app.py`, rispecchiando i pattern della Fase 1: nuova tabella `comparison_annotations` via migrazione idempotente, helper DB che prendono `conn` (testabili, owner-scoped), endpoint REST `/api/annotations*` gateati `require_role('iscritto')` e filtrati per `user_id` di sessione, UI nel modale di comparazione. Le annotazioni vivono nel DB sul volume Railway (`DATA_DIR`).

**Tech Stack:** Python 3 stdlib (`http.server`, `sqlite3`, `json`), pytest. Nessuna nuova dipendenza.

**Spec:** `docs/superpowers/specs/2026-06-26-personal-annotations-design.md`

## Global Constraints

- **Zero nuove dipendenze** (solo stdlib + pytest).
- **Eseguire pytest dalla root** (`python -m pytest -q`; nessun config; `import viewer_app` si risolve solo dalla root).
- **Owner-scoping rigoroso**: ogni SELECT/DELETE filtra `WHERE user_id = <sessione>`. Un utente non accede mai alle annotazioni di un altro (stesso codice `404` per "non esiste" e "non tua").
- **Tutte le rotte** `/api/annotations*` → `require_role('iscritto')`. `POST` richiede inoltre `user_id` di sessione **non None** (il bootstrap-admin a password condivisa non crea annotazioni → `403`).
- **Tabella**: `comparison_annotations(id, user_id NOT NULL, image_data TEXT NOT NULL, note_text TEXT, results_json TEXT, created_at)` + indice `idx_annotations_user(user_id)`.
- **Cap anti-bloat**: `image_data` ≤ 6 MB (6_000_000 char); `note_text` ≤ 4000 char; `results` troncati a 30 elementi.
- **Campi salvati per match** (whitelist, nomi come nel frontend): `id`, `collection`, `image_path`, `similarity`, `coarse_similarity`, `image_type`, `macro_period`, `period`.
- **`success` detection** sugli UPDATE/DELETE: `cursor.rowcount > 0` (convenzione Fase 1, NON `conn.total_changes`).
- **Frontend è una f-string** (`get_viewer_html`): il blocco JS nuovo va costruito come **stringa Python semplice** iniettata via singolo `{var}` (pattern del pannello admin Fase 1) per evitare il raddoppio graffe; gli **id numerici** vanno passati come numeri negli `onclick` e il **testo utente (nota)** va sempre escapato (mai interpolato dentro `on*=`). Riusare l'helper JS `umEsc` già presente.
- **Commit dopo ogni task. Niente trailer di AI-attribution.**

---

## File Structure

- **Modify `viewer_app.py`** (tutto il backend e l'UI vivono qui):
  - `migrate_annotations_table(conn)` accanto a `migrate_users_table`; chiamata in `run_auto_migrations()` (Task 1)
  - helper DB annotazioni dopo gli helper utenti: `create_annotation`, `list_annotations`, `get_annotation`, `delete_annotation` (Task 2)
  - rotte: `do_POST` (`/api/annotations`), `do_GET` (`/api/annotations`, `/api/annotations/<id>`), `do_DELETE` (`/api/annotations/<id>`) (Task 3)
  - UI in `get_viewer_html` (gated `is_member`): pulsante+flusso "Salva annotazione" nel modale ML (Task 4); pulsante+pannello "Le mie annotazioni" con Apri/Riusa/Elimina (Task 5)
- **Create `tests/test_annotations.py`**: unit per migrazione + helper (incl. owner-scoping) (Task 1–2).

---

### Task 1: Migrazione tabella `comparison_annotations`

**Files:**
- Modify: `viewer_app.py` (nuova `migrate_annotations_table` accanto a `migrate_users_table`; chiamata dentro `run_auto_migrations`)
- Test: `tests/test_annotations.py` (create)

**Interfaces:**
- Produces: `migrate_annotations_table(conn) -> None` — crea (idempotente) la tabella `comparison_annotations` + indice.

- [ ] **Step 1: Scrivere i test che falliscono**

Create `tests/test_annotations.py`:

```python
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
```

- [ ] **Step 2: Eseguire per verificare il fallimento**

Run: `python -m pytest tests/test_annotations.py -q`
Expected: FAIL — `AttributeError: module 'viewer_app' has no attribute 'migrate_annotations_table'`

- [ ] **Step 3: Implementare**

In `viewer_app.py`, subito prima di `def migrate_users_table(conn):` (o subito dopo — purché prima di `run_auto_migrations`):

```python
def migrate_annotations_table(conn):
    """Idempotently create the personal comparison-annotations table."""
    conn.execute("""CREATE TABLE IF NOT EXISTS comparison_annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        image_data TEXT NOT NULL,
        note_text TEXT,
        results_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_user ON comparison_annotations(user_id)")
    conn.commit()
```

Poi, dentro `run_auto_migrations()`, accanto alla chiamata esistente `migrate_users_table(conn)`, aggiungere:

```python
    migrate_annotations_table(conn)
```

- [ ] **Step 4: Eseguire i test (verde)**

Run: `python -m pytest tests/test_annotations.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add viewer_app.py tests/test_annotations.py
git commit -m "feat(annotations): comparison_annotations table migration"
```

---

### Task 2: Helper DB annotazioni (owner-scoped)

**Files:**
- Modify: `viewer_app.py` (dopo `migrate_annotations_table` o dopo gli helper utenti)
- Test: `tests/test_annotations.py`

**Interfaces:**
- Consumes: `migrate_annotations_table` (Task 1)
- Produces:
  - `create_annotation(conn, user_id, image_data, note_text, results_json) -> int` — inserisce, ritorna l'id. `results_json` è una stringa (JSON già serializzato dal chiamante) o None.
  - `list_annotations(conn, user_id) -> list[dict]` — metadati SENZA `image_data`: `id`, `note_text`, `created_at`, `result_count` (numero elementi in `results_json`, 0 se assente/illeggibile). Ordinati `created_at DESC, id DESC`.
  - `get_annotation(conn, annotation_id, user_id) -> sqlite3.Row | None` — record completo, filtrato per `user_id`.
  - `delete_annotation(conn, annotation_id, user_id) -> bool` — `cursor.rowcount > 0`, filtrato per `user_id`.

- [ ] **Step 1: Scrivere i test che falliscono**

Append a `tests/test_annotations.py`:

```python
import json


def _migrated():
    conn = _conn()
    viewer_app.migrate_annotations_table(conn)
    return conn


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
```

- [ ] **Step 2: Eseguire per verificare il fallimento**

Run: `python -m pytest tests/test_annotations.py -q`
Expected: FAIL — `AttributeError: ... 'create_annotation'`

- [ ] **Step 3: Implementare**

In `viewer_app.py` (dopo `migrate_annotations_table`):

```python
def create_annotation(conn, user_id, image_data, note_text, results_json):
    cur = conn.execute(
        "INSERT INTO comparison_annotations (user_id, image_data, note_text, results_json) VALUES (?, ?, ?, ?)",
        (user_id, image_data, note_text, results_json),
    )
    conn.commit()
    return cur.lastrowid


def list_annotations(conn, user_id):
    rows = conn.execute(
        """SELECT id, note_text, results_json, created_at FROM comparison_annotations
           WHERE user_id = ? ORDER BY created_at DESC, id DESC""",
        (user_id,),
    ).fetchall()
    out = []
    for r in rows:
        try:
            count = len(json.loads(r["results_json"])) if r["results_json"] else 0
        except (ValueError, TypeError):
            count = 0
        out.append({"id": r["id"], "note_text": r["note_text"],
                    "created_at": r["created_at"], "result_count": count})
    return out


def get_annotation(conn, annotation_id, user_id):
    return conn.execute(
        "SELECT * FROM comparison_annotations WHERE id = ? AND user_id = ?",
        (annotation_id, user_id),
    ).fetchone()


def delete_annotation(conn, annotation_id, user_id):
    cur = conn.execute(
        "DELETE FROM comparison_annotations WHERE id = ? AND user_id = ?",
        (annotation_id, user_id),
    )
    conn.commit()
    return cur.rowcount > 0
```

> `json` è già importato in viewer_app.py.

- [ ] **Step 4: Eseguire i test (verde)**

Run: `python -m pytest tests/test_annotations.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add viewer_app.py tests/test_annotations.py
git commit -m "feat(annotations): owner-scoped CRUD helpers"
```

---

### Task 3: Endpoint REST `/api/annotations*`

**Files:**
- Modify: `viewer_app.py` (`do_POST`, `do_GET`, `do_DELETE`)

**Interfaces:**
- Consumes: `require_role`, `get_db`, `get_session`, `create_annotation`, `list_annotations`, `get_annotation`, `delete_annotation`

**Contract dei campi match** (whitelist applicata server-side prima di salvare): `id, collection, image_path, similarity, coarse_similarity, image_type, macro_period, period`.

- [ ] **Step 1: Implementare `POST /api/annotations`** (in `do_POST`, dopo i blocchi annotazioni-non-esistenti, es. vicino agli altri `/api/...`):

```python
        # Crea annotazione personale (iscritto+, owner = sessione)
        if parsed.path == '/api/annotations':
            if not self.require_role('iscritto'):
                return
            session = get_session(self.headers.get('Cookie')) or {}
            user_id = session.get('user_id')
            if not user_id:
                self.send_json({'error': 'Le annotazioni richiedono un account registrato'}, 403)
                return
            image = post_data.get('image') or ''
            note = (post_data.get('note') or '')[:4000]
            if not image or len(image) > 6_000_000:
                self.send_json({'error': 'Immagine mancante o troppo grande'}, 400)
                return
            allowed = ('id', 'collection', 'image_path', 'similarity',
                       'coarse_similarity', 'image_type', 'macro_period', 'period')
            raw = post_data.get('results') or []
            results = [{k: m.get(k) for k in allowed} for m in raw[:30] if isinstance(m, dict)]
            conn = get_db()
            try:
                rid = create_annotation(conn, user_id, image, note, json.dumps(results))
            finally:
                conn.close()
            self.send_json({'success': True, 'id': rid})
            return
```

- [ ] **Step 2: Implementare `GET /api/annotations` e `GET /api/annotations/<id>`** (in `do_GET`, zona API protette):

```python
        # Lista annotazioni dell'utente corrente
        if parsed.path == '/api/annotations':
            if not self.require_role('iscritto'):
                return
            user_id = (get_session(self.headers.get('Cookie')) or {}).get('user_id')
            if not user_id:
                self.send_json({'annotations': []})
                return
            conn = get_db()
            try:
                anns = list_annotations(conn, user_id)
            finally:
                conn.close()
            self.send_json({'annotations': anns})
            return

        # Singola annotazione (owner-only)
        if parsed.path.startswith('/api/annotations/'):
            if not self.require_role('iscritto'):
                return
            user_id = (get_session(self.headers.get('Cookie')) or {}).get('user_id')
            ann_id = parsed.path.rsplit('/', 1)[-1]
            conn = get_db()
            try:
                row = get_annotation(conn, ann_id, user_id) if user_id else None
            finally:
                conn.close()
            if not row:
                self.send_json({'error': 'Annotazione non trovata'}, 404)
                return
            self.send_json({
                'id': row['id'], 'image_data': row['image_data'],
                'note_text': row['note_text'], 'created_at': row['created_at'],
                'results': json.loads(row['results_json']) if row['results_json'] else [],
            })
            return
```

- [ ] **Step 3: Implementare `DELETE /api/annotations/<id>`** (in `do_DELETE`, prima del 404 finale; NON sotto il gate editor di delete-image — gate proprio):

```python
        # Elimina una propria annotazione (iscritto+, owner-only)
        if parsed.path.startswith('/api/annotations/'):
            if not self.require_role('iscritto'):
                return
            user_id = (get_session(self.headers.get('Cookie')) or {}).get('user_id')
            ann_id = parsed.path.rsplit('/', 1)[-1]
            conn = get_db()
            try:
                ok = delete_annotation(conn, ann_id, user_id) if user_id else False
            finally:
                conn.close()
            self.send_json({'success': True} if ok else {'error': 'Annotazione non trovata'}, 200 if ok else 404)
            return
```

> NOTA `do_DELETE`: oggi inizia con `parsed = urlparse`, `query = parse_qs`, poi il ramo `/api/delete-image` gateato `require_role('editor')`. Inserire il ramo `/api/annotations/` PRIMA del `self.send_json({'error': 'Not found'}, 404)` finale. Poiché `delete_annotation` confronta `id` come testo? No: SQLite confronta `id = ?` con la stringa `ann_id` — SQLite fa type-affinity e confronta correttamente l'intero con la stringa numerica, quindi va bene passare `ann_id` stringa.

- [ ] **Step 4: Verifica unit (nessuna regressione)**

Run: `python -m pytest -q`
Expected: PASS (34: 15 preprocess + 13 auth + 6 annotations).

- [ ] **Step 5: Verifica manuale (curl) — owner-scoping end-to-end**

Avvia il server (`python viewer_app.py`, porta 8080; usa ./ceramica.db). Crea due iscritti approvati (registra → login admin `admin2024` → approva), poi:

```bash
# login utente A (cookie jar A), utente B (jar B) — entrambi approvati
# A crea un'annotazione:
curl -s -b A.jar -X POST localhost:8080/api/annotations -d '{"image":"data:image/png;base64,iVBOR","note":"prova A","results":[{"id":"Righetti/x","similarity":80}]}'
# -> {"success": true, "id": 1}
curl -s -b A.jar localhost:8080/api/annotations            # A vede la sua
curl -s -b B.jar localhost:8080/api/annotations            # B vede lista vuota
curl -s -o /dev/null -w "%{http_code}\n" -b B.jar localhost:8080/api/annotations/1   # 404 (non è di B)
curl -s -o /dev/null -w "%{http_code}\n" -b B.jar -X DELETE localhost:8080/api/annotations/1  # 404
curl -s -o /dev/null -w "%{http_code}\n" -b A.jar localhost:8080/api/annotations/1   # 200
# bootstrap admin (cookie da login con password sola) non può creare:
curl -s -o /dev/null -w "%{http_code}\n" -b ADMIN.jar -X POST localhost:8080/api/annotations -d '{"image":"x"}'  # 403
```
Termina il server, `git checkout ceramica.db`, stage solo `viewer_app.py`.

- [ ] **Step 6: Commit**

```bash
git add viewer_app.py
git commit -m "feat(annotations): REST endpoints (create/list/get/delete, owner-scoped)"
```

---

### Task 4: UI — "Salva annotazione" nel modale di comparazione

**Files:**
- Modify: `viewer_app.py` (`get_viewer_html`, blocco UI/JS del modale ML, gated `is_member`)

**Interfaces:** consuma `POST /api/annotations`. Variabili JS esistenti: `mlImageData`, `manualDrawingData`, e i risultati correnti sono `result.similar_items` (passati a `displaySimilarMatches(items, result)`); ogni item ha i campi `id, collection, image_path, similarity, coarse_similarity, image_type, macro_period, period`.

- [ ] **Step 1: Tenere traccia dei risultati correnti**

In `runSimilaritySearch()` (dopo aver ottenuto `result`), salvare i match correnti in una variabile JS a livello di modulo per il salvataggio: aggiungere `lastSearchResults = result.similar_items || [];` (dichiarare `let lastSearchResults = [];` accanto a `let mlImageData`). Questo è l'unico aggancio nello script f-string esistente (graffe già raddoppiate nel file: rispettare il contesto — questa riga non introduce graffe).

- [ ] **Step 2: Aggiungere il pulsante + flusso "Salva annotazione"**

Renderizzare (solo `is_member`) un pulsante "💾 Salva annotazione" nell'area risultati del modale ML. Il relativo JS va aggiunto come **stringa Python semplice** iniettata via `{annotations_js}` nell'f-string (pattern del pannello admin), così si scrive JS normale a graffe singole. Comportamento di `saveAnnotation()`:
- `var image = manualDrawingData || mlImageData; if (!image) { alert('Carica e confronta prima un'immagine'); return; }`
- chiedere la nota con una textarea/`prompt` (`var note = prompt('Nota (facoltativa):', '') || '';`)
- `fetch('/api/annotations', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({image: image, note: note, results: lastSearchResults})})`
- su success → feedback "Annotazione salvata"; su errore → mostrare `data.error`.

- [ ] **Step 3: Verifica (render + curl del flusso)**

Render per-ruolo (il pulsante e il JS solo per member):
```
python -c "import viewer_app as v
for r in ['admin','editor','iscritto',None]:
    h=v.get_viewer_html(r); print(r, 'saveAnnotation' in h)
print('OK')"
```
Atteso: `saveAnnotation` presente per admin/editor/iscritto, assente per None. `python -m pytest -q` → 34 passed. Avvia il server e verifica che `/viewer` risponda 200 e che la pagina contenga il pulsante per un iscritto (lato sorgente). `git checkout ceramica.db`; stage solo `viewer_app.py`.

- [ ] **Step 4: Commit**

```bash
git add viewer_app.py
git commit -m "feat(annotations): 'Salva annotazione' in comparison modal"
```

---

### Task 5: UI — pannello "Le mie annotazioni" (Apri / Riusa / Elimina)

**Files:**
- Modify: `viewer_app.py` (`get_viewer_html`, stesso blocco `{annotations_js}` + markup del pannello, gated `is_member`)

**Interfaces:** consuma `GET /api/annotations`, `GET /api/annotations/<id>`, `DELETE /api/annotations/<id>`. Riusa `umEsc` (escaping testo). Per ricaricare un'immagine: assegna a `mlImageData` e mostra l'anteprima (stesso meccanismo del caricamento file, vedi `mlImageData = e.target.result; preview.src = mlImageData`).

- [ ] **Step 1: Markup del pannello + pulsante d'apertura**

Aggiungere (solo `is_member`) un pulsante "📁 Le mie annotazioni" (vicino a quello ML) e un modale/pannello `#myAnnotationsModal` con un contenitore lista `#annotationsList`. Markup nell'f-string: se contiene graffe JS/CSS letterali vanno raddoppiate; preferire markup statico semplice e popolare via JS.

- [ ] **Step 2: JS del pannello (stringa Python semplice, graffe singole, id numerici, nota escapata)**

In `{annotations_js}` aggiungere:
- `openMyAnnotations()` → mostra il modale e chiama `loadAnnotations()`.
- `loadAnnotations()` → `fetch('/api/annotations')` → `data.annotations`; costruisce le righe. Per ogni riga: nota troncata via `umEsc(a.note_text || '(senza nota)')`, data, `a.result_count + ' match'`, e bottoni con **id numerico**:
  `'<button onclick="openAnnotation(' + Number(a.id) + ')">Apri</button>'`,
  `'<button onclick="reuseAnnotation(' + Number(a.id) + ')">Riusa</button>'`,
  `'<button onclick="deleteAnnotation(' + Number(a.id) + ')">Elimina</button>'`.
- `openAnnotation(id)` → `fetch('/api/annotations/'+id)` → mostra l'immagine (`<img src=data.image_data>`), la nota (`umEsc`), e i match salvati (per ognuno una thumbnail `<img src=umEsc(m.image_path)>` + `umEsc(m.id)` + `m.similarity + '%'`).
- `reuseAnnotation(id)` → `fetch('/api/annotations/'+id)` → `mlImageData = data.image_data;` mostra l'anteprima nel modale ML, chiude il pannello, e chiama `runSimilaritySearch()` (rifà il confronto).
- `deleteAnnotation(id)` → `if(!confirm('Eliminare questa annotazione?')) return;` `fetch('/api/annotations/'+id, {method:'DELETE'})` → `loadAnnotations()`.
- Tutti gli errori di rete → `alert('Errore: ' + e)`.

- [ ] **Step 3: Verifica (render + e2e)**

Render per-ruolo:
```
python -c "import viewer_app as v
for r in ['admin','editor','iscritto',None]:
    h=v.get_viewer_html(r); print(r, 'openMyAnnotations' in h, 'reuseAnnotation' in h)
print('OK')"
```
Atteso: presenti per admin/editor/iscritto, assenti per None; tutti i ruoli renderizzano senza errore f-string. `python -m pytest -q` → 34 passed. Avvia il server e con un iscritto approvato (via curl per creare 1–2 annotazioni) verifica che `GET /api/annotations` e `GET /api/annotations/<id>` restituiscano i dati attesi (il rendering del pannello è verificabile in browser). `git checkout ceramica.db`; stage solo `viewer_app.py`.

- [ ] **Step 4: Commit**

```bash
git add viewer_app.py
git commit -m "feat(annotations): 'Le mie annotazioni' panel (open/reuse/delete)"
```

---

## Self-Review

**Spec coverage:**
- Tabella + migrazione idempotente → Task 1 ✓
- Helper owner-scoped (create/list senza image_data + result_count/get/delete) → Task 2 ✓
- Endpoint POST/GET/GET{id}/DELETE, `iscritto+`, owner-scoped, user_id da sessione, bootstrap-admin 403, cap immagine/nota/30-match, whitelist campi match → Task 3 ✓
- UI "Salva annotazione" (image = manualDrawingData||mlImageData, nota, results correnti) → Task 4 ✓
- Pannello "Le mie annotazioni" con Apri/Riusa/Elimina, escaping nota, id numerici → Task 5 ✓
- Owner-only (no vista admin) → garantito dal filtro `user_id` ovunque ✓
- Test con focus scoping → Task 2 (`test_owner_scoping_get_and_delete`) ✓

**Placeholder scan:** nessun TBD/"handle errors" generico; codice completo per Task 1–3; per Task 4–5 (UI) sono indicati i contratti esatti (funzioni, payload, campi) e le regole f-string/escaping/id-numerici — coerente con come è stata gestita la UI in Fase 1.

**Type consistency:** firme coerenti tra task — `create_annotation(conn, user_id, image_data, note_text, results_json)`, `list_annotations`/`get_annotation`/`delete_annotation(conn, …, user_id)`; campi match whitelist identici in spec, Task 3 e Task 5; `result_count` definito in Task 2 e usato in Task 5; `lastSearchResults`/`mlImageData`/`manualDrawingData` coerenti tra Task 4 e 5.

---

## Note di esecuzione

- Task 1–2 sono TDD puro (pytest verde prima di proseguire). Task 3 è rotte (curl + suite verde). Task 4–5 sono UI nell'f-string (verifica via render per-ruolo + curl sugli endpoint; il pannello in browser).
- Branch di lavoro: `feat/personal-annotations` (creato; spec committata). Base: `main` (Fase 1 già mergiata).
- Dopo i test-su-server, ripristinare sempre `ceramica.db` (`git checkout ceramica.db`) e killare il server; non lasciare processi orfani su 8080.
