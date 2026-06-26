# Design — Account iscritti, ruoli e approvazione admin (Fase 1)

**Data:** 2026-06-26
**Stato:** approvato per l'implementazione (Fase 1)
**Scope:** sistema di registrazione utenti per-account con approvazione admin e ruoli, sostituendo l'attuale auth a due password condivise. Le annotazioni personali sono **Fase 2** (fuori scope qui).

## Problema

Oggi `viewer_app.py` autentica con due password condivise (`admin` / `viewer`, `verify_credentials()` a riga ~1707): non esistono account individuali, la tabella `users` è creata ma inutilizzata, e le funzioni di comparazione (`/api/ml/*`, `/api/3d/reconstruct`) sono completamente pubbliche. L'admin vuole poter far registrare degli "iscritti", approvarli manualmente, dare loro accesso alle sole funzioni di **comparazione** (caricare per comparare, 3D, interrogazione AI) senza poter modificare/cancellare il catalogo, e successivamente promuoverli a **editor**.

## Obiettivi

1. Registrazione self-service con **email + password**; nuovo utente in stato `pending`.
2. L'admin **approva / rifiuta** le registrazioni e può **promuovere/retrocedere** (Iscritto ↔ Editor) e **sospendere** account.
3. Quattro livelli di accesso: `anonimo` < `iscritto` < `editor` < `admin`.
4. Catalogo in **sola lettura pubblico**; comparazione/3D/AI **solo per iscritti approvati**; edit/delete **solo editor+**; gestione utenti **solo admin**.
5. La password `admin` condivisa resta come **super-admin di emergenza** (bootstrap); la password `viewer` condivisa viene ritirata.
6. Nessuna nuova dipendenza; rimanere nello stile monolite stdlib + routing manuale.

## Non-obiettivi (fuori scope Fase 1)

- Annotazioni personali sulle immagini caricate (→ **Fase 2**).
- Verifica email, invio email (niente SMTP), reset password self-service via email.
- Sessioni persistenti su disco (restano in-memory: al riavvio del processo si rifà login). Gli **account** invece persistono nel DB.
- OAuth / identity provider esterni.

## Modello dei ruoli

Ordinamento gerarchico (un ruolo include i permessi di quelli sotto):

| Ruolo | Valore DB | Permessi aggiuntivi |
|---|---|---|
| Anonimo | *(nessuna sessione)* | Sfogliare/cercare il catalogo (sola lettura) |
| Iscritto | `iscritto` | + caricare immagini per comparare, ricostruzione 3D, interrogare l'AI |
| Editor | `editor` | + modificare/cancellare item del catalogo (attuali poteri di scrittura) |
| Admin | `admin` | + gestione utenti (approva/rifiuta/promuovi/sospendi), tutto |

Stato dell'account (indipendente dal ruolo): `pending` → `approved` → (eventualmente) `suspended` / `rejected`. Solo `approved` può effettuare login. L'admin condiviso bootstrap non ha record in tabella (sessione speciale `role=admin`, `user_id=None`).

## Modello dati

Si **attiva** la tabella `users` esistente estendendola tramite il pattern di migrazione idempotente già in uso (`run_auto_migrations()` con `PRAGMA table_info`). Colonne risultanti:

```
users(
  id            INTEGER PK AUTOINCREMENT,
  username      TEXT,                       -- esistente, non più usato per il login (nullable)
  email         TEXT UNIQUE,                -- NUOVO: identità di login (case-insensitive, salvata lowercase)
  password_hash TEXT NOT NULL,              -- NUOVO formato salato: "pbkdf2_sha256$<iter>$<salt_hex>$<hash_hex>"
  role          TEXT DEFAULT 'iscritto',    -- 'iscritto' | 'editor' | 'admin'
  status        TEXT DEFAULT 'pending',     -- 'pending' | 'approved' | 'rejected' | 'suspended'
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  approved_at   TIMESTAMP,                  -- NUOVO
  approved_by   TEXT,                       -- NUOVO: email/identificativo dell'admin che ha approvato
  last_login    TIMESTAMP                   -- NUOVO
)
```

La migrazione aggiunge solo le colonne mancanti (`email`, `status`, `approved_at`, `approved_by`, `last_login`) e va inserita in `run_auto_migrations()`, che per convenzione del repo è la **sorgente di verità** della migrazione in-app (il superset). Lo script standalone `migrate_add_decoration_fields.py` resta un sottoinsieme e non è necessario aggiornarlo.

### Hashing password

Funzioni nuove nello stdlib (nessuna dipendenza):
- `hash_password_salted(password) -> str`: genera salt con `secrets.token_bytes(16)`, deriva con `hashlib.pbkdf2_hmac('sha256', pw, salt, 200_000)`, ritorna stringa `pbkdf2_sha256$200000$<salt_hex>$<hash_hex>`.
- `verify_password_salted(password, stored) -> bool`: parsing della stringa + confronto in tempo costante (`hmac.compare_digest`).

L'attuale `hash_password()` (SHA256 nudo) resta **solo** per la password admin condivisa (variabile d'ambiente `ADMIN_HASH`), per retrocompatibilità.

## Autenticazione e sessioni

- `SESSIONS` (dict in-memory) invariato come storage. Il valore di sessione passa da `{role, created}` a `{user_id, email, role, created}` (`user_id=None`, `email=None` per il bootstrap admin).
- `create_session(role, user_id=None, email=None)` esteso.
- `get_session()` / `get_role()` invariati nella firma; `get_role()` continua a ritornare il ruolo o `None`.
- **Nuovo helper** `require_role(self, min_role) -> bool`: confronta il ruolo corrente con l'ordinamento `{anon:0, iscritto:1, editor:2, admin:3}`; se insufficiente risponde `403 {'error': ...}` (per le API) e ritorna `False`. `require_auth()` esistente (redirect 302) resta per le **pagine**.
- `require_admin()` esistente viene reimplementato come alias di `require_role('admin')` per non rompere i call-site di gestione utenti; i call-site che gestiscono il **catalogo** (update/rotate/flip/vocabolario/delete) passano invece a `require_role('editor')`.
- **Logout**: `POST /api/logout` ora **rimuove** il token da `SESSIONS` lato server (oggi non lo fa) oltre a scadere il cookie.

## Endpoint

### Nuovi

- `POST /api/register` — pubblico. Body `{email, password}`. Validazione: email formato base + non già presente, password lunghezza minima (≥8). Crea utente `role='iscritto'`, `status='pending'`. Risposta `200 {ok, message:"Registrazione ricevuta, in attesa di approvazione"}`. Email duplicata → `409`. **Non** crea sessione (l'utente non è ancora approvato).
- `GET /api/admin/users` — admin. Ritorna elenco utenti (id, email, role, status, created_at, last_login), separabili in pending/attivi lato UI.
- `POST /api/admin/users/approve` — admin. Body `{user_id}`. `status='approved'`, set `approved_at`/`approved_by`.
- `POST /api/admin/users/reject` — admin. Body `{user_id}`. `status='rejected'`.
- `POST /api/admin/users/role` — admin. Body `{user_id, role}` con `role∈{iscritto,editor}` (non si può creare admin da UI). Promuove/retrocede.
- `POST /api/admin/users/suspend` — admin. Body `{user_id, suspend:bool}`. Sospende/riattiva (`suspended`↔`approved`).
- `DELETE /api/admin/users` — admin. Body/query `{user_id}`. Elimina l'account.
- (Opzionale, basso rischio) `POST /api/admin/users/reset-password` — admin. Body `{user_id, new_password}`. Reset manuale (sostituisce il flusso email mancante).

### Modificati

- `POST /api/login` — body può contenere `{email, password}` **oppure** `{password}`:
  - se `email` presente → cerca in `users` (per email lowercase). Se trovato e `verify_password_salted` ok:
    - `status=='approved'` → crea sessione con ruolo/utente, aggiorna `last_login`, `200 {ok, role}`.
    - `status=='pending'` → `403 {error:"in attesa di approvazione", status:'pending'}`.
    - `status∈{rejected,suspended}` → `403 {error:"account non attivo"}`.
    - password errata / email inesistente → `401`.
  - se `email` assente → confronto con `ADMIN_HASH` (bootstrap admin). Match → sessione `role=admin`. La vecchia via `VIEWER_HASH` viene **rimossa**.
- `verify_credentials(password)` — semplificata: ritorna `'admin'` solo se match `ADMIN_HASH`, altrimenti `None` (rimosso il ramo viewer).

### Apertura lettura pubblica

- `GET /viewer` — non fa più `require_auth()` (redirect): serve la pagina a **chiunque**, passando `role` corrente (`None` per anonimo) a `get_viewer_html(role)`.
- `GET /api/data` e `GET /api/config` — resi pubblici (sola lettura). Nessuna nuova esposizione di dato: `/api/v1/items` è già pubblico. `/api/config` continua a includere `user_role` (ora può essere `null`).

### Gating — tabella riepilogo (oggi → nuovo)

| Endpoint(s) | Oggi | Nuovo |
|---|---|---|
| `/api/v1/*`, `/api/data`, `/api/config`, `/viewer` | pubblico / `require_auth` | **pubblico (lettura)** |
| `/api/ml/*`, `/api/3d/reconstruct` | **pubblico** | `require_role('iscritto')` |
| `/api/update-item`, `/api/update-batch`, `/api/rotate-image`, `/api/flip-image`, `/api/vocabulary`, `do_DELETE`(delete-image) | `require_admin` | `require_role('editor')` |
| `/api/register` | — | pubblico |
| `/api/admin/users*` | — | `require_role('admin')` |

## UI

### `WELCOME_PAGE` (pagina iniziale)

- Form **Login**: email + password → `POST /api/login`. Gestione messaggi `pending`/`non attivo`/credenziali errate.
- Form/link **Registrati**: email + password (+ conferma password) → `POST /api/register`, con messaggio di esito "in attesa di approvazione".
- **Accesso amministratore**: campo password singolo (email vuota) per il bootstrap admin.
- È tutto in un'unica pagina (coerente con il template f-string esistente: **ricordarsi di raddoppiare le graffe** `{{ }}` in JS/CSS).

### Viewer

- I pulsanti di **edit/delete/rotate/flip/vocabolario** sono mostrati solo se `role∈{editor,admin}`; i controlli di **comparazione/3D/AI** sono mostrati solo se loggati come iscritto+ (anonimo vede solo browse/ricerca, con un invito a registrarsi quando tenta una funzione riservata).
- **Pannello "Gestione utenti"** (solo admin): tab/sezione con
  - elenco **In attesa**: email, data → [Approva] [Rifiuta]
  - elenco **Attivi**: email, ruolo, ultimo accesso → selettore ruolo [Iscritto/Editor], [Sospendi]/[Riattiva], [Elimina]
- Il backend resta l'unica vera linea di difesa (la UI nasconde, il server impone con `require_role`).

## Gestione errori

- Registrazione: email malformata → `400`; duplicata → `409`; password troppo corta → `400`.
- Tutte le rotte admin: `403` se non admin; `404` se `user_id` inesistente; `400` su ruolo non valido.
- Login: distinguere `401` (credenziali) da `403` (pending/sospeso) per messaggi UI chiari.
- Race/robustezza: `email UNIQUE` a livello DB; inserimento gestito con `try/except` su `IntegrityError` → `409`.

## Testing

- Lo stile dei test del repo è pytest su funzioni pure (`tests/test_preprocess.py`, nessun config file, eseguire dalla root).
- Aggiungere `tests/test_auth.py` con test sulle **funzioni pure** estraibili senza far girare il server:
  - `hash_password_salted` / `verify_password_salted` (round-trip, password errata, formato malformato, salt diversi → hash diversi).
  - logica di ordinamento ruoli (`role_rank` / `require_role` su una funzione pura `role_allows(min, current)`).
  - `verify_credentials` (solo admin, niente viewer).
- Le rotte HTTP si verificano manualmente con `curl` (register → login pending → approva via admin → login ok → accesso `/api/ml/*`). Documentare la sequenza nel piano.

## Rischi / note

- **Sicurezza pregressa** (documentata in CLAUDE.md): cookie senza flag `Secure`, possibili path-traversal sugli endpoint immagine. Fuori scope qui, ma il logout-fix e l'hashing salato riducono parte della superficie. Valutare `Secure` sul cookie in un secondo momento (richiede HTTPS, presente su Railway).
- **Sessioni in-memory**: ogni redeploy Railway disconnette tutti. Accettato. Gli account restano nel DB (sul volume `DATA_DIR`).
- **DB committato**: `ceramica.db` è nel repo; la migrazione che aggiunge colonne a `users` verrà applicata all'avvio sia in locale sia sul volume Railway (`sync_bundled_data` non sovrascrive il volume esistente).
- **Bootstrap del primo admin reale**: in Fase 1 l'admin resta la password condivisa. Promuovere un utente registrato ad `admin` non è esposto in UI (per sicurezza); se servirà, si fa via DB o si aggiunge un endpoint protetto in seguito.

## Fase 2 (promemoria, non in questa spec)

Annotazioni personali: tabella `comparison_annotations(user_id, image ref, note_text, annotation_json, created_at)`, salvataggio immagine caricata sul volume, UI "Le mie annotazioni" nel modale comparazione. Visibili solo all'utente; non modificano il catalogo.
