# Design — Annotazioni personali sulle immagini di comparazione (Fase 2)

**Data:** 2026-06-26
**Stato:** approvato per l'implementazione (Fase 2)
**Dipende da:** Fase 1 (account/ruoli/`require_role`), già in `main`.
**Scope:** consentire a un iscritto approvato di salvare annotazioni personali — immagine caricata per il confronto (con eventuale disegno appiattito) + nota testuale + snapshot dei risultati — visibili e gestibili solo dal proprietario, senza toccare il catalogo.

## Problema

Dopo la Fase 1, un iscritto approvato può caricare un'immagine, disegnarci sopra e lanciare la ricerca di similarità (modale "ML Classify", `mlImageData` / `manualDrawingData`, `runSimilaritySearch()`), ma **nulla viene salvato**: chiudendo il modale si perde tutto. Manca un modo per conservare il proprio lavoro di confronto (immagine + nota + match trovati) e ritrovarlo in seguito.

## Obiettivi

1. Salvare un'annotazione = **immagine** (la query usata, `manualDrawingData || mlImageData`, quindi col disegno incluso) + **nota** testuale + **snapshot dei match** del confronto corrente.
2. Pannello **"Le mie annotazioni"**: lista delle proprie annotazioni con **Apri** (vedi immagine+nota+match salvati), **Riusa** (ricarica l'immagine nel modale e rifà il confronto), **Elimina**.
3. **Owner-only**: ogni utente vede/gestisce solo le proprie annotazioni.
4. Feature riservata a **`iscritto+`** (stessa soglia della comparazione).
5. Nessuna nuova dipendenza; coerenza con i pattern della Fase 1.

## Non-obiettivi (fuori scope)

- Modifica della nota dopo il salvataggio (scelta: visualizza/elimina/riusa, niente edit).
- Condivisione delle annotazioni tra utenti o vista admin di tutte le annotazioni (owner-only).
- Strokes di disegno editabili: si salva l'immagine **appiattita** (composito), non i tratti vettoriali.
- Annotazioni sugli item del catalogo (quello era un'altra opzione, scartata in Fase 1).
- Annotazioni per il bootstrap-admin a password condivisa (`user_id` assente): le annotazioni sono personali e richiedono un account registrato.

## Modello dati

Nuova tabella, creata in modo idempotente dentro `run_auto_migrations()` tramite un nuovo helper `migrate_annotations_table(conn)` (stesso pattern di `migrate_users_table`):

```
comparison_annotations(
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id      INTEGER NOT NULL,        -- proprietario (users.id); nessuna FK enforced (coerente col repo)
  image_data   TEXT NOT NULL,           -- data URL PNG base64 (immagine query + disegno appiattiti)
  note_text    TEXT,                    -- nota libera dell'utente (può essere vuota)
  results_json TEXT,                    -- JSON: lista match al momento del salvataggio
  created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```
Indice: `CREATE INDEX IF NOT EXISTS idx_annotations_user ON comparison_annotations(user_id)`.

`results_json` è una lista di oggetti con i soli campi utili a ri-visualizzare i match senza ricomputarli:
`[{ "item_id", "collection", "image_path", "similarity", "coarse_similarity", "image_type" }, ...]`
(troncata a max 30 elementi — il default della ricerca).

## Helper DB

Tutti prendono `conn` come primo argomento (unit-testabili su DB in-memory) e sono **owner-scoped**:

- `create_annotation(conn, user_id, image_data, note_text, results_json) -> int` — inserisce, ritorna l'id. `results_json` è una stringa JSON (serializzata dal chiamante).
- `list_annotations(conn, user_id) -> list[dict]` — **solo metadati** per la lista (NON `image_data`): `id`, `note_text`, `created_at`, e `result_count` (numero di match, derivato da `results_json`). Ordinati per `created_at DESC`.
- `get_annotation(conn, annotation_id, user_id) -> sqlite3.Row | None` — record completo (incl. `image_data`, `results_json`) **filtrando per `user_id`** (ritorna None se l'annotazione non esiste o non è del richiedente).
- `delete_annotation(conn, annotation_id, user_id) -> bool` — elimina **filtrando per `user_id`** (usa `cursor.rowcount > 0`, coerente con la convenzione Fase 1).

## Endpoint (tutti `require_role('iscritto')`)

`user_id` ed eventuale identità sono presi dalla sessione (`get_session(...)`), mai dal client.

- `POST /api/annotations` — body `{image, note, results}`.
  - Richiede una sessione con `user_id` **reale** (non None). Se `user_id` è None (bootstrap-admin a password condivisa) → `403 {'error': 'Le annotazioni richiedono un account registrato'}`.
  - Validazioni: `image` presente e data-URL/base64 di dimensione ≤ ~6 MB (altrimenti `400`); `note` troncata/limite 4000 caratteri (`400` se eccede); `results` lista, troncata a 30 elementi e ridotta ai soli campi previsti prima di serializzare in `results_json`.
  - Inserisce via `create_annotation`; risposta `200 {success, id}`.
- `GET /api/annotations` — `200 {annotations: list_annotations(conn, user_id)}` (metadati del solo utente corrente).
- `GET /api/annotations/{id}` — `get_annotation(conn, id, user_id)`; `404` se non trovata/non propria; altrimenti `200` col record completo (`image_data`, `note_text`, `results`, `created_at`).
- `DELETE /api/annotations/{id}` — `delete_annotation`; `200 {success}` se eliminata, `404` se non trovata/non propria.

Routing: `GET/POST /api/annotations` e `GET/DELETE /api/annotations/{id}` si aggiungono alle catene if/elif di `do_GET`/`do_POST`/`do_DELETE`. Per i path con id si usa `startswith('/api/annotations/')` + parsing dell'id (coerente con `/api/v1/items/`).

## UI (nel modale di comparazione, solo `iscritto+`)

Tutti i nuovi controlli sono renderizzati solo quando `is_member` (la stessa condizione del pulsante ML), dentro `get_viewer_html` (f-string → **graffe raddoppiate** nel JS/CSS nuovo, oppure blocco JS come stringa Python semplice iniettata via `{var}` — pattern già usato per il pannello admin in Fase 1).

- **Salva annotazione**: pulsante visibile dopo una ricerca riuscita. Apre un piccolo input nota (textarea) + conferma; al salvataggio invia `image = manualDrawingData || mlImageData`, `note`, `results = ` (i `similar_items` correnti ridotti ai campi previsti) a `POST /api/annotations`. Feedback di esito.
- **Le mie annotazioni**: pulsante che apre un pannello/modale; al caricamento fa `GET /api/annotations` e mostra la lista (anteprima testuale: nota troncata, data, n° match) con azioni per riga:
  - **Apri** → `GET /api/annotations/{id}` → mostra l'immagine (`image_data`) + nota + i match salvati (thumbnail tramite gli URL statici `image_path` del catalogo + punteggi).
  - **Riusa** → carica `image_data` in `mlImageData`, mostra l'anteprima e rilancia/abilita `runSimilaritySearch()`.
  - **Elimina** → `DELETE /api/annotations/{id}` con conferma; ricarica la lista.
- **Escaping**: la nota e ogni stringa proveniente dai dati va inserita con escaping (riuso di `umEsc` / equivalente) o via `textContent`; gli id numerici nei gestori `onclick` vanno passati come numeri (lezione XSS della Fase 1: niente stringhe utente dentro `onclick`).

## Gestione errori

- `POST`: `403` se sessione senza `user_id`; `400` per immagine mancante/troppo grande o nota troppo lunga.
- `GET/DELETE {id}`: `404` se l'id non esiste o non appartiene al richiedente (stesso codice in entrambi i casi, per non rivelare l'esistenza di annotazioni altrui).
- Body malformato: gestito come per le altre rotte (vedi nota sulla fragilità pre-esistente di `do_POST`).

## Sicurezza

- **Owner-scoping** rigoroso: ogni `SELECT`/`DELETE` filtra `WHERE user_id = <sessione>`; impossibile leggere/eliminare annotazioni altrui anche conoscendone l'id.
- **XSS**: la nota è testo libero dell'utente → escaping obbligatorio in fase di render (la Fase 1 ha già mostrato che l'entity-escaping NON basta dentro `onclick`: gli id vanno passati come numeri, le stringhe non vanno mai interpolate in attributi `on*`).
- **Bloat/abuso**: cap su dimensione immagine (~6 MB) e lunghezza nota (4000), max 30 match — evita che il DB sul volume cresca senza limiti.
- Le annotazioni vivono nel DB sul volume Railway (`DATA_DIR`), quindi persistono tra i redeploy (a differenza delle sessioni in-memory).

## Testing

- Unit (pytest, da `tests/test_auth.py` o nuovo `tests/test_annotations.py`) sugli helper, su DB in-memory:
  - `migrate_annotations_table` crea la tabella ed è idempotente.
  - `create_annotation` + `get_annotation` round-trip.
  - `list_annotations` ritorna metadati senza `image_data` e con `result_count` corretto.
  - **Owner-scoping**: `get_annotation`/`delete_annotation` con uno `user_id` diverso ritornano None/False (l'utente A non accede a quelle di B).
  - `delete_annotation` ritorna False per id inesistente.
- Rotte/UI: verifica manuale via `curl` (crea/lista/leggi/elimina come due utenti diversi per dimostrare lo scoping) e in-browser per il pannello.

## Rischi / note

- **Fragilità pre-esistente** (non introdotta qui): `do_POST` fa `json.loads` del body per ogni POST; un body malformato solleva prima del routing (il server si riprende per-richiesta). Fuori scope.
- `image_data` base64 nel DB: scelta dell'utente; il cap di dimensione contiene la crescita. Se in futuro il volume diventasse un problema, si potrà migrare allo storage su file senza cambiare l'API (l'`image_data` resterebbe l'astrazione).
- I match salvati referenziano `image_path` del catalogo: se un item viene poi cancellato/rinominato, la thumbnail del match salvato può risultare mancante — accettabile (snapshot storico).
