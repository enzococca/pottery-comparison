# CeramicaDatabase - Archaeological Ceramic Viewer

Sistema unificato per la visualizzazione e gestione di collezioni di ceramica archeologica.

## Collezioni Incluse

| Collezione | Elementi | Descrizione |
|------------|----------|-------------|
| **Degli Espositi** | 232 | Ceramica da Sequenza T7 - ST1, Umm an-Nar e Wadi Suq |
| **Righetti** | 210 | Ceramica da Hili 8, periodo Umm an-Nar (III millennio) |
| **Pellegrino** | 355 | Ceramica da Masafi-5, Dibba, Tell Abraq - Bronze Récent / Fer I |

**Totale: 797 elementi**

## Demo Online

👉 **[Visualizza la Demo](https://enzococca.github.io/pottery-comparison/)**

Per accedere alla demo, contattare l'autore per ottenere le credenziali.

## Periodi Cronologici

- **Umm an-Nar** (2700-2000 a.C.) - Prima Età del Bronzo
- **Wadi Suq** (2000-1600 a.C.) - Media Età del Bronzo
- **Bronze Récent / Fer I** (1600-600 a.C.) - Tarda Età del Bronzo / Età del Ferro

## Funzionalità

- Visualizzazione multi-collezione con tab separati
- Filtri per periodo, decorazione, tipo vaso, parte
- Ricerca testuale nei metadati
- Navigazione con frecce ← →
- Visualizzazione metadati dettagliati

### Funzionalità Desktop (viewer_app.py)

Eseguendo localmente con Python si ottengono funzionalità aggiuntive:

- Apertura PDF alla pagina di riferimento
- Rotazione immagini
- Modifica metadati
- Eliminazione elementi

```bash
# Installazione dipendenze
pip install pandas opencv-python numpy

# Avvio
python viewer_app.py
```

## Struttura Dati

I metadati sono organizzati in CSV con le seguenti colonne principali:

- `id` - Identificativo univoco immagine
- `period` - Periodo cronologico
- `collection` - Nome collezione
- `vessel_type` - Tipo di vaso (jar, bowl, cup, plate, pot)
- `decoration` - Decorazione (decorated, plain)
- `part_type` - Parte del vaso (complete, rim, base, wall, fragment)

## Autore

**Enzo Cocca**

## Licenza

Questo progetto è destinato a scopi di ricerca archeologica.
