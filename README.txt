╔══════════════════════════════════════════════════════════════════════════════╗
║                     CERAMICADATABASE - SISTEMA UNIFICATO                     ║
║                    Visualizzatore Ceramica Archeologica                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

STRUTTURA CARTELLE:
==================
CeramicaDatabase/
├── Degli_Espositi/           # Collezione Degli Espositi
│   ├── TAV_XXII/             # Cartelle per tavola
│   ├── TAV_XXIII/
│   └── ...
├── Righetti/                 # Collezione Righetti
│   ├── Fig.111/              # Cartelle per figura
│   ├── Fig.157/
│   └── ...
├── PDFs/                     # PDF di riferimento
│   ├── 2- Capp.4-5-6+bibliografia.pdf
│   └── Righetti_Thèse_Volume_II.pdf
├── ceramica_metadata.csv     # Metadati unificati (CSV)
├── ceramica_metadata.xlsx    # Metadati unificati (Excel)
├── config.json               # Configurazione sistema
├── viewer_app.py             # Applicazione viewer
└── README.txt                # Questo file


AVVIO VIEWER:
=============
1. Aprire Terminale
2. Navigare alla cartella: cd /Users/enzo/Downloads/CeramicaDatabase
3. Eseguire: python viewer_app.py
4. Il browser si aprirà automaticamente su http://localhost:8080


COMANDI VIEWER:
===============
- ← → : Naviga tra le immagini
- P   : Apri PDF alla pagina di riferimento
- Click su "Rif. PDF" nei metadati: Apre il PDF alla pagina specifica


AGGIUNGERE NUOVA COLLEZIONE:
============================
1. Creare una nuova cartella con il nome della collezione (es: "NuovaCollezione")
2. Copiare le immagini organizzate in sottocartelle (Fig.X o TAV_X)
3. Creare il file CSV dei metadati con le colonne:
   - id, type, period, figure_num, page_num, pottery_id, caption_text,
   - position, rotation, folder, image_path, page_ref, collection, source_pdf
4. Aggiornare ceramica_metadata.csv aggiungendo le nuove righe
5. Copiare il PDF di riferimento in PDFs/
6. Aggiornare config.json aggiungendo la nuova collezione:
   
   "NuovaCollezione": {
     "name": "Nome Visualizzato",
     "pdf": "PDFs/nome_pdf.pdf",
     "color": "#COLORE",
     "description": "Descrizione"
   }


AGGIORNARE METADATI ESISTENTI:
==============================
1. Modificare ceramica_metadata.csv direttamente
2. Oppure modificare i CSV nelle singole cartelle collezione e rieseguire
   lo script di unificazione


COLONNE METADATI:
=================
- id: Nome file immagine
- type: Tipo oggetto (FRAG, COMPLETE, ecc.)
- period: Periodo cronologico
- figure_num: Numero figura/tavola nel PDF
- page_num: Numero pagina nel PDF
- pottery_id: ID ceramica (se presente)
- caption_text: Didascalia/descrizione
- position: Posizione nell'immagine originale
- rotation: Rotazione applicata
- folder: Cartella contenente l'immagine
- image_path: Percorso relativo dell'immagine
- page_ref: Riferimento pagina PDF (es: "p. 76")
- collection: Nome collezione
- source_pdf: Nome file PDF di riferimento


STATISTICHE ATTUALI:
====================
- Degli_Espositi: 232 elementi
- Righetti: 210 elementi
- Pellegrino: 355 elementi (93 figure + 262 tavole ANNEXE)
- TOTALE: 797 elementi

Periodi principali:
- Umm an-Nar / Hili 8 (2700-2000 av. J.-C.)
- Umm an-Nar / Prima Età del Bronzo
- Wadi Suq I-II (2000-1600 av. J.-C.)
- Bronze récent (1600-1250 av. J.-C.)
- Fer I-II (1250-300 av. J.-C.)

Siti principali Pellegrino:
- Masafi-5 (Bronze Récent / Fer I)
- Dibba/Qidfa (Wadi Suq)
- Tell Abraq (Umm an-Nar / Wadi Suq)
- Husn Salut (Bronze Récent / Fer I)


CONTATTI:
=========
Sistema creato con PyPotteryLens
Ultima modifica: 19/12/2025
