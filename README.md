# CeramicaDatabase - Archaeological Ceramic Viewer

A unified web-based viewer and management system for archaeological ceramic collections from the Arabian Peninsula.

## Live Demo

**[View Demo on Railway](https://pottery-comparison.up.railway.app/)**

**Credentials:**
- **Admin**: `admin2024` (full access: edit, delete, rotate)
- **Viewer**: `viewer2024` (browse and search only)

## Collections

| Collection | Items | Description |
|------------|-------|-------------|
| **Degli Espositi** | 232 | Ceramics from Sequence T7 - ST1, Umm an-Nar and Wadi Suq periods |
| **Righetti** | 210 | Ceramics from Hili 8, Umm an-Nar period (3rd millennium BCE) |
| **Pellegrino** | 355 | Ceramics from Masafi-5, Dibba, Tell Abraq - Late Bronze Age / Iron I |

**Total: 797 artifacts**

## Chronological Periods

- **Umm an-Nar** (2700-2000 BCE) - Early Bronze Age
- **Wadi Suq** (2000-1600 BCE) - Middle Bronze Age
- **Late Bronze Age** (1600-1250 BCE)
- **Iron Age I-II** (1250-300 BCE)

## Features

- Multi-collection viewer with separate tabs
- Filters by macro-period, period, decoration, vessel type, part type
- Full-text search across metadata
- Dynamic autocomplete for editing fields (add new terms)
- Keyboard navigation (← →, P for PDF, E for edit, R for rotate)
- Role-based access (Admin vs Viewer)
- SQLite database backend
- Public REST API for ML integration

## ML & API Access

Public endpoints for machine learning and research integration:

```
GET /api/v1/items           # All artifacts with metadata
GET /api/v1/items/{id}      # Single artifact by ID
GET /api/v1/vocabulary      # Controlled vocabulary terms
GET /api/v1/periods         # All chronological periods
GET /api/v1/stats           # Database statistics
GET /{collection}/{folder}/{image}.png  # Direct image access
```

## Local Development

```bash
# Install dependencies
pip install pandas opencv-python numpy

# Run locally
python viewer_app.py

# Opens browser at http://localhost:8080
```

## Tech Stack

- **Backend**: Python 3 with built-in HTTP server
- **Database**: SQLite (migrated from CSV)
- **Frontend**: Single-page application (HTML/CSS/JavaScript)
- **Deployment**: Railway

## Author

**Enzo Cocca** - [GitHub](https://github.com/enzococca)

## License

This project is intended for archaeological research purposes.
