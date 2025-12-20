#!/usr/bin/env python3
"""
Extract plates/figures from PDFs as high-resolution images.
These plates contain the scale bars and can be calibrated.
"""

import os
import json
import sqlite3
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")
    exit(1)

DB_PATH = "ceramica.db"
CONFIG_FILE = "config.json"
OUTPUT_DIR = "plates"
DPI = 200  # Resolution for extraction

def load_config():
    config_path = Path(__file__).parent / CONFIG_FILE
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def get_pages_with_items():
    """Get list of pages that have items in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT collection, page_ref, source_pdf
        FROM items
        WHERE page_ref IS NOT NULL AND page_ref != ''
        ORDER BY collection, page_ref
    """)

    pages = cursor.fetchall()
    conn.close()

    # Parse page numbers
    import re
    result = {}
    for collection, page_ref, source_pdf in pages:
        if collection not in result:
            result[collection] = set()

        # Extract page number from various formats: "p. 123", "pp. 45-46", "123"
        matches = re.findall(r'(\d+)', page_ref)
        for m in matches:
            result[collection].add(int(m))

    return result

def extract_plates(pdf_path, collection_name, pages_to_extract, output_dir):
    """Extract specific pages from PDF as images."""

    if not os.path.exists(pdf_path):
        print(f"   PDF not found: {pdf_path}")
        return []

    # Create output directory
    collection_dir = os.path.join(output_dir, collection_name)
    os.makedirs(collection_dir, exist_ok=True)

    extracted = []

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        print(f"   PDF has {total_pages} pages")
        print(f"   Extracting {len(pages_to_extract)} pages with items...")

        for page_num in sorted(pages_to_extract):
            if page_num < 1 or page_num > total_pages:
                continue

            page = doc[page_num - 1]  # 0-indexed

            # Render at high resolution
            zoom = DPI / 72  # 72 is default PDF DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Save as PNG
            output_file = os.path.join(collection_dir, f"page_{page_num:04d}.png")
            pix.save(output_file)

            extracted.append({
                'collection': collection_name,
                'page_num': page_num,
                'file_path': output_file,
                'width': pix.width,
                'height': pix.height
            })

            if len(extracted) % 10 == 0:
                print(f"      Extracted {len(extracted)} pages...")

        doc.close()

    except Exception as e:
        print(f"   Error: {e}")

    return extracted

def create_plates_database(plates):
    """Create database table for plates with calibration data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create plates table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection TEXT NOT NULL,
            page_num INTEGER NOT NULL,
            file_path TEXT,
            width INTEGER,
            height INTEGER,
            pixels_per_cm REAL,
            scale_text TEXT,
            calibrated_at TIMESTAMP,
            UNIQUE(collection, page_num)
        )
    ''')

    # Insert plates
    for plate in plates:
        cursor.execute('''
            INSERT OR REPLACE INTO plates (collection, page_num, file_path, width, height)
            VALUES (?, ?, ?, ?, ?)
        ''', (plate['collection'], plate['page_num'], plate['file_path'],
              plate['width'], plate['height']))

    conn.commit()
    conn.close()

    print(f"\n   Created plates table with {len(plates)} entries")

def main():
    print("=" * 60)
    print("   EXTRACTING PLATES FROM PDFs")
    print("=" * 60)

    # Load config
    config = load_config()
    collections = config.get('collections', {})

    if not collections:
        print("No collections configured")
        return

    # Get pages that have items
    pages_with_items = get_pages_with_items()

    print(f"\nPages with database items:")
    for col, pages in pages_with_items.items():
        print(f"   {col}: {len(pages)} pages")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_plates = []

    for collection_name, collection_config in collections.items():
        pdf_path = collection_config.get('pdf', '')
        if not pdf_path:
            continue

        print(f"\n[{collection_name}]")
        print(f"   PDF: {pdf_path}")

        base_path = Path(__file__).parent
        full_pdf_path = str(base_path / pdf_path)

        pages = pages_with_items.get(collection_name, set())
        if not pages:
            print("   No pages to extract")
            continue

        plates = extract_plates(full_pdf_path, collection_name, pages, OUTPUT_DIR)
        all_plates.extend(plates)

        print(f"   Extracted {len(plates)} plates")

    # Create database table
    if all_plates:
        create_plates_database(all_plates)

    # Summary
    print("\n" + "=" * 60)
    print("   EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\n   Total plates extracted: {len(all_plates)}")
    print(f"   Output directory: {OUTPUT_DIR}/")
    print(f"\n   Next: Run the plate calibration viewer to calibrate each plate")

if __name__ == "__main__":
    main()
