#!/usr/bin/env python3
"""
Extract scale information from all PDFs and update database items.
Scans each PDF page for scale patterns like "1:3", "scala 1:2", etc.
Then updates items in the database with the scale from their source page.
"""

import os
import re
import json
import sqlite3
from pathlib import Path

# Try to import PyMuPDF
try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")
    exit(1)

DB_PATH = "ceramica.db"
CONFIG_FILE = "config.json"

def load_config():
    """Load configuration with PDF paths"""
    config_path = Path(__file__).parent / CONFIG_FILE
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def extract_scales_from_pdf(pdf_path):
    """
    Extract scale information from all pages of a PDF.
    Returns dict: {page_number: scale_string}
    """
    if not os.path.exists(pdf_path):
        print(f"   PDF not found: {pdf_path}")
        return {}

    scales_by_page = {}

    # Patterns to find scales
    scale_patterns = [
        r'(?:scala|scale|échelle|escala)\s*[=:]?\s*(\d+)\s*:\s*(\d+)',  # scala 1:3, scale = 1:2
        r'\b(\d+)\s*:\s*(\d+)\b',  # Just 1:3 pattern
    ]

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        print(f"   Scanning {total_pages} pages...")

        for page_idx in range(total_pages):
            page = doc[page_idx]
            text = page.get_text()

            # Look for scale patterns
            for pattern in scale_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        try:
                            num, denom = int(match[0]), int(match[1])
                            # Valid scales are usually 1:1 to 1:10 or 2:3, etc.
                            if 1 <= num <= 10 and 1 <= denom <= 10:
                                scale_str = f"{num}:{denom}"
                                page_num = page_idx + 1

                                # Store the most common archaeological scale (prefer 1:X)
                                if page_num not in scales_by_page:
                                    scales_by_page[page_num] = scale_str
                                elif num == 1:  # Prefer 1:X scales
                                    scales_by_page[page_num] = scale_str
                        except:
                            pass

        doc.close()
        return scales_by_page

    except Exception as e:
        print(f"   Error reading PDF: {e}")
        return {}

def update_database_scales(collection, scales_by_page):
    """
    Update items in database with scales from their source pages.
    """
    if not scales_by_page:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    updated = 0

    # Get all items from this collection
    cursor.execute("""
        SELECT id, page_ref, scala_metrica
        FROM items
        WHERE collection = ? AND page_ref IS NOT NULL AND page_ref != ''
    """, (collection,))

    items = cursor.fetchall()

    for item_id, page_ref, existing_scale in items:
        # Extract page number from page_ref (e.g., "p. 123" -> 123)
        page_match = re.search(r'(\d+)', page_ref)
        if page_match:
            page_num = int(page_match.group(1))

            if page_num in scales_by_page:
                new_scale = scales_by_page[page_num]

                # Only update if no scale exists or it's different
                if not existing_scale or existing_scale != new_scale:
                    cursor.execute("""
                        UPDATE items SET scala_metrica = ? WHERE id = ?
                    """, (new_scale, item_id))
                    updated += 1

    conn.commit()
    conn.close()

    return updated

def main():
    print("=" * 60)
    print("   EXTRACTING SCALES FROM PDFs")
    print("=" * 60)

    # Load config to get PDF paths
    config = load_config()
    collections = config.get('collections', {})

    if not collections:
        print("No collections configured in config.json")
        return

    total_updated = 0
    all_scales = {}

    for collection_name, collection_config in collections.items():
        pdf_path = collection_config.get('pdf', '')
        if not pdf_path:
            continue

        print(f"\n[{collection_name}]")
        print(f"   PDF: {pdf_path}")

        # Get full path
        base_path = Path(__file__).parent
        full_pdf_path = str(base_path / pdf_path)

        # Extract scales
        scales_by_page = extract_scales_from_pdf(full_pdf_path)

        if scales_by_page:
            print(f"   Found scales on {len(scales_by_page)} pages")

            # Show some examples
            examples = list(scales_by_page.items())[:5]
            for page, scale in examples:
                print(f"      Page {page}: {scale}")
            if len(scales_by_page) > 5:
                print(f"      ... and {len(scales_by_page) - 5} more")

            # Update database
            updated = update_database_scales(collection_name, scales_by_page)
            print(f"   Updated {updated} items in database")
            total_updated += updated

            all_scales[collection_name] = scales_by_page
        else:
            print("   No scales found in PDF text")

    print("\n" + "=" * 60)
    print(f"   COMPLETE: Updated {total_updated} items total")
    print("=" * 60)

    # Save scales to JSON for reference
    output_file = "extracted_scales.json"
    with open(output_file, 'w') as f:
        json.dump(all_scales, f, indent=2)
    print(f"\n   Scale data saved to: {output_file}")

if __name__ == "__main__":
    main()
