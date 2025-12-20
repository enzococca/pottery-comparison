#!/usr/bin/env python3
"""
Migration script to add new decoration analysis fields:
- motivo_decorativo: decorative motif type
- sintassi_decorativa: decorative syntax/pattern arrangement
- scala_metrica: metric scale of the drawing (e.g., "1:3")
- larghezza_cm: actual width in centimeters
- altezza_cm: actual height in centimeters
"""

import sqlite3
import os

DB_PATH = "ceramica.db"

def migrate():
    print("=" * 60)
    print("   ADDING NEW DECORATION ANALYSIS FIELDS")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found")
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(items)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    new_columns = [
        ("motivo_decorativo", "TEXT", "Decorative motif type"),
        ("sintassi_decorativa", "TEXT", "Decorative syntax/pattern"),
        ("scala_metrica", "TEXT", "Drawing scale (e.g., 1:3)"),
        ("larghezza_cm", "REAL", "Actual width in cm"),
        ("altezza_cm", "REAL", "Actual height in cm"),
    ]

    added = 0
    for col_name, col_type, description in new_columns:
        if col_name not in existing_columns:
            print(f"   Adding column: {col_name} ({col_type}) - {description}")
            cursor.execute(f"ALTER TABLE items ADD COLUMN {col_name} {col_type}")
            added += 1
        else:
            print(f"   Column already exists: {col_name}")

    # Add vocabulary entries for the new fields
    vocabulary_entries = [
        # Motivo decorativo (decorative motifs)
        ("motivo_decorativo", "wavy lines"),
        ("motivo_decorativo", "geometric patterns"),
        ("motivo_decorativo", "triangular"),
        ("motivo_decorativo", "crosshatched"),
        ("motivo_decorativo", "incised lines"),
        ("motivo_decorativo", "painted bands"),
        ("motivo_decorativo", "dotted"),
        ("motivo_decorativo", "zigzag"),
        ("motivo_decorativo", "chevron"),
        ("motivo_decorativo", "spiral"),
        ("motivo_decorativo", "hatched triangles"),
        ("motivo_decorativo", "pendant triangles"),
        ("motivo_decorativo", "horizontal bands"),
        ("motivo_decorativo", "vertical lines"),
        ("motivo_decorativo", "net pattern"),

        # Sintassi decorativa (decorative syntax)
        ("sintassi_decorativa", "rim band"),
        ("sintassi_decorativa", "shoulder decoration"),
        ("sintassi_decorativa", "body decoration"),
        ("sintassi_decorativa", "base decoration"),
        ("sintassi_decorativa", "full coverage"),
        ("sintassi_decorativa", "register division"),
        ("sintassi_decorativa", "metope arrangement"),
        ("sintassi_decorativa", "frieze pattern"),
        ("sintassi_decorativa", "random distribution"),
        ("sintassi_decorativa", "symmetrical"),
        ("sintassi_decorativa", "asymmetrical"),

        # Scala metrica (common scales)
        ("scala_metrica", "1:1"),
        ("scala_metrica", "1:2"),
        ("scala_metrica", "1:3"),
        ("scala_metrica", "1:4"),
        ("scala_metrica", "1:5"),
        ("scala_metrica", "2:3"),
    ]

    print("\n   Adding vocabulary entries...")
    for field, value in vocabulary_entries:
        cursor.execute('''
            INSERT OR IGNORE INTO vocabulary (field, value, count)
            VALUES (?, ?, 0)
        ''', (field, value))

    conn.commit()
    conn.close()

    print(f"\n   Migration complete! Added {added} new columns.")
    print("   New fields:")
    print("   - motivo_decorativo: Type of decorative motif")
    print("   - sintassi_decorativa: Arrangement/syntax of decoration")
    print("   - scala_metrica: Scale of the drawing (e.g., 1:3)")
    print("   - larghezza_cm: Actual width in centimeters")
    print("   - altezza_cm: Actual height in centimeters")

    return True

if __name__ == "__main__":
    migrate()
