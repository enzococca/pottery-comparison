#!/usr/bin/env python3
"""
Detect scale bars in ceramic images and calculate pixels per cm.
Uses computer vision to find horizontal ruler lines and OCR to read scale labels.
"""

import os
import re
import cv2
import numpy as np
import sqlite3
from pathlib import Path

# Try OCR
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    print("Warning: pytesseract not installed. OCR disabled. Run: pip install pytesseract")
    HAS_OCR = False

DB_PATH = "ceramica.db"

def detect_scale_bar(image_path):
    """
    Detect scale bar in image and return pixels per cm if found.
    Returns: dict with scale_info or None
    """
    if not os.path.exists(image_path):
        return None

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None

    height, width = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=50, maxLineGap=10)

    if lines is None:
        return None

    # Find horizontal lines (potential scale bars)
    # Scale bars can be anywhere in the image
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Check if line is horizontal (angle < 5 degrees)
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 5 or angle > 175:
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            # Scale bars are usually:
            # - At least 30 pixels long
            # - Not too long (not image border)
            if length > 30 and length < width * 0.6:
                horizontal_lines.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'length': length,
                    'y_pos': (y1 + y2) / 2
                })

    if not horizontal_lines:
        return None

    # Sort by y position (bottom of image first) and length
    horizontal_lines.sort(key=lambda l: (-l['y_pos'], -l['length']))

    # Take the best candidate (longest line near bottom)
    best_line = horizontal_lines[0]

    result = {
        'line_length_px': best_line['length'],
        'line_position': (best_line['x1'], best_line['y1'], best_line['x2'], best_line['y2']),
        'image_size': (width, height)
    }

    # Try OCR to read scale label - scan entire image first
    if HAS_OCR:
        try:
            # First, try OCR on entire image to find "cm" text
            full_text = pytesseract.image_to_string(gray, config='--psm 6')
            result['full_ocr'] = full_text[:200] if full_text else ''

            # Look for scale patterns in full text: "10 cm", "5cm", "0 10 cm", etc.
            scale_patterns = [
                r'(\d+)\s*cm',  # "10 cm" or "10cm"
                r'0\s*[-—]\s*(\d+)\s*cm',  # "0 — 10 cm" or "0-10 cm"
                r'(\d+(?:[.,]\d+)?)\s*(?:cm|CM|Cm)',  # General pattern
            ]

            for pattern in scale_patterns:
                scale_match = re.search(pattern, full_text, re.IGNORECASE)
                if scale_match:
                    cm_value = float(scale_match.group(1).replace(',', '.'))
                    if cm_value > 0:
                        # Estimate pixels per cm based on detected line
                        pixels_per_cm = best_line['length'] / cm_value
                        result['cm_value'] = cm_value
                        result['pixels_per_cm'] = pixels_per_cm
                        result['ocr_match'] = scale_match.group(0)
                        break

            # If no match in full text, try region around the line
            if 'cm_value' not in result:
                y_min = max(0, int(best_line['y_pos']) - 40)
                y_max = min(height, int(best_line['y_pos']) + 40)
                x_min = max(0, min(best_line['x1'], best_line['x2']) - 30)
                x_max = min(width, max(best_line['x1'], best_line['x2']) + 80)

                roi = gray[y_min:y_max, x_min:x_max]
                roi_text = pytesseract.image_to_string(roi, config='--psm 7')

                for pattern in scale_patterns:
                    scale_match = re.search(pattern, roi_text, re.IGNORECASE)
                    if scale_match:
                        cm_value = float(scale_match.group(1).replace(',', '.'))
                        if cm_value > 0:
                            pixels_per_cm = best_line['length'] / cm_value
                            result['cm_value'] = cm_value
                            result['pixels_per_cm'] = pixels_per_cm
                            result['ocr_match'] = scale_match.group(0)
                            break

        except Exception as e:
            result['ocr_error'] = str(e)

    return result

def analyze_sample_images(limit=20):
    """Analyze a sample of images to detect scale bars."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, image_path, collection
        FROM items
        WHERE image_path IS NOT NULL AND image_path != ''
        ORDER BY RANDOM()
        LIMIT ?
    """, (limit,))

    items = cursor.fetchall()
    conn.close()

    print("=" * 60)
    print("   DETECTING SCALE BARS IN IMAGES")
    print("=" * 60)
    print(f"\nAnalyzing {len(items)} sample images...\n")

    results = []

    for item_id, image_path, collection in items:
        if not os.path.exists(image_path):
            continue

        print(f"[{item_id}] {os.path.basename(image_path)}")

        result = detect_scale_bar(image_path)

        if result:
            print(f"   Line detected: {result['line_length_px']:.0f} px")
            if 'pixels_per_cm' in result:
                print(f"   Scale: {result['cm_value']} cm = {result['pixels_per_cm']:.1f} px/cm")
                print(f"   OCR: {result.get('ocr_text', 'N/A')}")
            results.append({
                'id': item_id,
                'collection': collection,
                **result
            })
        else:
            print("   No scale bar detected")

    # Summary
    print("\n" + "=" * 60)
    print("   SUMMARY")
    print("=" * 60)

    detected = len(results)
    with_ocr = len([r for r in results if 'pixels_per_cm' in r])

    print(f"\n   Images analyzed: {len(items)}")
    print(f"   Scale bars detected: {detected}")
    print(f"   With OCR measurement: {with_ocr}")

    if with_ocr > 0:
        avg_ppcm = np.mean([r['pixels_per_cm'] for r in results if 'pixels_per_cm' in r])
        print(f"   Average pixels/cm: {avg_ppcm:.1f}")

    # Group by collection
    by_collection = {}
    for r in results:
        col = r['collection']
        if col not in by_collection:
            by_collection[col] = []
        by_collection[col].append(r)

    print("\n   By collection:")
    for col, items in by_collection.items():
        with_scale = [r for r in items if 'pixels_per_cm' in r]
        if with_scale:
            avg = np.mean([r['pixels_per_cm'] for r in with_scale])
            print(f"      {col}: {len(with_scale)} with scale, avg {avg:.1f} px/cm")
        else:
            print(f"      {col}: {len(items)} lines detected, no OCR")

    return results


def visualize_detection(image_path, output_path=None):
    """Visualize scale bar detection on an image."""

    result = detect_scale_bar(image_path)

    if not result:
        print("No scale bar detected")
        return

    # Load and draw on image
    img = cv2.imread(image_path)

    x1, y1, x2, y2 = result['line_position']
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    # Add text
    text = f"{result['line_length_px']:.0f} px"
    if 'pixels_per_cm' in result:
        text += f" = {result['cm_value']} cm ({result['pixels_per_cm']:.1f} px/cm)"

    cv2.putText(img, text, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")
    else:
        # Save to temp
        out = "/tmp/scale_detection.png"
        cv2.imwrite(out, img)
        print(f"Saved: {out}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Analyze specific image
        image_path = sys.argv[1]
        print(f"Analyzing: {image_path}")
        result = visualize_detection(image_path)
        if result:
            print(f"\nResult: {result}")
    else:
        # Analyze sample
        analyze_sample_images(20)
