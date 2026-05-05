#!/usr/bin/env python3
"""
Compute visual embeddings for all ceramic images in the database using
DINOv2-small (Meta's self-supervised ViT, facebook/dinov2-small). Its CLS
token is a 384-dim embedding trained on a much larger and more diverse
corpus than what the v2 classifier saw, and produces strong fine-grained
similarity without requiring decoration labels.
"""

import os
import sys
import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("    COMPUTING IMAGE EMBEDDINGS (DINOv2-small features)")
print("=" * 60)

try:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoImageProcessor
except ImportError:
    print("Installing required packages...")
    os.system("pip install torch transformers pillow")
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoImageProcessor

# Configuration
DB_PATH = "ceramica.db"
OUTPUT_DIR = "ml_model"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "image_embeddings.npz")
METADATA_FILE = os.path.join(OUTPUT_DIR, "embeddings_metadata.json")
DINOV2_MODEL_ID = "facebook/dinov2-small"

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def content_bbox_crop(img, white_thresh=240, padding_ratio=0.03, min_area_ratio=0.02):
    """Crop a PIL image to its non-white content bbox.

    Removes the dominant white background that was biasing the global
    embedding (Issue B in similarity search).
    """
    arr = np.asarray(img.convert('L'))
    h, w = arr.shape
    mask = arr < white_thresh
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_w = x1 - x0
    bbox_h = y1 - y0
    if bbox_w * bbox_h < min_area_ratio * w * h:
        return img
    pad_x = int(bbox_w * padding_ratio)
    pad_y = int(bbox_h * padding_ratio)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    return img.crop((x0, y0, x1 + 1, y1 + 1))


def main():
    # Load DINOv2-small (self-supervised ViT, CLS token = 384-dim)
    print("\n[1/4] Loading DINOv2-small from HuggingFace...")
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
    model = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(device)
    model.eval()
    print(f"   Model loaded (embedding dim: {model.config.hidden_size})")

    def load_image_pil(image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            return content_bbox_crop(img)
        except Exception as e:
            print(f"   Warning: Could not load {image_path}: {e}")
            return None

    # Load data from database
    print("\n[2/4] Loading items from database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Strict filter: keep only items whose decoration is explicitly known and
    # not 'plain'. Pieces with NULL or empty decoration are excluded too —
    # the user wants a strictly decorated corpus for similarity search.
    cursor.execute("""
        SELECT id, image_path, macro_period, period, decoration,
               vessel_type, collection, page_ref, source_pdf
        FROM items
        WHERE image_path IS NOT NULL AND image_path != ''
          AND decoration IS NOT NULL AND TRIM(decoration) != ''
          AND LOWER(TRIM(decoration)) != 'plain'
    """)
    items = cursor.fetchall()
    conn.close()

    print(f"   Found {len(items)} items with images")

    # Extract embeddings
    print("\n[3/4] Extracting embeddings...")
    embeddings = []
    metadata = []
    valid_count = 0

    with torch.no_grad():
        for i, item in enumerate(items):
            item_id, image_path, macro_period, period, decoration, vessel_type, collection, page_ref, source_pdf = item

            # Progress indicator
            if (i + 1) % 50 == 0 or i == 0:
                print(f"   Processing {i+1}/{len(items)}...")

            # Load image and extract DINOv2 CLS embedding
            img_pil = load_image_pil(image_path)
            if img_pil is None:
                continue
            inputs = processor(images=img_pil, return_tensors='pt').to(device)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

            # Normalize embedding for cosine similarity
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            embeddings.append(embedding)
            metadata.append({
                'id': item_id,
                'image_path': image_path,
                'macro_period': macro_period or '',
                'period': period or '',
                'decoration': decoration or '',
                'vessel_type': vessel_type or '',
                'collection': collection or '',
                'page_ref': page_ref or '',
                'source_pdf': source_pdf or '',
                'index': valid_count
            })
            valid_count += 1

    print(f"   Successfully processed {valid_count} images")

    # Save embeddings
    print("\n[4/4] Saving embeddings...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save as compressed numpy file
    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.savez_compressed(EMBEDDINGS_FILE, embeddings=embeddings_array)
    print(f"   Embeddings saved: {EMBEDDINGS_FILE}")
    print(f"   Shape: {embeddings_array.shape}")
    print(f"   Size: {os.path.getsize(EMBEDDINGS_FILE) / 1024 / 1024:.2f} MB")

    # Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump({
            'created': datetime.now().isoformat(),
            'total_images': valid_count,
            'embedding_dim': int(embeddings_array.shape[1]),
            'model': DINOV2_MODEL_ID,
            'items': metadata
        }, f, indent=2)
    print(f"   Metadata saved: {METADATA_FILE}")

    # Statistics
    print("\n" + "=" * 60)
    print("   EMBEDDING COMPUTATION COMPLETE")
    print("=" * 60)
    print(f"\n   Total images: {valid_count}")
    print(f"   Embedding dimension: {embeddings_array.shape[1]}")
    print(f"   Storage: {os.path.getsize(EMBEDDINGS_FILE) / 1024 / 1024:.2f} MB")

    # Distribution by collection
    collections = {}
    periods = {}
    for m in metadata:
        col = m['collection'] or 'Unknown'
        per = m['macro_period'] or 'Unknown'
        collections[col] = collections.get(col, 0) + 1
        periods[per] = periods.get(per, 0) + 1

    print(f"\n   By Collection:")
    for col, count in sorted(collections.items(), key=lambda x: -x[1]):
        print(f"      {col}: {count}")

    print(f"\n   By Period:")
    for per, count in sorted(periods.items(), key=lambda x: -x[1]):
        print(f"      {per}: {count}")

if __name__ == "__main__":
    main()
