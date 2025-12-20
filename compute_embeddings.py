#!/usr/bin/env python3
"""
Compute visual embeddings for all ceramic images in the database.
Uses ResNet18 to extract feature vectors for similarity search.
"""

import os
import sys
import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("    COMPUTING IMAGE EMBEDDINGS FOR SIMILARITY SEARCH")
print("=" * 60)

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from PIL import Image
except ImportError:
    print("Installing required packages...")
    os.system("pip install torch torchvision pillow")
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from PIL import Image

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
DB_PATH = "ceramica.db"
OUTPUT_DIR = "ml_model"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "image_embeddings.npz")
METADATA_FILE = os.path.join(OUTPUT_DIR, "embeddings_metadata.json")

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create feature extractor (ResNet18 without final classification layer)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final FC layer - output is 512-dimensional
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 512)
        return x

# Image transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image(image_path):
    """Load and transform an image."""
    try:
        img = Image.open(image_path).convert('RGB')
        return transform(img)
    except Exception as e:
        print(f"   Warning: Could not load {image_path}: {e}")
        return None

def main():
    # Load model
    print("\n[1/4] Loading feature extractor model...")
    model = FeatureExtractor().to(device)
    model.eval()
    print(f"   Model loaded (output: 512-dimensional vectors)")

    # Load data from database
    print("\n[2/4] Loading items from database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, image_path, macro_period, period, decoration,
               vessel_type, collection, page_ref, source_pdf
        FROM items
        WHERE image_path IS NOT NULL AND image_path != ''
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

            # Load image
            img_tensor = load_image(image_path)
            if img_tensor is None:
                continue

            # Extract embedding
            img_tensor = img_tensor.unsqueeze(0).to(device)
            embedding = model(img_tensor).cpu().numpy().flatten()

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
            'embedding_dim': 512,
            'model': 'ResNet18',
            'items': metadata
        }, f, indent=2)
    print(f"   Metadata saved: {METADATA_FILE}")

    # Statistics
    print("\n" + "=" * 60)
    print("   EMBEDDING COMPUTATION COMPLETE")
    print("=" * 60)
    print(f"\n   Total images: {valid_count}")
    print(f"   Embedding dimension: 512")
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
