#!/usr/bin/env python3
"""Compute decoration-focused DINOv2 embeddings (v3).

For every items row whose image_type belongs to the kept set, run the
shared preprocess_for_dinov2 pipeline (bbox crop → silhouette mask →
valid_patch_mask), forward DINOv2-small and read last_hidden_state, and
store three tensors:

- cls_embeddings    (N, 384)         fp32, normalized
- mean_embeddings   (N, 384)         fp32, normalized — primary ranking key
- patch_embeddings  (N, 256, 384)    fp16, NOT normalized (rerank
                                     normalizes per-patch on the fly)

plus valid_patch_masks (N, 256) packed as bool, in
ml_model/embeddings_v3.npz. Metadata in embeddings_metadata_v3.json.
"""
import os
import json
import sqlite3
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

from preprocess import preprocess_for_dinov2, ink_density

# Configuration
DB_PATH = "ceramica.db"
OUTPUT_DIR = "ml_model"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "image_embeddings_v3.npz")
METADATA_FILE = os.path.join(OUTPUT_DIR, "embeddings_metadata_v3.json")
DINOV2_MODEL_ID = "facebook/dinov2-small"

# Search-time filter rule (mirrored in viewer_app.py).
KEPT_TYPES = {"decorated_vessel", "decoration_only"}
LOW_CONF_KEEP = "unclassified"
LOW_CONF_THRESHOLD = 0.4

# Last-line defence: even after image_type filter, drop any item whose
# image is essentially blank inside the content bbox.
PROFILE_ONLY_DENSITY = 0.05


def is_kept(image_type, conf):
    if image_type in KEPT_TYPES:
        return True
    if image_type == LOW_CONF_KEEP and (conf or 0.0) < LOW_CONF_THRESHOLD:
        return True
    return False


def main():
    print("=" * 60)
    print("    COMPUTING IMAGE EMBEDDINGS V3 (decoration-focused)")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/4] Loading DINOv2-small...")
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
    model = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(device).eval()
    print(f"   Hidden size: {model.config.hidden_size}")

    print("\n[2/4] Loading items...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, image_path, macro_period, period, decoration,
               vessel_type, collection, page_ref, source_pdf,
               image_type, image_type_confidence
        FROM items
        WHERE image_path IS NOT NULL AND image_path != ''
          AND decoration IS NOT NULL AND TRIM(decoration) != ''
          AND LOWER(TRIM(decoration)) != 'plain'
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"   Found {len(rows)} candidate items (after decoration filter)")

    print("\n[3/4] Extracting embeddings...")
    cls_list, mean_list, patch_list, valid_list, metadata = [], [], [], [], []
    skipped_image_type = 0
    skipped_density = 0
    skipped_load = 0
    valid_count = 0

    with torch.no_grad():
        for i, row in enumerate(rows):
            (item_id, image_path, macro_period, period, decoration,
             vessel_type, collection, page_ref, source_pdf,
             image_type, image_type_confidence) = row

            if (i + 1) % 50 == 0 or i == 0:
                print(f"   {i+1}/{len(rows)}...")

            if not is_kept(image_type, image_type_confidence):
                skipped_image_type += 1
                continue

            try:
                raw = Image.open(image_path).convert("RGB")
            except Exception:
                skipped_load += 1
                continue

            if ink_density(raw) < PROFILE_ONLY_DENSITY:
                skipped_density += 1
                continue

            final_img, valid_mask = preprocess_for_dinov2(raw)

            if not valid_mask.any():
                # No content patches survive the silhouette mask + density
                # check (e.g., a profile-only drawing whose bbox-cropped
                # decoration interior is all blank). Rerank would produce
                # NaN / division by zero.
                skipped_density += 1
                continue

            inputs = processor(images=final_img, return_tensors="pt").to(device)
            out = model(**inputs)
            hidden = out.last_hidden_state[0]                 # (257, 384)

            cls = hidden[0]                                   # (384,)
            patches = hidden[1:]                              # (256, 384)
            mean_emb = patches.mean(dim=0)                    # (384,)

            cls_n = (cls / (cls.norm() + 1e-8)).cpu().numpy().astype(np.float32)
            mean_n = (mean_emb / (mean_emb.norm() + 1e-8)).cpu().numpy().astype(np.float32)
            patch_arr = patches.cpu().numpy().astype(np.float16)

            cls_list.append(cls_n)
            mean_list.append(mean_n)
            patch_list.append(patch_arr)
            valid_list.append(valid_mask)

            metadata.append({
                "id": item_id,
                "image_path": image_path,
                "macro_period": macro_period or "",
                "period": period or "",
                "decoration": decoration or "",
                "vessel_type": vessel_type or "",
                "collection": collection or "",
                "page_ref": page_ref or "",
                "source_pdf": source_pdf or "",
                "image_type": image_type or "",
                "image_type_confidence": float(image_type_confidence or 0.0),
                "index": valid_count,
            })
            valid_count += 1

    print(f"\n   Wrote {valid_count} embeddings")
    print(f"   Skipped: image_type={skipped_image_type}, density={skipped_density}, load_error={skipped_load}")

    print("\n[4/4] Saving v3 npz...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cls_arr = np.stack(cls_list)
    mean_arr = np.stack(mean_list)
    patch_arr_full = np.stack(patch_list)              # (N, 256, 384) fp16
    valid_arr = np.stack(valid_list).astype(bool)      # (N, 256)
    np.savez_compressed(
        EMBEDDINGS_FILE,
        cls_embeddings=cls_arr,
        mean_embeddings=mean_arr,
        patch_embeddings=patch_arr_full,
        valid_patch_masks=valid_arr,
    )
    print(f"   {EMBEDDINGS_FILE} ({os.path.getsize(EMBEDDINGS_FILE) / 1024 / 1024:.1f} MB)")
    print(f"   shapes: cls={cls_arr.shape} mean={mean_arr.shape} patch={patch_arr_full.shape} valid={valid_arr.shape}")

    with open(METADATA_FILE, "w") as f:
        json.dump({
            "created": datetime.now().isoformat(),
            "total_images": valid_count,
            "embedding_dim": int(cls_arr.shape[1]),
            "model": DINOV2_MODEL_ID,
            "version": 3,
            "items": metadata,
        }, f, indent=2)
    print(f"   {METADATA_FILE}")


if __name__ == "__main__":
    main()
