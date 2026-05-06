"""One-shot classifier: assign every items.image_path to one of 5
image_type labels using CLIP ViT-B/32 zero-shot, write the chosen label
plus its softmax confidence back to the DB.

Run after every corpus change (rare). Idempotent: rerunning re-classifies
all rows.
"""
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DB_PATH = "ceramica.db"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

PROMPTS = {
    "decorated_vessel": "a technical drawing of an archaeological pottery vessel with decorative patterns",
    "profile_section":  "a thin profile section drawing of a pottery rim or base, no decoration",
    "decoration_only":  "a close-up of decorative ornament patterns on archaeological ceramics",
    "decorated_photo":  "a color photograph of an archaeological ceramic vessel with decoration",
    "plain_vessel":     "a plain undecorated pottery vessel drawing",
    "unclassified":     "a scale bar, text label, or unrelated archaeological figure",
}


def main():
    device = torch.device("cpu")
    print(f"Loading {CLIP_MODEL_ID}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    labels = list(PROMPTS.keys())
    prompts = [PROMPTS[k] for k in labels]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, image_path FROM items WHERE image_path IS NOT NULL AND image_path != ''")
    rows = cur.fetchall()
    print(f"Classifying {len(rows)} items...")

    counts = {k: 0 for k in labels}
    written = 0
    skipped = 0
    with torch.no_grad():
        for i, (item_id, image_path) in enumerate(rows):
            if not Path(image_path).exists():
                skipped += 1
                continue
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception:
                skipped += 1
                continue
            img_inputs = processor(images=img, return_tensors="pt").to(device)
            img_emb = model.get_image_features(**img_inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sims = (img_emb @ text_emb.T).squeeze(0)
            probs = torch.softmax(sims * 100.0, dim=-1)
            best = int(torch.argmax(probs))
            label = labels[best]
            confidence = float(probs[best])
            cur.execute(
                "UPDATE items SET image_type = ?, image_type_confidence = ? WHERE id = ?",
                (label, confidence, item_id),
            )
            counts[label] += 1
            written += 1
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)}...")
    conn.commit()
    conn.close()

    print(f"\nWrote {written} rows ({skipped} skipped — image missing).")
    print("Distribution:")
    for k, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {c}")


if __name__ == "__main__":
    main()
