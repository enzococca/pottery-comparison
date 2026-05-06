# Decoration-Only Similarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace global DINOv2 CLS-token similarity with a decoration-focused pipeline that masks out the vessel silhouette, ranks by mean-of-patch embeddings, then re-ranks the top-50 with patch-level set similarity. Add a CLIP-derived `image_type` column on `items` to filter profile-only / unrelated images at search time.

**Architecture:** A shared `preprocess.py` module produces (masked image, valid_patch_mask) for both corpus build and query time. `compute_embeddings.py` writes a v3 npz with cls + mean-patch + per-patch embeddings. `viewer_app.py` does coarse cosine rank on mean embeddings → patch-level Chamfer rerank on top-50. CLIP zero-shot classifies every item once into 5 categories stored in `ceramica.db`. v2 embeddings stay on disk so a regression can be rolled back instantly.

**Tech Stack:** Python 3.13, PyTorch CPU, HuggingFace transformers (DINOv2-small + CLIP ViT-B/32), OpenCV, SQLite, pytest (newly introduced for `preprocess.py` only).

**Spec:** [docs/superpowers/specs/2026-05-06-decoration-only-similarity-design.md](../specs/2026-05-06-decoration-only-similarity-design.md)

---

## File map

**Created:**
- `preprocess.py` — pure functions for image preprocessing (shared by corpus build + query time)
- `classify_image_types.py` — one-shot script that runs CLIP zero-shot over `items` and writes `image_type` + `image_type_confidence` back to the DB
- `tests/test_preprocess.py` — pytest unit tests for `preprocess.py`
- `ml_model/embeddings_v3.npz` — generated artefact (cls + mean + patch + valid_masks)
- `ml_model/embeddings_metadata_v3.json` — generated metadata with `image_type` per item

**Modified:**
- `requirements.txt` — add `pytest` (dev), `open-clip-torch` not needed (we use HF `transformers.CLIPModel`)
- `ceramica.db` — `ALTER TABLE items ADD COLUMN image_type TEXT, image_type_confidence REAL`
- `compute_embeddings.py` — full rewrite using `preprocess.py`, stores three embedding tensors + valid masks, filters by `image_type`
- `viewer_app.py`:
  - `find_similar_images()` — replace path with `_v3` implementation that does coarse rank + image_type filter + patch rerank; remove dead `_content_bbox_crop` (now in `preprocess.py`)
  - `load_embeddings()` — load v3 npz with v2 fallback
  - frontend (`get_viewer_html`): match-card score reads "X% similar (decoration)" with tooltip showing global score; subtitle updated

---

## Conventions

- **Python interpreter for local commands**: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3` (the system Anaconda Python at `/Users/enzo/anaconda3/bin/python` hangs on `transformers` import — confirmed earlier in the session).
- **Run all commands from project root**: `/Users/enzo/Downloads/CeramicaDatabase`.
- **Commit style**: conventional commits (feat/fix/chore/docs/refactor). Author **Enzo Cocca** (no Claude co-author trailer).
- **No `git push`** until the user explicitly says to deploy. The plan creates 4 logical commits; user pushes them when satisfied with local verification.

---

## Task 1: Add pytest dev dependency + tests directory

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Read current requirements.txt**

```bash
cat requirements.txt
```

- [ ] **Step 2: Append pytest**

Edit `requirements.txt` — append a new line `pytest>=8.0.0` after the existing dependencies. Final file should look like:

```
pandas>=2.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
Pillow>=9.0.0
PyMuPDF>=1.23.0
requests>=2.31.0
--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
transformers>=4.40.0
pytest>=8.0.0
```

- [ ] **Step 3: Install pytest locally**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pip install 'pytest>=8.0.0'
```

Expected: pytest installs without error.

- [ ] **Step 4: Create empty test package**

Create file `tests/__init__.py` with empty content.

- [ ] **Step 5: Verify pytest discovers nothing yet**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/ -v
```

Expected: `no tests ran`. (The collection succeeds, no tests yet — that's correct.)

---

## Task 2: TDD `bbox_crop` in `preprocess.py`

**Files:**
- Create: `preprocess.py`
- Create: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing test for `bbox_crop`**

Create `tests/test_preprocess.py` with:

```python
"""Pure-function tests for preprocess.py."""
from PIL import Image, ImageDraw
import numpy as np
import pytest

from preprocess import bbox_crop


def _white(w=200, h=200):
    return Image.new('RGB', (w, h), color='white')


def _circle(w=200, h=200, radius=30, fill='black'):
    img = _white(w, h)
    d = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
              outline=fill, width=2)
    return img


def test_bbox_crop_white_image_unchanged():
    img = _white(200, 200)
    out = bbox_crop(img)
    assert out.size == img.size


def test_bbox_crop_circle_at_center_crops_to_circle():
    img = _circle(200, 200, radius=30)
    out = bbox_crop(img, padding_ratio=0.0)
    # Original 200x200; circle bbox is ~62x62; with padding 0 should be ~62
    assert 55 < out.size[0] < 75
    assert 55 < out.size[1] < 75


def test_bbox_crop_padding_grows_bbox():
    img = _circle(200, 200, radius=30)
    out0 = bbox_crop(img, padding_ratio=0.0)
    out10 = bbox_crop(img, padding_ratio=0.1)
    assert out10.size[0] > out0.size[0]
```

- [ ] **Step 2: Run test, verify it fails**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/test_preprocess.py -v
```

Expected: `ModuleNotFoundError: No module named 'preprocess'`.

- [ ] **Step 3: Create `preprocess.py` with `bbox_crop`**

Create `preprocess.py`:

```python
"""Shared image preprocessing for DINOv2 similarity pipeline.

Pure functions only — no global state, no side effects beyond return values.
Used by both compute_embeddings.py (corpus build) and viewer_app.py (query
time) so the corpus and queries pass through identical transforms.
"""
from PIL import Image
import numpy as np
import cv2


def bbox_crop(pil_img, white_thresh=240, padding_ratio=0.03, min_area_ratio=0.02):
    """Crop a PIL image to its non-white content bbox, with proportional padding.

    If the dark-pixel bbox is smaller than `min_area_ratio` of the full image
    (likely a stray speck on an otherwise white page), return the input
    unchanged.
    """
    arr = np.asarray(pil_img.convert('L'))
    h, w = arr.shape
    mask = arr < white_thresh
    if not mask.any():
        return pil_img
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_w = x1 - x0
    bbox_h = y1 - y0
    if bbox_w * bbox_h < min_area_ratio * w * h:
        return pil_img
    pad_x = int(bbox_w * padding_ratio)
    pad_y = int(bbox_h * padding_ratio)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    return pil_img.crop((x0, y0, x1 + 1, y1 + 1))
```

- [ ] **Step 4: Run test, verify it passes**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/test_preprocess.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Do NOT commit yet** (Task 6 collects everything for one commit).

---

## Task 3: TDD `ink_density`

**Files:**
- Modify: `preprocess.py` (add `ink_density`)
- Modify: `tests/test_preprocess.py` (append tests)

- [ ] **Step 1: Append failing tests**

Append to `tests/test_preprocess.py`:

```python
from preprocess import ink_density


def test_ink_density_white_image_is_zero():
    assert ink_density(_white()) == 0.0


def test_ink_density_circle_outline_is_nonzero():
    img = _circle(200, 200, radius=30)
    assert ink_density(img) > 0.02


def test_ink_density_uses_bbox_not_full_image():
    """A 200x200 image with a 60x60 dense scribble at one corner should
    have density measured inside the bbox of the scribble, not over the
    full image (else density would be tiny just because the page is large)."""
    img = _white(200, 200)
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 70, 70], fill='black')  # 60x60 fully filled
    # bbox is the 60x60 square; density inside should be ~1.0
    assert ink_density(img) > 0.9
```

- [ ] **Step 2: Run tests, verify the new ones fail with ImportError**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/test_preprocess.py -v
```

Expected: 3 passed (existing) + 3 errors (`ImportError: cannot import name 'ink_density'`).

- [ ] **Step 3: Add `ink_density` to `preprocess.py`**

Append to `preprocess.py`:

```python
def ink_density(pil_img, white_thresh=240):
    """Fraction of dark pixels inside the image's content bbox.

    Profile-only / outline drawings register low density (~2-5%); decorated
    surfaces register much higher (~15-30%). Used as a quick first-pass
    filter at corpus build time before running DINOv2.
    """
    arr = np.asarray(pil_img.convert('L'))
    mask = arr < white_thresh
    if not mask.any():
        return 0.0
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox = arr[y0:y1 + 1, x0:x1 + 1]
    return float((bbox < white_thresh).mean())
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/test_preprocess.py -v
```

Expected: 6 passed.

---

## Task 4: TDD `extract_decoration_region`

**Files:**
- Modify: `preprocess.py`
- Modify: `tests/test_preprocess.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_preprocess.py`:

```python
from preprocess import extract_decoration_region


def _circle_with_inner_lines(w=200, h=200, radius=80):
    """White image, big circle outline, and horizontal hatching INSIDE the circle.
    The decoration extraction should keep the hatching, drop the outline."""
    img = _white(w, h)
    d = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
              outline='black', width=3)
    for y in range(cy - radius + 15, cy + radius - 15, 8):
        d.line([cx - radius + 20, y, cx + radius - 20, y],
               fill='black', width=2)
    return img


def test_extract_decoration_white_image_unchanged():
    img = _white()
    out = extract_decoration_region(img)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(img))


def test_extract_decoration_keeps_inner_pattern():
    img = _circle_with_inner_lines(200, 200, radius=80)
    inp_dark = (np.asarray(img.convert('L')) < 240).sum()
    out = extract_decoration_region(img)
    out_dark = (np.asarray(out.convert('L')) < 240).sum()
    # Some content survives (the hatching). The outline should be removed.
    assert out_dark > 50
    # And we shouldn't have kept the FULL input (the outline should be gone).
    assert out_dark < inp_dark


def test_extract_decoration_no_silhouette_returns_input():
    """An image with only a few small marks (no big closed contour) is
    treated as a decoration-only crop and returned unchanged."""
    img = _white(200, 200)
    d = ImageDraw.Draw(img)
    # Just a few short horizontal lines, no big closed shape.
    for y in [40, 60, 80]:
        d.line([20, y, 60, y], fill='black', width=2)
    out = extract_decoration_region(img)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(img))
```

- [ ] **Step 2: Run tests, verify the new ones fail**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/test_preprocess.py -v
```

Expected: 6 passed (prior) + 3 errors (`ImportError: cannot import name 'extract_decoration_region'`).

- [ ] **Step 3: Implement `extract_decoration_region`**

Append to `preprocess.py`:

```python
def extract_decoration_region(pil_img, white_thresh=240, erode_px_ratio=0.03):
    """Mask out the vessel silhouette outline, keep only interior decoration.

    For pottery line drawings the silhouette is the largest closed contour
    of dark pixels. Filling it gives the silhouette region; eroding by a
    few percent of the smaller image dimension removes the outline itself
    plus a thin margin. The eroded interior is the decoration zone — the
    rest is replaced with white so DINOv2 sees only decoration content.

    Falls through to the original image when:
    - there are no dark pixels at all,
    - the largest contour is too small to be a vessel (the input is likely
      already a decoration-only crop),
    - or the silhouette has no inner area after erosion (profile-only
      drawing — `image_type` will drop these later).
    """
    arr_gray = np.asarray(pil_img.convert('L'))
    arr_rgb = np.asarray(pil_img.convert('RGB'))
    h, w = arr_gray.shape

    mask = (arr_gray < white_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.05 * h * w:
        return pil_img

    interior = np.zeros_like(mask)
    cv2.drawContours(interior, [largest], -1, 255, thickness=cv2.FILLED)

    erode_px = max(1, int(erode_px_ratio * min(h, w)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
    interior_eroded = cv2.erode(interior, kernel)

    if (interior_eroded > 0).sum() < 0.01 * h * w:
        return pil_img

    out = arr_rgb.copy()
    out[interior_eroded == 0] = 255
    return Image.fromarray(out)
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/test_preprocess.py -v
```

Expected: 9 passed.

---

## Task 5: TDD `valid_patch_mask`

**Files:**
- Modify: `preprocess.py`
- Modify: `tests/test_preprocess.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_preprocess.py`:

```python
from preprocess import valid_patch_mask


def test_valid_patch_mask_white_returns_all_false():
    img = _white(224, 224)
    m = valid_patch_mask(img)
    assert m.shape == (256,)
    assert m.dtype == np.bool_
    assert not m.any()


def test_valid_patch_mask_full_black_returns_all_true():
    img = Image.new('RGB', (224, 224), color='black')
    m = valid_patch_mask(img)
    assert m.all()


def test_valid_patch_mask_corner_dot_marks_only_that_corner():
    img = _white(224, 224)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 28, 28], fill='black')  # top-left ~2 patches
    m = valid_patch_mask(img).reshape(16, 16)
    assert m[0, 0]  # top-left patch valid
    assert not m[15, 15]  # bottom-right invalid
```

- [ ] **Step 2: Run tests, verify failures**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/test_preprocess.py -v
```

Expected: 9 passed (prior) + 3 errors.

- [ ] **Step 3: Implement `valid_patch_mask`**

Append to `preprocess.py`:

```python
def valid_patch_mask(pil_img, patch_grid=16, white_thresh=240,
                     content_frac_min=0.05):
    """Return a (patch_grid**2,) bool array marking which 14x14-px patches
    of a 224x224 DINOv2 input region have at least `content_frac_min`
    dark pixels.

    The image is resized to (patch_grid*14, patch_grid*14) = 224x224 first
    (matching DINOv2-small's input geometry) so the mask aligns 1:1 with
    the patch tokens. Patches that are essentially blank background are
    excluded from rerank computations to avoid white-vs-white spurious
    matches.
    """
    side = patch_grid * 14
    img = pil_img.convert('L').resize((side, side))
    arr = np.asarray(img)
    # Reshape into (patch_grid, 14, patch_grid, 14) and average dark-frac
    # per cell.
    cells = arr.reshape(patch_grid, 14, patch_grid, 14)
    dark_frac = (cells < white_thresh).mean(axis=(1, 3))
    return (dark_frac >= content_frac_min).reshape(-1)
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/test_preprocess.py -v
```

Expected: 12 passed.

---

## Task 6: TDD `preprocess_for_dinov2` wrapper + commit Task 1-6

**Files:**
- Modify: `preprocess.py`
- Modify: `tests/test_preprocess.py`

- [ ] **Step 1: Append failing test**

Append to `tests/test_preprocess.py`:

```python
from preprocess import preprocess_for_dinov2


def test_preprocess_for_dinov2_returns_pil_and_mask():
    img = _circle_with_inner_lines(200, 200, radius=80)
    out_img, mask = preprocess_for_dinov2(img)
    assert isinstance(out_img, Image.Image)
    assert mask.shape == (256,)
    assert mask.dtype == np.bool_


def test_preprocess_for_dinov2_white_image_yields_empty_mask():
    out_img, mask = preprocess_for_dinov2(_white())
    assert not mask.any()
```

- [ ] **Step 2: Run, verify failures**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/test_preprocess.py -v
```

Expected: 12 passed (prior) + 2 errors.

- [ ] **Step 3: Implement wrapper**

Append to `preprocess.py`:

```python
def preprocess_for_dinov2(pil_img):
    """Full preprocessing: bbox crop → silhouette mask → re-bbox-crop the
    masked output (so DINOv2 doesn't waste resolution on white margins) →
    valid_patch_mask on the final image.

    Returns (final_pil, valid_patch_mask).
    """
    cropped = bbox_crop(pil_img)
    masked = extract_decoration_region(cropped)
    final = bbox_crop(masked)
    return final, valid_patch_mask(final)
```

- [ ] **Step 4: Run, verify all pass**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/test_preprocess.py -v
```

Expected: 14 passed.

- [ ] **Step 5: Commit preprocess module + tests**

```bash
git add preprocess.py tests/__init__.py tests/test_preprocess.py requirements.txt
git commit -m "feat: Add preprocess module for decoration-only DINOv2 pipeline

Pure functions, shared by corpus build and query time:
- bbox_crop: crop to non-white content bbox with proportional padding
- ink_density: dark-pixel fraction inside that bbox
- extract_decoration_region: detect vessel silhouette outline as the
  largest closed contour, fill it, erode 3% inward, mask everything
  outside the eroded interior to white. Decoration-only crops (no big
  contour) and profile-only drawings (no interior after erosion) fall
  through unchanged.
- valid_patch_mask: 16x16 bool grid marking patches with >5% dark pixels
- preprocess_for_dinov2: bbox -> mask -> re-bbox + return mask

Pytest introduced for this module; no impact on the embedded SPA."
```

---

## Task 7: DB migration — add `image_type` columns

**Files:**
- Modify: `ceramica.db` (schema + data; tracked in git)

- [ ] **Step 1: Inspect current schema**

```bash
sqlite3 ceramica.db ".schema items" | head -25
```

Expected: shows the `items` table without `image_type` column.

- [ ] **Step 2: Apply migration**

```bash
sqlite3 ceramica.db "ALTER TABLE items ADD COLUMN image_type TEXT; ALTER TABLE items ADD COLUMN image_type_confidence REAL;"
```

Expected: no output (success).

- [ ] **Step 3: Verify columns exist**

```bash
sqlite3 ceramica.db "PRAGMA table_info(items);" | grep -E "image_type|image_type_confidence"
```

Expected: 2 lines, one for each new column.

- [ ] **Step 4: Sanity check that the viewer still loads**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -c "
import sqlite3
c = sqlite3.connect('ceramica.db')
n = c.execute('SELECT COUNT(*) FROM items').fetchone()[0]
print(f'items: {n}')
c.close()
"
```

Expected: prints item count (typically 1346).

- [ ] **Step 5: Do NOT commit yet** — Task 9 commits the populated DB.

---

## Task 8: Write `classify_image_types.py`

**Files:**
- Create: `classify_image_types.py`

- [ ] **Step 1: Create the script**

Create `classify_image_types.py`:

```python
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
            sims = (img_emb @ text_emb.T).squeeze(0)            # (5,)
            probs = torch.softmax(sims * 100.0, dim=-1)         # CLIP convention: scale by ~100
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
```

- [ ] **Step 2: Syntax check**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m py_compile classify_image_types.py && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 3: Do NOT run yet** (Task 9 runs it).

---

## Task 9: Run classifier, populate DB, sanity check, commit

**Files:**
- Modify: `ceramica.db` (data only)
- Add to commit: `classify_image_types.py`, `ceramica.db`

- [ ] **Step 1: Run the classifier**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 classify_image_types.py
```

Expected: progress lines every 100 items, then a distribution table. Total wall time ~5-10 minutes on CPU. Distribution should be roughly:
- `decorated_vessel`: majority (~60-75% of items)
- `profile_section`: 50-150 items (mostly Smith_*)
- `decoration_only`: small (<50)
- `plain_vessel`: small-medium
- `unclassified`: small

- [ ] **Step 2: Spot-check problematic items from the screenshot**

```bash
sqlite3 ceramica.db "SELECT id, image_type, image_type_confidence FROM items WHERE id IN ('Smith_177','Smith_156','Smith_286','Smith_271','Smith_292');"
```

Expected: most/all should be classified `profile_section` or low-confidence `unclassified`. If any of these is `decorated_vessel` with high confidence, the classifier is too lenient — see Step 4.

- [ ] **Step 3: Spot-check a known decorated item**

```bash
sqlite3 ceramica.db "SELECT id, image_type, image_type_confidence FROM items WHERE id LIKE 'Pl_XXVI_01' OR id LIKE 'Degli_Espositi_57' OR id LIKE 'Pellegrino_%' LIMIT 5;"
```

Expected: classified `decorated_vessel` with confidence > 0.4.

- [ ] **Step 4: If results look wrong**

If too many decorated items are misclassified as `profile_section`, the prompts are too aggressive. Edit `PROMPTS` in `classify_image_types.py` to make the `decorated_vessel` prompt more inclusive (e.g., add "or fragment with painted bands"), rerun. Document the rationale in the commit message.

- [ ] **Step 5: Commit DB + classifier script**

```bash
git add classify_image_types.py ceramica.db
git commit -m "feat: Add image_type column to items via CLIP zero-shot classifier

ALTER TABLE items ADD image_type TEXT, image_type_confidence REAL.

classify_image_types.py runs CLIP ViT-B/32 against 5 archaeological
prompts (decorated_vessel / profile_section / decoration_only /
plain_vessel / unclassified) and stores the argmax label + softmax
confidence per row.

Distribution (paste actual numbers from Step 1's output)."
```

(Edit the message to include the actual distribution numbers from Step 1.)

---

## Task 10: Rewrite `compute_embeddings.py` for v3 storage

**Files:**
- Modify: `compute_embeddings.py`

- [ ] **Step 1: Read current compute_embeddings.py to understand the structure**

```bash
wc -l compute_embeddings.py
```

Expected: ~210 lines.

- [ ] **Step 2: Rewrite the script**

Overwrite `compute_embeddings.py` with:

```python
#!/usr/bin/env python3
"""Compute decoration-focused DINOv2 embeddings (v3).

For every items row whose image_type belongs to the kept set, run the
shared preprocess_for_dinov2 pipeline (bbox crop → silhouette mask →
valid_patch_mask), forward DINOv2-small with output_hidden_states, and
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
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

from preprocess import preprocess_for_dinov2, ink_density

# Configuration
DB_PATH = "ceramica.db"
OUTPUT_DIR = "ml_model"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "image_embeddings_v3.npz")
METADATA_FILE   = os.path.join(OUTPUT_DIR, "embeddings_metadata_v3.json")
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
```

- [ ] **Step 3: Syntax check**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m py_compile compute_embeddings.py && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 4: Do NOT run yet** (Task 11 runs it).

---

## Task 11: Run new compute_embeddings, generate v3 artefacts, commit

**Files:**
- Generated: `ml_model/image_embeddings_v3.npz`, `ml_model/embeddings_metadata_v3.json`

- [ ] **Step 1: Run**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 compute_embeddings.py
```

Expected: takes ~5-10 minutes, finishes with a count summary. Likely 600-750 valid items (less than the 771 of v2 because image_type filtering is stricter).

- [ ] **Step 2: Verify file sizes**

```bash
ls -la ml_model/image_embeddings_v3.npz ml_model/embeddings_metadata_v3.json
```

Expected: npz around 100-200 MB; json around 200-500 KB.

- [ ] **Step 3: Spot-check the npz can be loaded**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -c "
import numpy as np
d = np.load('ml_model/image_embeddings_v3.npz')
print('keys:', list(d.keys()))
print('cls shape:', d['cls_embeddings'].shape, d['cls_embeddings'].dtype)
print('mean shape:', d['mean_embeddings'].shape, d['mean_embeddings'].dtype)
print('patch shape:', d['patch_embeddings'].shape, d['patch_embeddings'].dtype)
print('valid shape:', d['valid_patch_masks'].shape, d['valid_patch_masks'].dtype)
print('valid frac:', d['valid_patch_masks'].mean())
"
```

Expected: `keys: ['cls_embeddings', 'mean_embeddings', 'patch_embeddings', 'valid_patch_masks']`; shapes consistent (N, 384), (N, 384), (N, 256, 384), (N, 256). Valid fraction should be 0.20-0.45 (decoration covers a fraction of the image).

- [ ] **Step 4: Commit**

```bash
git add compute_embeddings.py ml_model/image_embeddings_v3.npz ml_model/embeddings_metadata_v3.json
git commit -m "feat: Compute v3 DINOv2 embeddings (cls + mean-patch + per-patch + valid masks)

Per-item pipeline: bbox crop -> silhouette mask via preprocess.py
-> DINOv2 forward with output_hidden_states -> store

- cls_embeddings  (N, 384)        fp32 normalized (fallback)
- mean_embeddings (N, 384)        fp32 normalized (primary coarse rank)
- patch_embeddings (N, 256, 384)  fp16 raw (rerank normalizes)
- valid_patch_masks (N, 256)      bool

Filters: image_type IN (decorated_vessel, decoration_only) OR
(unclassified AND confidence<0.4); ink_density>=0.05 backstop.

v2 image_embeddings.npz left untouched for instant rollback."
```

---

## Task 12: viewer_app.py — load v3 embeddings with v2 fallback

**Files:**
- Modify: `viewer_app.py`

- [ ] **Step 1: Locate the embeddings loader**

```bash
grep -n "load_embeddings\b\|EMBEDDINGS_METADATA\|EMBEDDINGS_ARRAY\|image_embeddings\.npz" viewer_app.py | head -20
```

Note the exact line ranges for `load_embeddings`, the module-level globals it sets, and where it's called.

- [ ] **Step 2: Replace `load_embeddings` with a v3-first version**

Find the existing `def load_embeddings():` function (around line 700-770) and the module-level globals near line 90-100.

Replace the globals block with:

```python
# ===== Embeddings (v3 = decoration-focused; v2 = legacy CLS-only) =====
EMBEDDINGS_VERSION = 0          # 0 = not loaded, 2 = v2, 3 = v3
EMBEDDINGS_METADATA = None
EMBEDDINGS_CLS = None           # (N, 384) fp32 normalized — v2 OR v3
EMBEDDINGS_MEAN = None          # (N, 384) fp32 normalized — v3 only
EMBEDDINGS_PATCH = None         # (N, 256, 384) fp16 — v3 only
EMBEDDINGS_VALID = None         # (N, 256) bool — v3 only
EMBEDDINGS_TYPE_KEEP = None     # (N,) bool: True if image_type passes filter — v3 only

DINOV2_PROCESSOR = None
DINOV2_MODEL = None
DINOV2_MODEL_ID = "facebook/dinov2-small"
```

Replace the body of `load_embeddings()` with:

```python
def load_embeddings():
    """Load v3 if available, else v2. Returns True on success.

    v3 layout: cls + mean + patch + valid_patch_masks. Search uses mean
    coarse rank + patch rerank.

    v2 fallback: cls only. Search uses cls cosine. No image_type filter.
    """
    global EMBEDDINGS_VERSION, EMBEDDINGS_METADATA, EMBEDDINGS_CLS, EMBEDDINGS_MEAN
    global EMBEDDINGS_PATCH, EMBEDDINGS_VALID, EMBEDDINGS_TYPE_KEEP, ML_DISABLED

    if EMBEDDINGS_VERSION:
        return True
    if ML_DISABLED:
        return False

    import numpy as np
    base = Path(__file__).parent / "ml_model"
    v3_npz = base / "image_embeddings_v3.npz"
    v3_meta = base / "embeddings_metadata_v3.json"
    v2_npz = base / "image_embeddings.npz"
    v2_meta = base / "embeddings_metadata.json"

    if v3_npz.exists() and v3_meta.exists():
        try:
            with open(v3_meta) as f:
                EMBEDDINGS_METADATA = json.load(f)
            d = np.load(v3_npz)
            EMBEDDINGS_CLS   = d["cls_embeddings"].astype(np.float32)
            EMBEDDINGS_MEAN  = d["mean_embeddings"].astype(np.float32)
            EMBEDDINGS_PATCH = d["patch_embeddings"]    # fp16
            EMBEDDINGS_VALID = d["valid_patch_masks"].astype(bool)
            # Build the image_type filter once at load time.
            kept = []
            for it in EMBEDDINGS_METADATA["items"]:
                t = it.get("image_type", "")
                c = float(it.get("image_type_confidence", 0.0))
                kept.append(t in {"decorated_vessel", "decoration_only"}
                            or (t == "unclassified" and c < 0.4))
            EMBEDDINGS_TYPE_KEEP = np.array(kept, dtype=bool)
            EMBEDDINGS_VERSION = 3
            print(f"   Loaded v3 embeddings: {len(EMBEDDINGS_METADATA['items'])} items "
                  f"({int(EMBEDDINGS_TYPE_KEEP.sum())} pass image_type filter)")
            return True
        except Exception as e:
            print(f"   v3 load failed: {e!r} — falling back to v2")

    if v2_npz.exists() and v2_meta.exists():
        try:
            with open(v2_meta) as f:
                EMBEDDINGS_METADATA = json.load(f)
            d = np.load(v2_npz)
            EMBEDDINGS_CLS = d["embeddings"].astype(np.float32)
            EMBEDDINGS_VERSION = 2
            print(f"   Loaded v2 embeddings: {len(EMBEDDINGS_METADATA['items'])} items "
                  f"(legacy CLS-only path)")
            return True
        except Exception as e:
            print(f"   v2 load failed: {e!r}")

    ML_DISABLED = True
    return False
```

- [ ] **Step 3: Remove dead `_content_bbox_crop` from viewer_app.py**

Search for `def _content_bbox_crop` and replace its body with a thin wrapper:

```python
def _content_bbox_crop(img):
    """Backwards-compat wrapper; new code uses preprocess.bbox_crop directly."""
    from preprocess import bbox_crop
    return bbox_crop(img)
```

- [ ] **Step 4: Add `from preprocess import ...` at the top of viewer_app.py imports**

Find the existing import block (around line 1-30) and add:

```python
from preprocess import bbox_crop, preprocess_for_dinov2
```

- [ ] **Step 5: Syntax check**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m py_compile viewer_app.py 2>&1 | grep -v SyntaxWarning && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 6: Smoke test load**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -c "
import viewer_app as v
ok = v.load_embeddings()
print('loaded:', ok, 'version:', v.EMBEDDINGS_VERSION)
print('cls shape:', None if v.EMBEDDINGS_CLS is None else v.EMBEDDINGS_CLS.shape)
print('mean shape:', None if v.EMBEDDINGS_MEAN is None else v.EMBEDDINGS_MEAN.shape)
print('keep frac:', None if v.EMBEDDINGS_TYPE_KEEP is None else float(v.EMBEDDINGS_TYPE_KEEP.mean()))
"
```

Expected: `loaded: True version: 3 cls shape: (N, 384) mean shape: (N, 384) keep frac: ~0.7-0.9`.

- [ ] **Step 7: Do NOT commit yet** — Task 14 collects all viewer changes.

---

## Task 13: viewer_app.py — `find_similar_images` v3 path

**Files:**
- Modify: `viewer_app.py`

- [ ] **Step 1: Locate `find_similar_images`**

```bash
grep -n "def find_similar_images" viewer_app.py
```

- [ ] **Step 2: Replace its body with the v3-aware implementation**

Replace the entire `def find_similar_images(image_data, top_k=20, threshold=0.5):` function (around line 790-880) with:

```python
def find_similar_images(image_data, top_k=20, threshold=0.5):
    """Decoration-focused similarity search.

    v3 path:
      1. Preprocess query (bbox+silhouette mask) → DINOv2 → mean_q,
         patches_q, valid_q.
      2. Filter corpus by image_type (precomputed EMBEDDINGS_TYPE_KEEP).
      3. Coarse rank by cosine(mean_q, mean_embeddings) → top-50.
      4. Patch-level Chamfer rerank on those 50:
           for each valid query patch q_i, max cosine with valid candidate
           patches; mean of top-min(64, |valid_q|) max-sims.
      5. Return top-k after rerank.

    v2 path (fallback): cosine on cls embeddings only, no rerank, no
    image_type filter.
    """
    global ML_DISABLED, DINOV2_MODEL, DINOV2_PROCESSOR
    if ML_DISABLED:
        return {"error": "ML features are disabled", "similar_items": []}
    if not load_embeddings():
        return {"error": "Embeddings not available", "similar_items": []}
    if not load_dinov2():
        return {"error": "DINOv2 not available", "similar_items": []}

    import torch
    import numpy as np

    # Decode query
    if "," in image_data:
        image_data = image_data.split(",")[1]
    q_bytes = base64.b64decode(image_data)
    raw = Image.open(io.BytesIO(q_bytes)).convert("RGB")

    final_img, valid_q = preprocess_for_dinov2(raw)

    inputs = DINOV2_PROCESSOR(images=final_img, return_tensors="pt")
    with torch.no_grad():
        out = DINOV2_MODEL(**inputs)
        hidden = out.last_hidden_state[0]                  # (257, 384)
        cls_q = hidden[0]
        patches_q_t = hidden[1:]                           # (256, 384)
        mean_q_t = patches_q_t.mean(dim=0)
    cls_q = (cls_q / (cls_q.norm() + 1e-8)).cpu().numpy().astype(np.float32)
    mean_q = (mean_q_t / (mean_q_t.norm() + 1e-8)).cpu().numpy().astype(np.float32)
    # Patch tensors stay un-normalized; rerank normalizes per-patch.
    patches_q = patches_q_t.cpu().numpy().astype(np.float32)

    items = EMBEDDINGS_METADATA["items"]

    if EMBEDDINGS_VERSION == 2:
        sims = EMBEDDINGS_CLS @ cls_q                      # (N,)
        order = np.argsort(-sims)[:top_k]
        results = []
        for idx in order:
            s = float(sims[idx])
            if s < threshold:
                continue
            it = dict(items[idx])
            it["similarity"] = round(s * 100.0, 1)
            results.append(it)
        return {"similar_items": results, "total_corpus": len(items),
                "version": 2}

    # ---- v3 path ----
    # 1) Coarse: cosine on mean embeddings, restricted to image_type-kept items.
    keep = EMBEDDINGS_TYPE_KEEP
    coarse = (EMBEDDINGS_MEAN @ mean_q)                    # (N,)
    coarse_masked = np.where(keep, coarse, -np.inf)
    top50_idx = np.argpartition(-coarse_masked, min(50, keep.sum()))[:min(50, int(keep.sum()))]
    top50_idx = top50_idx[np.argsort(-coarse_masked[top50_idx])]

    # 2) Rerank: patch-level Chamfer mean.
    if not valid_q.any():
        # Empty/all-white query — fall back to coarse only.
        rerank_scores = coarse[top50_idx]
    else:
        # Normalize valid query patches once.
        q_valid_idx = np.where(valid_q)[0]
        q_valid = patches_q[q_valid_idx]                       # (Q, 384)
        q_valid_norm = q_valid / (np.linalg.norm(q_valid, axis=1, keepdims=True) + 1e-8)
        K = min(64, len(q_valid))
        rerank_scores = np.empty(len(top50_idx), dtype=np.float32)
        for ri, idx in enumerate(top50_idx):
            c_patches = EMBEDDINGS_PATCH[idx].astype(np.float32)   # (256, 384)
            c_valid_idx = np.where(EMBEDDINGS_VALID[idx])[0]
            if len(c_valid_idx) == 0:
                rerank_scores[ri] = coarse[idx]                # degenerate candidate
                continue
            c_valid = c_patches[c_valid_idx]
            c_valid_norm = c_valid / (np.linalg.norm(c_valid, axis=1, keepdims=True) + 1e-8)
            sim_mat = q_valid_norm @ c_valid_norm.T            # (Q, C)
            best_per_q = sim_mat.max(axis=1)                   # (Q,)
            top = np.partition(-best_per_q, K - 1)[:K]
            rerank_scores[ri] = float(-top.mean())             # negate twice = mean of top-K positive sims

    # 3) Sort top-50 by rerank score, take top-k.
    order = np.argsort(-rerank_scores)
    final = []
    for ri in order[:top_k]:
        idx = int(top50_idx[ri])
        rerank_s = float(rerank_scores[ri])
        coarse_s = float(coarse[idx])
        if rerank_s < threshold:
            continue
        it = dict(items[idx])
        it["similarity"] = round(rerank_s * 100.0, 1)
        it["coarse_similarity"] = round(coarse_s * 100.0, 1)
        it["image_type"] = items[idx].get("image_type", "")
        final.append(it)

    # Below-threshold fallback (kept from v2 behavior).
    below_threshold_fallback = False
    if not final and len(top50_idx) > 0:
        below_threshold_fallback = True
        for ri in order[:5]:
            idx = int(top50_idx[ri])
            it = dict(items[idx])
            it["similarity"] = round(float(rerank_scores[ri]) * 100.0, 1)
            it["coarse_similarity"] = round(float(coarse[idx]) * 100.0, 1)
            it["image_type"] = items[idx].get("image_type", "")
            it["below_threshold"] = True
            final.append(it)

    return {
        "similar_items": final,
        "total_corpus": int(keep.sum()),
        "below_threshold_fallback": below_threshold_fallback,
        "max_similarity": round(float(rerank_scores.max() * 100.0), 1) if len(rerank_scores) else 0.0,
        "threshold_pct": round(threshold * 100.0, 1),
        "version": 3,
    }
```

- [ ] **Step 3: Syntax check**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m py_compile viewer_app.py 2>&1 | grep -v SyntaxWarning && echo "syntax OK"
```

- [ ] **Step 4: Smoke test against the previously known query**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -c "
import base64, viewer_app as v
v.load_embeddings(); v.load_dinov2()
with open('Pellegrino/Planches/Pl_XXVI_01.png','rb') as f:
    q = 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
r = v.find_similar_images(q, top_k=5, threshold=0.4)
print('version:', r.get('version'))
print('total_corpus:', r.get('total_corpus'))
for it in r['similar_items']:
    print(f\"  {it['id']:30s} sim={it['similarity']}%  coarse={it.get('coarse_similarity','-')}%  type={it.get('image_type','-')}\")
"
```

Expected: `version: 3`, top-5 items printed, the same image (`Pl_XXVI_01`) is rank 1 with similarity ≥ 95% (mean-of-patches isn't exactly 100% even for an identical image because the bbox crop and silhouette mask can produce slightly different inputs vs the stored one — but it must be the highest by far).

If the same-image self-match is below 90%, abort and inspect: most likely the silhouette mask is being applied differently to the stored vs query path. Don't proceed until this passes.

- [ ] **Step 5: Do NOT commit yet** — Task 14 commits viewer changes together.

---

## Task 14: Frontend score label + v3-aware UI hints + commit

**Files:**
- Modify: `viewer_app.py` (HTML inside `get_viewer_html`)

- [ ] **Step 1: Find the match-card markup**

```bash
grep -n "match-confidence\|similar.*span style\|Visually Similar Ceramics" viewer_app.py | head -10
```

- [ ] **Step 2: Update the card score label**

Find the line that renders `${item.similarity}% similar` (around line 6574) inside `displaySimilarMatches`. Replace this:

```javascript
<div class="match-confidence">${{item.similarity}}% similar${{item.below_threshold ? ' <span style=&quot;color:#b8860b&quot;>(sotto soglia)</span>' : ''}}</div>
```

with:

```javascript
<div class="match-confidence" title="Coarse global similarity (whole-image, shape-aware): ${{item.coarse_similarity || '-'}}%">${{item.similarity}}% similar (decoration)${{item.below_threshold ? ' <span style=&quot;color:#b8860b&quot;>(sotto soglia)</span>' : ''}}</div>
```

- [ ] **Step 3: Update the section subtitle**

Search for `Ranked by visual similarity based on decoration patterns` and replace with:

```html
Ranked by decoration similarity (vessel shape ignored)
```

- [ ] **Step 4: Syntax check**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m py_compile viewer_app.py 2>&1 | grep -v SyntaxWarning && echo "syntax OK"
```

- [ ] **Step 5: Local smoke test — run viewer, hit /api/ml/similar**

In one terminal:

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 viewer_app.py
```

In another terminal, send a query:

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -c "
import base64, json, urllib.request
with open('Pellegrino/Planches/Pl_XXVI_01.png','rb') as f:
    q = 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
data = json.dumps({'image': q, 'top_k': 5, 'threshold': 0.4}).encode()
r = urllib.request.urlopen(urllib.request.Request(
    'http://localhost:8080/api/ml/similar', data=data,
    headers={'Content-Type':'application/json'}, method='POST'))
print(r.read().decode()[:600])
"
```

Expected: JSON with `version: 3`, top-5 results including coarse_similarity per item.

Stop the server (Ctrl-C in the first terminal).

- [ ] **Step 6: Commit viewer changes**

```bash
git add viewer_app.py
git commit -m "feat: Decoration-only similarity in viewer (v3 ranking + image_type filter + patch rerank)

- load_embeddings() now reads ml_model/image_embeddings_v3.npz with
  fallback to v2 (legacy CLS-only path). Builds an image_type-keep
  bitmask once at load time.
- find_similar_images() v3 path: preprocess query through shared
  preprocess.preprocess_for_dinov2 -> DINOv2 -> mean+patch+valid ->
  coarse cosine on mean_embeddings restricted to keep-mask -> top-50
  -> patch-level Chamfer rerank (mean of top-K best per-query-patch
  cosines vs valid candidate patches) -> top-K returned.
- Match cards now show 'X% similar (decoration)' with the global
  coarse score in a tooltip; subtitle copy updated.
- _content_bbox_crop becomes a thin wrapper around preprocess.bbox_crop
  (single source of truth)."
```

---

## Task 15: End-to-end smoke test before deploy

- [ ] **Step 1: Start the local server**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 viewer_app.py
```

Expected log output includes `Loaded v3 embeddings: N items (M pass image_type filter)` with M < N.

- [ ] **Step 2: Open the viewer in a browser**

Visit `http://localhost:8080`. Log in (existing flow).

- [ ] **Step 3: Run the canonical query Marta uses**

Use the same upload Marta used in the screenshot in `/Volumes/Extreme Pro/Dropbox/Screenshot/Screenshot 2026-05-06 alle 13.49.57.png` (or any whole-vessel decorated query).

Verify:
- Top results contain decorated vessels, not Smith profile sections
- Each card shows "X% similar (decoration)" with a tooltip on hover
- Heatmap toggle still works and overlays on the match images
- Page subtitle reads "Ranked by decoration similarity (vessel shape ignored)"

- [ ] **Step 4: Edge case: pure decoration crop**

Crop a small portion of any decorated vessel (e.g., a band of chevrons via Preview/macOS), upload as query. Verify top results have similar-looking patterns regardless of vessel shape.

- [ ] **Step 5: Edge case: empty/almost-blank upload**

Upload a mostly-white image. Verify graceful handling (either an empty result with the below-threshold banner, or a clean error message; no crash).

- [ ] **Step 6: Stop the server, push the chain to Railway**

```bash
git push
```

Wait for deploy (use `railway deployment list`). Verify `Loaded v3 embeddings: ...` appears in `railway logs --lines 200`.

- [ ] **Step 7: Tell Marta**

Tell the user the deploy is live, and what to test:
- Hard reload not strictly needed anymore (cache-control no-store from earlier commit), but encourage one for the first try.
- New score labels and tooltip explained.
- The image_type filter dropped some items: the `total_corpus` field now shows the post-filter count.

---

## Self-review

**Spec coverage check:**

| Spec section                              | Implementation task           |
|-------------------------------------------|-------------------------------|
| `preprocess.py` (shared)                  | Tasks 2-6                     |
| Silhouette detection algorithm            | Task 4                        |
| `classify_image_types.py` + 5 categories  | Task 8                        |
| DB migration `image_type` columns         | Task 7                        |
| Run classifier + populate DB              | Task 9                        |
| `compute_embeddings.py` rewritten with cls/mean/patch storage | Task 10-11 |
| `embeddings_v3.npz` layout                | Task 10 (storage step)        |
| `viewer_app.py` `load_embeddings_v3` w/ v2 fallback | Task 12              |
| Filter rule kept-set                      | Task 12 (Step 2 keep-mask)    |
| `find_similar_images_v3` coarse + patch rerank | Task 13                  |
| Frontend score relabel + tooltip          | Task 14                       |
| 4-commit migration plan                   | Tasks 6, 9, 11, 14 (4 commits)|
| Manual test plan items 1-5 from spec      | Task 15 steps 3-5             |
| Backward compat: v3 missing → v2 fallback | Task 12                       |

All spec sections map to a task. ✓

**Placeholder scan**: searched for "TBD", "TODO", "implement later", "fill in details" — none found. Each step has actual code or actual commands. ✓

**Type / name consistency**: `EMBEDDINGS_VERSION`, `EMBEDDINGS_CLS`, `EMBEDDINGS_MEAN`, `EMBEDDINGS_PATCH`, `EMBEDDINGS_VALID`, `EMBEDDINGS_TYPE_KEEP` are used identically in Tasks 12 and 13. `image_type_confidence` matches between DB schema (Task 7), classifier (Task 8), compute_embeddings filter (Task 10), and viewer load (Task 12). ✓

**Note on tests**: pytest is introduced for `preprocess.py` only because that file is pure functions that benefit from unit tests. The rest of the codebase has no test infrastructure, and the established protocol (per spec) is manual smoke tests — followed in Tasks 9 (classifier sanity), 11 (npz load), 13 (single-shot Python), 14 (HTTP request via running server), 15 (browser).
