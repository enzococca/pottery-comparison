# Decoration-Only Similarity — Design

**Date**: 2026-05-06
**Status**: design approved, awaiting implementation plan
**Author**: enzo + Claude

## Problem

The current similarity search ranks candidates by global DINOv2 CLS-token cosine
similarity. The CLS embedding mixes vessel **shape** with **decoration** —
two pots with the same outline but different decoration score very high; two
fragments with the same decoration but different shapes score low. Marta's
research workflow needs the opposite: she wants to find pottery whose
**decoration patterns** look like her query, regardless of vessel shape.

A second pain point: the corpus is dominated by Schmidt_Bat (51.5% after the
recent profile-only filter), and several Schmidt items are still
profile-section drawings whose `decoration='painted'` field refers to the
original vessel rather than what the image shows. The 5%-ink-density filter
catches the worst cases but lets through items with text/scale-bar artifacts.

Marta's queries are a mix of (a) whole-vessel sketches/photos and (b)
decoration-only crops, so the system must handle both modes uniformly.

## Goals

1. **Shape-invariant ranking**: cosine similarity should respond to decoration
   patterns, not vessel silhouette.
2. **Cleaner corpus**: filter out profile-only / text-only / unrelated images
   via per-image categorical labels stored in the DB.
3. **Two-stage ranking**: fast coarse pass over the whole corpus, slow precise
   rerank over the top-50.
4. **Backward-compatible deploy**: keep v2 embeddings on disk during rollout
   so a regression can be rolled back by flipping a feature flag.

## Non-goals

- Fine-tuning DINOv2 or training new models. Zero-shot only.
- Manual decoration motif annotation (chevron / spiral / reticolo / etc.).
  Could come later as a re-rank input but is out of scope here.
- Photo-quality preprocessing (deshadowing, orientation, etc.). The corpus
  is already mostly line drawings and the few photos work well enough.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│            shared preprocessing pipeline (preprocess.py)    │
│  bbox_crop → silhouette_detect → decoration_mask           │
│            → ink_density → valid_patch_mask                │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │   DINOv2 forward     │   (on masked-interior image)
                │   output_hidden_states│
                └──────────────────────┘
                            │
        ┌───────────────────┼────────────────────────┐
        ▼                   ▼                        ▼
   cls_embedding     mean_patch_embedding       patch_embeddings
   (384, fp32)       (384, fp32)                (256 × 384, fp16)
        │                   │                        │
        └────────┬──────────┴──────────┬─────────────┘
                 ▼                     ▼
          ┌──────────────────────────────────┐
          │  ml_model/embeddings_v3.npz      │ (~145 MB packed)
          │  ml_model/embeddings_metadata    │
          │  _v3.json                        │
          └──────────────────────────────────┘

DB: items.image_type ← populated once via CLIP zero-shot
    items.image_type_confidence
```

### Query pipeline

```
1. Preprocess query (same pipeline as corpus)
   → cls_q, mean_q, patches_q (256×384), valid_mask_q (256 bool)

2. Filter corpus by image_type:
   IN ('decorated_vessel','decoration_only')
   OR (image_type='unclassified' AND confidence < 0.4)

3. Coarse rank (A): cosine(mean_q, all.mean_embeddings) → top-50 indices

4. Fine rerank (B) on those 50, for each candidate:
   • for each valid patch q_i in query:
       s_i = max(cosine(q_i, c_j) for c_j valid in candidate)
   • score = mean( top_K(s_i) )  with K = min(64, |valid_q|)

5. Return top-10 ordered by B score.
```

## Components

### `preprocess.py` (new, shared)

Pure functions, no global state, importable from both `compute_embeddings.py`
and `viewer_app.py`.

- `bbox_crop(pil_img, white_thresh=240, padding_ratio=0.03) -> PIL.Image`
  Existing logic, moved here verbatim.
- `extract_decoration_region(pil_img, white_thresh=240, erode_px_ratio=0.03) -> PIL.Image`
  See "Silhouette detection" below.
- `ink_density(pil_img, white_thresh=240) -> float`
  Existing, moved here.
- `valid_patch_mask(pil_img, patch_grid=16, white_thresh=240, content_frac_min=0.05) -> np.ndarray[bool]`
  Returns 256-bool array marking which patches have at least 5% dark pixels
  in their corresponding 14×14 input region.
- `preprocess_for_dinov2(pil_img) -> (PIL.Image, np.ndarray[bool])`
  Convenience wrapper: bbox_crop → extract_decoration_region → valid_patch_mask
  on the post-mask image.

### Silhouette detection algorithm

```python
def extract_decoration_region(pil_img, white_thresh=240, erode_px_ratio=0.03):
    arr_gray = np.asarray(pil_img.convert('L'))
    arr_rgb  = np.asarray(pil_img.convert('RGB'))
    h, w = arr_gray.shape

    mask = (arr_gray < white_thresh).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img  # all white, nothing to do

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.05 * h * w:
        # No recognizable silhouette → already a decoration crop
        return pil_img

    interior = np.zeros_like(mask)
    cv2.drawContours(interior, [largest], -1, 255, thickness=cv2.FILLED)

    erode_px = max(1, int(erode_px_ratio * min(h, w)))
    kernel   = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
    interior_eroded = cv2.erode(interior, kernel)

    if (interior_eroded > 0).sum() < 0.01 * h * w:
        # Profile-only: nothing inside after erosion. Image_type filter
        # will drop it; for the embedding fall back to the original.
        return pil_img

    out = arr_rgb.copy()
    out[interior_eroded == 0] = 255
    return Image.fromarray(out)
```

### `classify_image_types.py` (new, one-shot)

Uses CLIP `openai/clip-vit-base-patch32` (HuggingFace) to classify every row
of `items` and write back `image_type` + `image_type_confidence`.

Categories (5):

| value              | meaning                                                              |
|--------------------|----------------------------------------------------------------------|
| `decorated_vessel` | whole/fragment vessel showing decoration (line drawing or photo)     |
| `profile_section`  | rim/base section drawing without visible decoration                  |
| `decoration_only`  | crop of just decoration (band, motif, no full vessel outline)        |
| `plain_vessel`     | vessel drawing without decoration                                    |
| `unclassified`     | text-only, scale bar, illegible, anything else                       |

Classifier prompts are anchored to the archaeological domain so CLIP doesn't
drift:

```python
PROMPTS = {
  "decorated_vessel": "a technical drawing of an archaeological pottery vessel with decorative patterns",
  "profile_section":  "a thin profile section drawing of a pottery rim or base, no decoration",
  "decoration_only":  "a close-up of decorative ornament patterns on archaeological ceramics",
  "plain_vessel":     "a plain undecorated pottery vessel drawing",
  "unclassified":     "a scale bar, text label, or unrelated archaeological figure",
}
```

For each image: cosine(image_emb, prompt_embeddings) → softmax → pick argmax.
Store the chosen label and its softmax score.

Filtering rule applied at search time and at corpus build:
- **kept**: `decorated_vessel`, `decoration_only`, `unclassified` with confidence < 0.4
  (low-confidence "unclassified" likely means CLIP couldn't decide — better
   include than risk losing real matches)
- **dropped**: `profile_section`, `plain_vessel`, `unclassified` with confidence ≥ 0.4

### DB migration

```sql
ALTER TABLE items ADD COLUMN image_type TEXT;
ALTER TABLE items ADD COLUMN image_type_confidence REAL;
```

Old viewer code that reads `SELECT *` keeps working: ignores the new columns.

### `compute_embeddings.py` (rewritten)

- Reads items filtered by `decoration` AND by `image_type` ∈ kept set
- For each item: load image → `preprocess_for_dinov2` → DINOv2 forward
  with `output_hidden_states=True`
- Stores **three** arrays + valid_patch_masks in
  `ml_model/embeddings_v3.npz`:
  - `cls_embeddings`     (N, 384)        fp32
  - `mean_embeddings`    (N, 384)        fp32
  - `patch_embeddings`   (N, 256, 384)   fp16
  - `valid_patch_masks`  (N, 256)        bool, packed bits
- Metadata in `embeddings_metadata_v3.json`: same shape as v2 + `image_type` field

### `viewer_app.py` (query path)

- New module-level loader `load_embeddings_v3()` — replaces `load_embeddings()`
  with backward-compat fallback to v2 if v3 missing on disk.
- New `find_similar_images_v3(query_image_data, top_k=10)`:
  1. Preprocess query → mean_q, patches_q, valid_mask_q (and cls_q for fallback)
  2. Filter corpus indices by `image_type` (precomputed once at load time:
     a `np.bool_` array, AND'ed in)
  3. `cosine(mean_q, mean_embeddings[filtered])` → top-50 indices
  4. For those 50, compute the patch-level set similarity (vectorized:
     a single matmul `patches_q @ patches_c.T` per candidate, masked by
     valid masks, then `.max(axis=1).top_k_mean(64)`)
  5. Return top-10 with score and metadata
- `similarity_heatmap` unchanged (already on match image, still uses CLS path
  for the gradient — that's fine because the heatmap explains "global"
  similarity which is still meaningful for visualization).

### Frontend (small updates)

- Score field in card now reads "X% similar (decoration)" with a tooltip
  showing the coarse global score for transparency.
- Subtitle changes: "Ranked by decoration similarity (shape ignored)".
- Below-threshold banner copy updated.

## Data flow

**Corpus build** (once per change):
```
items table → SELECT decorated/non-plain → for each:
  load image → bbox crop → silhouette mask → DINOv2 forward
            → cls/mean/patch embeddings + valid_patch_mask
            → embeddings_v3.npz / metadata_v3.json
```

**Image type classification** (once after each corpus change):
```
items table → for each image:
  CLIP image emb → cosine vs prompt embs → softmax → argmax
  write image_type + image_type_confidence back to DB
```

**Query**:
```
user upload → base64 PNG → backend
  → preprocess → DINOv2 → mean_q, patches_q, valid_mask_q
  → filter by image_type → coarse top-50 by mean_q
  → patch-level rerank on top-50 → top-10 returned
```

## Error handling

- **Silhouette detection no contours found** → return original image (likely
  already a decoration crop). No error.
- **Erosion produces empty interior** → return original (profile-only;
  image_type filter will drop it later).
- **DINOv2 forward fails** (OOM, malformed input) → log + return
  `{'error': str(e)}` (existing behavior).
- **Image_type column missing** at query time → fall back: don't filter
  by it, just rank all items. Logged warning at startup.
- **embeddings_v3.npz missing** → fall back to v2 path (current behavior).
- **CLIP model download fails** at classification time → script aborts cleanly,
  no DB writes.

## Storage and performance

| asset                        | size                          | location          |
|------------------------------|-------------------------------|-------------------|
| `embeddings_v3.npz`          | ~145 MB (patches fp16-packed) | git-tracked       |
| `embeddings_metadata_v3.json`| ~400 KB                       | git-tracked       |
| CLIP model weights           | ~150 MB (downloaded at runtime once)| HF cache    |
| `image_type` column          | ~10 KB total                  | ceramica.db       |

Query-time CPU cost (Railway worker, single core):
- DINOv2 forward on query: ~200 ms
- Coarse cosine over ~700 mean embeddings: <10 ms
- Patch rerank over top-50: ~100 ms (50 matmuls of 256×384 vs 256×384)
- **Total ~300 ms** vs current ~2-3 s for similarity_heatmap-bound work.

## Testing

Manual test plan (run after every commit in this track):

1. **Smoke**: Marta-style upload (vessel sketch with decoration) → top-3
   contains a known decorated match (will pick a canonical query+match
   pair from her existing screenshots).
2. **Regression**: same query as in the screenshot used today (sketch of vessel
   with rim decoration), confirm previously-incorrect Smith profile matches
   no longer appear.
3. **Edge case A**: pure decoration crop (band of chevrons, no vessel
   outline) → top-3 contains items with chevron-like patterns regardless
   of vessel shape.
4. **Edge case B**: empty/all-white upload → graceful fallback, no crash,
   error message in UI.
5. **Backward compat**: rename `embeddings_v3.npz` to disable v3, confirm
   viewer falls back to v2 and still serves results.

No automated test suite exists in this repo; manual verification is the
established protocol.

## Migration plan

Implementation in 4 commits, each independently deployable on Railway:

1. **Commit 1** — `preprocess.py` (pure functions, unit-callable),
   `classify_image_types.py`, DB migration. Backend changes only;
   viewer still uses v2 embeddings.
2. **Commit 2** — Run classifier locally, populate DB, sync `ceramica.db`
   to git. Verify viewer still works (image_type column ignored).
3. **Commit 3** — `compute_embeddings.py` rewritten, generate
   `embeddings_v3.npz` locally, commit. Viewer now reads v3 if present
   and uses image_type filter + coarse rank only (no rerank yet).
4. **Commit 4** — Patch-level rerank in `viewer_app.py`, frontend label
   updates. Final shipping.

If commit 3 reveals quality regressions (e.g., silhouette detection
breaks too many images), I can either fix the heuristic or revert just
that commit while keeping commits 1–2.

## Open questions

None blocking implementation. Two things to revisit after rollout:

- **Weighted mean-of-patches** instead of plain mean: weight each patch
  embedding by its `valid_patch_mask` value. Probably worth it but adds
  one more knob; ship plain mean first.
- **Symmetric Chamfer rerank** (also do C → Q direction and average):
  more rigorous but 2× cost. Asymmetric is simpler and works well in
  the image retrieval literature.
