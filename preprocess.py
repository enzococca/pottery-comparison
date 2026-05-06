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
    unchanged. Zero-area bboxes (single-row or single-column content) are also
    treated as specks and returned unchanged.
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
    pil_img = pil_img.convert('RGB')
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
    cells = arr.reshape(patch_grid, 14, patch_grid, 14)
    dark_frac = (cells < white_thresh).mean(axis=(1, 3))
    return (dark_frac >= content_frac_min).reshape(-1)


def preprocess_for_dinov2(pil_img):
    """Full preprocessing: bbox crop → silhouette mask → re-bbox-crop the
    masked output (so DINOv2 doesn't waste resolution on white margins) →
    valid_patch_mask on the final image.

    Returns (final_pil, patch_mask).
    """
    cropped = bbox_crop(pil_img)
    masked = extract_decoration_region(cropped)
    final = bbox_crop(masked)
    return final, valid_patch_mask(final)
