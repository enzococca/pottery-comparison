"""Pure-function tests for preprocess.py."""
from PIL import Image, ImageDraw
import numpy as np

from preprocess import (
    bbox_crop,
    ink_density,
    extract_decoration_region,
    valid_patch_mask,
    preprocess_for_dinov2,
)


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
    assert 55 < out.size[0] < 75
    assert 55 < out.size[1] < 75


def test_bbox_crop_padding_grows_bbox():
    img = _circle(200, 200, radius=30)
    out0 = bbox_crop(img, padding_ratio=0.0)
    out10 = bbox_crop(img, padding_ratio=0.1)
    assert out10.size[0] > out0.size[0]


def test_bbox_crop_stray_speck_returns_unchanged():
    """A 1-pixel speck has bbox_w*bbox_h = 0 < min_area_ratio*200*200, so
    bbox_crop falls through and returns the original."""
    img = _white(200, 200)
    ImageDraw.Draw(img).point((100, 100), fill='black')
    out = bbox_crop(img)
    assert out.size == img.size


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
    assert ink_density(img) > 0.9


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
    assert out_dark > 50
    assert out_dark < inp_dark


def test_extract_decoration_no_silhouette_returns_input():
    """An image with only a few small marks (no big closed contour) is
    treated as a decoration-only crop and returned unchanged."""
    img = _white(200, 200)
    d = ImageDraw.Draw(img)
    for y in [40, 60, 80]:
        d.line([20, y, 60, y], fill='black', width=2)
    out = extract_decoration_region(img)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(img))


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


def test_valid_patch_mask_corner_block_marks_only_corner_patch():
    img = _white(224, 224)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 13, 13], fill='black')   # exactly 14x14 = one patch
    m = valid_patch_mask(img).reshape(16, 16)
    assert m[0, 0]
    assert m[1:, 1:].sum() == 0   # nothing else should be valid


def test_preprocess_for_dinov2_returns_pil_and_mask():
    img = _circle_with_inner_lines(200, 200, radius=80)
    out_img, mask = preprocess_for_dinov2(img)
    assert isinstance(out_img, Image.Image)
    assert mask.shape == (256,)
    assert mask.dtype == np.bool_


def test_preprocess_for_dinov2_white_image_yields_empty_mask():
    out_img, mask = preprocess_for_dinov2(_white())
    assert not mask.any()
