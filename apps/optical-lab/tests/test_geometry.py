"""Optical compositor geometry (D-001..D-004, DSP-004, DSP-005, B-003, B-010).

Expected values are derived by hand in the comments, NOT copied from the
implementation. Coordinates are continuous: pixel i covers [i, i+1).
"""
import math

import numpy as np
import pytest

from hcc_lab.geometry import (
    GeometryParams,
    Offset,
    Viewport,
    build_transform,
    reflect_point,
    warp,
)

SRC = (200, 100)  # w, h
DST = (400, 400)


def mapped(params: GeometryParams, pt, src=SRC, dst=DST):
    t = build_transform(params, src[0], src[1], dst[0], dst[1])
    return t.apply_point(*pt)


def approx(a, b):
    return pytest.approx(b, abs=1e-9) == a


def test_identity_fit_preserves_aspect_and_centers():
    # fit f = min(400/200, 400/100) = 2 -> 400x200, centred vertically: ty = 100
    p = GeometryParams()
    assert approx(mapped(p, (0, 0)), (0, 100))
    assert approx(mapped(p, (200, 100)), (400, 300))
    assert approx(mapped(p, (100, 50)), (200, 200))


def test_mirror_x_moves_left_marker_to_right_side():
    # mirror: u -> 200 - u, then x2: left marker at u=10 lands at x=380
    p = GeometryParams(mirror_x=True)
    assert approx(mapped(p, (10, 50)), (380, 200))
    assert approx(mapped(p, (0, 0)), (400, 100))


def test_mirror_y_moves_top_marker_to_bottom():
    p = GeometryParams(mirror_y=True)
    assert approx(mapped(p, (0, 0)), (0, 300))


def test_rotation_90_is_clockwise():
    # rotated frame 100x200, (u,v)->(100-v, u); f = 2 -> 200x400, tx = 100
    p = GeometryParams(rotation_deg=90)
    assert approx(mapped(p, (0, 0)), (300, 0))       # top-left -> top-right
    assert approx(mapped(p, (0, 100)), (100, 0))     # bottom-left -> top-left
    assert approx(mapped(p, (200, 0)), (300, 400))   # top-right -> bottom-right


def test_rotation_180_and_270():
    assert approx(mapped(GeometryParams(rotation_deg=180), (0, 0)), (400, 300))
    # 270: (u,v)->(v, 200-u) in a 100x200 frame; f=2, tx=100 -> (0,0)->(100, 400)
    assert approx(mapped(GeometryParams(rotation_deg=270), (0, 0)), (100, 400))


def test_viewport_places_content_inside_region():
    # vp = (200, 0, 200, 200); f = min(1, 2) = 1; ty = (200-100)/2 = 50
    p = GeometryParams(viewport=Viewport(0.5, 0.0, 0.5, 0.5))
    assert approx(mapped(p, (0, 0)), (200, 50))
    assert approx(mapped(p, (200, 100)), (400, 150))


def test_offset_is_fraction_of_viewport():
    # tx = 0.1*400 = 40, ty = 100 - 0.1*400 = 60
    p = GeometryParams(offset=Offset(0.1, -0.1))
    assert approx(mapped(p, (0, 0)), (40, 60))


def test_scale_multiplies_fit_factor():
    # f = 2*0.5 = 1 -> 200x100 centred: (100, 150)
    p = GeometryParams(scale=0.5)
    assert approx(mapped(p, (0, 0)), (100, 150))


def test_aspect_ratio_constant_for_16_9_and_4_3_streams():
    # DSP-005: a circle stays a circle when the stream resolution changes.
    for src in ((1280, 720), (640, 480), (480, 640)):
        t = build_transform(GeometryParams(), src[0], src[1], 1080, 1920)
        sx = math.dist(t.apply_point(0, 0), t.apply_point(1, 0))
        sy = math.dist(t.apply_point(0, 0), t.apply_point(0, 1))
        assert sx == pytest.approx(sy)


def test_order_of_operations_rotate_before_mirror():
    # rotate 90 CW then mirror_x is NOT the same as mirror then rotate.
    # rotate: (0,0)->(100,0) in 100x200 frame; mirror_x: -> (0,0); f=2, tx=100 -> (100,0)
    p = GeometryParams(rotation_deg=90, mirror_x=True)
    assert approx(mapped(p, (0, 0)), (100, 0))


def test_warp_mirror_x_is_exact_pixel_flip():
    src = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    t = build_transform(GeometryParams(mirror_x=True), 3, 2, 3, 2)
    out = warp(src, t, 3, 2)
    assert np.array_equal(out, src[:, ::-1])


def test_warp_rotation_90_matches_numpy_clockwise():
    src = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    t = build_transform(GeometryParams(rotation_deg=90), 3, 2, 2, 3)
    out = warp(src, t, 2, 3)
    assert np.array_equal(out, np.rot90(src, k=-1))


def test_warp_outside_viewport_is_black():
    src = np.full((10, 10, 3), 255, dtype=np.uint8)
    t = build_transform(GeometryParams(viewport=Viewport(0.25, 0.25, 0.5, 0.5), scale=3.0), 10, 10, 40, 40)
    out = warp(src, t, 40, 40, clip=Viewport(0.25, 0.25, 0.5, 0.5))
    assert out[:10, :, :].max() == 0 and out[30:, :, :].max() == 0  # clipped to the viewport
    assert out[20, 20].tolist() == [255, 255, 255]


def test_reflect_point_plane_mirror_formula():
    # TZ 6.1: p' = p - 2 n (n . (p - p0))
    assert reflect_point((1, 2, 3), (0, 0, 0), (0, 0, 1)) == pytest.approx((1, 2, -3))
    # 45-degree plate through origin, n = (0, 1, -1)/sqrt2: screen point y=5 -> height 5
    n = (0.0, 1 / math.sqrt(2), -1 / math.sqrt(2))
    assert reflect_point((0, 5, 0), (0, 0, 0), n) == pytest.approx((0, 0, 5))


def test_reflect_point_rejects_zero_normal():
    with pytest.raises(ValueError):
        reflect_point((1, 1, 1), (0, 0, 0), (0, 0, 0))


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_scale_rejected(bad):
    with pytest.raises(ValueError):
        build_transform(GeometryParams(scale=bad), 10, 10, 10, 10)


@pytest.mark.parametrize("rot", [45, 1, -90, 360])
def test_invalid_rotation_rejected(rot):
    with pytest.raises(ValueError):
        build_transform(GeometryParams(rotation_deg=rot), 10, 10, 10, 10)


@pytest.mark.parametrize("size", [(0, 10, 10, 10), (10, 10, 0, 10), (-5, 10, 10, 10)])
def test_zero_or_negative_sizes_rejected(size):
    with pytest.raises(ValueError):
        build_transform(GeometryParams(), *size)
