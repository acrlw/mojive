from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from forge_viewer.render.forge import cascades as C

ATLAS = 4096
TILES = 2
TILE = ATLAS // TILES
SUN = np.array([0.3, 0.5, -1.0], np.float32)


def _set(focus=(0.0, 0.0, 0.0), extent=9.0, direction=SUN, clip=1.0, **kw):
    return C.build_cascades(direction, focus, extent, shadow_clip=clip, **kw)


def test_three_radii_are_r_over_9_3_1():

    extent, clip = 7.5, 1.4
    s = _set(extent=extent, clip=clip)
    r = extent * clip
    assert s.splits == pytest.approx([r / 9.0, r / 3.0, r], rel=1e-6)


def test_radius_scales_with_shadow_clip():

    a = _set(extent=4.0, clip=1.0).splits.copy()
    b = _set(extent=4.0, clip=2.5).splits.copy()
    assert b == pytest.approx(a * 2.5, rel=1e-6)


def test_cascades_are_concentric():

    s = _set(focus=(1.234, -2.345, 0.678), extent=9.0)
    for i in range(3):
        for j in range(3):
            slack = max(float(s.texel_world[i]), float(s.texel_world[j]))
            delta = float(np.max(np.abs(s.centers[i] - s.centers[j])))
            assert delta <= slack + 1e-5, f"第 {i} 级与第 {j} 级不同心：差 {delta} > {slack}"


def test_center_snaps_to_whole_texels():

    down = np.array([0.0, 0.0, -1.0], np.float32)
    assert C.light_basis(down) == pytest.approx(np.eye(3), abs=1e-6), (
        "前提：正下方的光空间即世界空间"
    )

    a = _set(focus=(0.0, 0.0, 0.0), extent=9.0, direction=down)
    texels = a.texel_world.copy()
    centers_a = a.centers.copy()

    step = np.array([0.371, -0.913, 0.157], np.float64) * float(texels[0])
    b = _set(focus=tuple(step), extent=9.0, direction=down)

    moved_any = False
    for i in range(3):
        t = float(texels[i])
        delta = (b.centers[i] - centers_a[i]).astype(np.float64)
        quotient = delta / t
        assert quotient == pytest.approx(np.round(quotient), abs=1e-3), (
            f"第 {i} 级的中心移动了 {delta}，不是纹素 {t} 的整数倍——吸附没生效"
        )
        moved_any = moved_any or float(np.max(np.abs(delta))) > 0.0

    assert moved_any, "平移了一整个第 0 级纹素还没有任何一级动过，判据在空转"


def test_snap_is_stable_under_sub_texel_jitter():

    down = np.array([0.0, 0.0, -1.0], np.float32)
    base = _set(focus=(0.0, 0.0, 0.0), extent=9.0, direction=down)
    t0 = float(base.texel_world[0])

    jitter = _set(focus=(0.1 * t0, 0.1 * t0, 0.0), extent=9.0, direction=down)
    assert jitter.centers == pytest.approx(base.centers, abs=1e-6)


def test_texel_world_ratio_equals_radius_ratio():

    s = _set(extent=9.0)
    assert float(s.texel_world[2]) / float(s.texel_world[1]) == pytest.approx(3.0, rel=1e-5)
    assert float(s.texel_world[2]) / float(s.texel_world[0]) == pytest.approx(9.0, rel=1e-5)

    assert s.texel_world == pytest.approx(2.0 * s.splits / TILE, rel=1e-5)


@pytest.mark.parametrize("pcf", [0, 1, 2, 3])
@pytest.mark.parametrize("slot", [0, 1, 2, 3])
def test_tile_uv_is_inset_by_pcf_radius_plus_half_texel(slot: int, pcf: int):

    col, row = slot % TILES, slot // TILES
    bare = np.array([col / TILES, row / TILES, (col + 1) / TILES, (row + 1) / TILES])
    margin = (pcf + 0.5) / ATLAS

    uv = C.slot_uv(slot, pcf_radius=pcf)
    assert uv[0] == pytest.approx(bare[0] + margin, abs=1e-9)
    assert uv[1] == pytest.approx(bare[1] + margin, abs=1e-9)
    assert uv[2] == pytest.approx(bare[2] - margin, abs=1e-9)
    assert uv[3] == pytest.approx(bare[3] - margin, abs=1e-9)


def test_cascade_tiles_do_not_overlap():

    s = _set()
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = s.tile_uv[i], s.tile_uv[j]
            disjoint = a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
            assert disjoint, f"第 {i} 级与第 {j} 级的瓦片重叠了"


def test_fourth_slot_is_free_and_addressable():

    s = _set()
    assert s.count == 3
    assert set(s.slots) == {0, 1, 2}

    assert C.slot_pixels(3) == (TILE, TILE, TILE, TILE)
    free = np.array([TILE / ATLAS, TILE / ATLAS, 1.0, 1.0])
    for i in range(s.count):
        a = s.tile_uv[i]
        disjoint = a[2] <= free[0] or free[2] <= a[0] or a[3] <= free[1] or free[3] <= a[1]
        assert disjoint, f"第 {i} 级占到了留给下一盏灯的那一块"

    with pytest.raises(ValueError):
        C.slot_pixels(4)


def test_matrix_maps_the_volume_onto_the_unit_cube():

    s = _set(focus=(1.0, -2.0, 0.5), extent=6.0)
    for i in range(3):
        m = s.matrices[i].astype(np.float64)
        center = s.centers[i].astype(np.float64)

        right = m[0, :3] / np.linalg.norm(m[0, :3])
        up = m[1, :3] / np.linalg.norm(m[1, :3])
        r = float(s.splits[i])

        def ndc(p, m=m):
            v = m @ np.array([p[0], p[1], p[2], 1.0])
            return v[:3] / v[3]

        assert ndc(center)[:2] == pytest.approx([0.0, 0.0], abs=1e-4)
        assert ndc(center + right * r)[0] == pytest.approx(1.0, abs=1e-4)
        assert ndc(center - right * r)[0] == pytest.approx(-1.0, abs=1e-4)
        assert ndc(center + up * r)[1] == pytest.approx(1.0, abs=1e-4)

        z_center = ndc(center)[2]
        assert -1.0 < z_center < 1.0
        light = C.light_basis(SUN)[2] * -1.0
        assert ndc(center + light * (0.1 * r))[2] > z_center


def test_depth_range_covers_the_whole_scene():

    extent = 9.0
    s = _set(focus=(0.0, 0.0, 0.0), extent=extent)
    light = C.light_basis(SUN)[2].astype(np.float64) * -1.0
    for i in range(3):
        m = s.matrices[i].astype(np.float64)
        for t in (-extent, -0.5 * extent, 0.5 * extent, extent):
            p = s.centers[i].astype(np.float64) + light * t
            z = (m @ np.array([p[0], p[1], p[2], 1.0]))[2]
            assert -1.0 <= z <= 1.0, f"第 {i} 级的深度范围盖不住光轴上 {t} 处的遮挡者"


_SHADER = Path(C.__file__).parent / "shaders" / "shadow_sample.glsl"


def test_glsl_pcf_radius_matches_python():

    src = _SHADER.read_text(encoding="utf-8")
    m = re.search(r"#define\s+SHADOW_PCF_RADIUS\s+(\d+)", src)
    assert m, "shadow_sample.glsl 里找不到 SHADOW_PCF_RADIUS"
    assert int(m.group(1)) == C.PCF_RADIUS


def test_glsl_bias_default_matches_python():

    from forge_viewer.render.forge.passes import shadow as S

    src = _SHADER.read_text(encoding="utf-8")
    m = re.search(r"FORGE_SHADOW_BIAS\s*=\s*vec2\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", src)
    assert m, "shadow_sample.glsl 里找不到 FORGE_SHADOW_BIAS"
    assert (float(m.group(1)), float(m.group(2))) == pytest.approx(S.SHADOW_BIAS)
