from __future__ import annotations

from pathlib import Path

import pytest

from forge_viewer.assets import (
    ASSET_SUFFIXES,
    AssetNotFoundError,
    assets_dir,
    list_assets,
    resolve,
)

EXPECTED_ASSETS = (
    "actuator_visuals.xml",
    "deformables.xml",
    "dense_mesh.xml",
    "gizmo.xml",
    "joint_types.xml",
    "many_lights.xml",
    "many_objects.xml",
    "material_matrix.xml",
    "mocap_equality.xml",
    "mujoco_visuals.xml",
    "outline.xml",
    "parity_scene.xml",
    "parity_texture.xml",
    "perturb_ghost.xml",
    "pick_scene.xml",
    "rangefinder.xml",
    "reflection.xml",
    "scale_extremes.xml",
    "showcase.xml",
    "sunlight_shadow.xml",
    "test_scene.urdf",
    "test_scene.xml",
    "transparency.xml",
)


def test_assets_dir_is_the_repo_own_one() -> None:

    root = assets_dir()
    assert root.is_dir()
    assert (root / "test_scene.xml").is_file()


def test_list_assets_has_every_scene() -> None:
    assert list(list_assets()) == list(EXPECTED_ASSETS)


def test_resolve_bare_name() -> None:

    assert resolve("test_scene").name == "test_scene.xml"


def test_resolve_name_with_suffix() -> None:

    assert resolve("test_scene.xml").name == "test_scene.xml"
    assert resolve("test_scene.urdf").name == "test_scene.urdf"


def test_resolve_absolute_path() -> None:

    target = assets_dir() / "showcase.xml"
    assert resolve(str(target)) == target.resolve()


def test_resolve_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    scene = tmp_path / "local_scene.xml"
    scene.write_text("<mujoco/>", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert resolve("local_scene.xml") == scene.resolve()
    assert resolve("./local_scene.xml") == scene.resolve()


def test_bare_name_prefers_xml_over_urdf() -> None:

    assert ASSET_SUFFIXES[0] == ".xml"
    assert resolve("test_scene") == (assets_dir() / "test_scene.xml").resolve()


def test_missing_asset_lists_the_available_names() -> None:

    with pytest.raises(AssetNotFoundError) as err:
        resolve("no_such_scene")
    text = str(err.value)
    assert "showcase" in text
    assert "test_scene" in text
    assert "no_such_scene" in text


def test_missing_asset_with_suffix_also_lists_names() -> None:

    with pytest.raises(AssetNotFoundError) as err:
        resolve("no_such_scene.xml")
    assert "showcase" in str(err.value)


def test_empty_name_is_reported() -> None:
    with pytest.raises(AssetNotFoundError):
        resolve("")


@pytest.mark.physics
@pytest.mark.parametrize("name", EXPECTED_ASSETS)
def test_every_scene_loads(name: str) -> None:

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(resolve(name)))
    assert model.ngeom > 0, f"{name} 加载成功但一个 geom 都没有"


@pytest.mark.physics
def test_many_objects_is_actually_many() -> None:

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(resolve("many_objects.xml")))
    assert model.ngeom >= 400


@pytest.mark.physics
def test_default_scene_has_no_moving_parts() -> None:

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(resolve("test_scene.xml")))
    assert model.njnt == 0
    assert model.nu == 0


BACKGROUND_GEOMS = frozenset({"floor", "wall"})


@pytest.mark.physics
@pytest.mark.parametrize(
    ("name", "min_selectable"),
    [("test_scene.xml", 8), ("pick_scene.xml", 30), ("outline.xml", 8)],
)
def test_interaction_scenes_are_selectable(name: str, min_selectable: int) -> None:

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(resolve(name)))

    selectable = 0
    for i in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom[{i}]"
        body = int(model.geom_bodyid[i])
        if gname in BACKGROUND_GEOMS:
            assert body == 0, f"{name} 的背景 geom {gname} 挂在 body {body} 上——它会变得能选中"
            continue
        assert body != 0, f"{name} 的 {gname} 挂在 worldbody 上，object_id 是 0，点不中"
        selectable += 1

    assert selectable >= min_selectable, f"{name} 只有 {selectable} 个可选中的 geom"


@pytest.mark.physics
def test_pick_scene_has_a_multi_geom_body() -> None:

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(resolve("pick_scene.xml")))
    counts: dict[int, int] = {}
    for body in model.geom_bodyid:
        if int(body) != 0:
            counts[int(body)] = counts.get(int(body), 0) + 1
    multi = [b for b, n in counts.items() if n >= 2]
    assert len(multi) >= 2, "pick_scene 里没有两个以上的多 geom body"
    assert max(counts.values()) >= 3


@pytest.mark.physics
def test_joint_types_covers_every_kind() -> None:

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(resolve("joint_types.xml")))
    kinds = {int(t) for t in model.jnt_type}
    for want in (
        mujoco.mjtJoint.mjJNT_FREE,
        mujoco.mjtJoint.mjJNT_BALL,
        mujoco.mjtJoint.mjJNT_SLIDE,
        mujoco.mjtJoint.mjJNT_HINGE,
    ):
        assert int(want) in kinds, f"joint_types 里没有 {want}"

    limited = {bool(x) for x in model.jnt_limited}
    assert limited == {True, False}


@pytest.mark.physics
def test_many_lights_has_many_lights() -> None:

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(resolve("many_lights.xml")))
    assert model.nlight >= 6
    kinds = {int(t) for t in model.light_type}
    assert len(kinds) >= 3, f"八盏灯只有 {len(kinds)} 种类型"


@pytest.mark.physics
def test_transparency_scene_has_both_kinds() -> None:

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(resolve("transparency.xml")))
    alpha = [
        float(model.mat_rgba[int(model.geom_matid[i]), 3])
        if int(model.geom_matid[i]) >= 0
        else float(model.geom_rgba[i, 3])
        for i in range(model.ngeom)
    ]
    assert sum(a < 1.0 for a in alpha) >= 5, "半透 geom 少于五片"
    assert sum(a >= 1.0 for a in alpha) >= 3, "没有不透明 geom 当参照"


@pytest.mark.physics
def test_dense_mesh_has_a_real_mesh_asset() -> None:

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(resolve("dense_mesh.xml")))
    assert model.nmesh >= 1
    assert int(model.mesh_facenum.max()) >= 1000, "最密的一份网格还不到一千个三角形"


@pytest.mark.physics
def test_scale_extremes_spans_two_orders() -> None:

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(resolve("scale_extremes.xml")))

    sizes = [
        float(model.geom_size[i, 0])
        for i in range(model.ngeom)
        if int(model.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
    ]
    assert max(sizes) / min(sizes) >= 50.0
    assert min(sizes) <= 0.02
