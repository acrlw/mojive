"""Contact marker contracts, batching, orientation, and model visual dimensions."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from mojive import ContactStyle
from mojive.adapters.base import DiagnosticSource, SceneFrame, SceneSource
from mojive.render.backend import RenderFlag
from mojive.render.debugdraw import DebugDraw, DrawPath, PrimitiveType
from mojive.render.overlay import OverlayPublisher


def test_contact_style_round_trip_and_validation():
    style = ContactStyle("sphere", [0.2, 0.8, 0.9, 0.7], 2, True)
    assert ContactStyle(**asdict(style)) == style
    assert isinstance(style.color, tuple)
    for kwargs in (
        {"shape": "bad"},
        {"color": (1, 0, 0)},
        {"color": (float("nan"), 0, 0, 1)},
        {"scale": 0},
        {"scale": float("inf")},
        {"use_model_color": "false"},
    ):
        with pytest.raises(ValueError):
            ContactStyle(**kwargs)


def test_contact_markers_batch_orient_scale_and_clear_without_mutating_frame():
    source = SceneSource(
        diagnostics=DiagnosticSource(
            contact_point_rgba=np.array((1, 1, 0, 1), np.float32),
            contact_point_radius=0.2,
            contact_point_half_height=0.05,
        )
    )
    normals = np.array(((1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 2, 3)), np.float32)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    contacts = np.column_stack((np.arange(12).reshape(4, 3), normals, np.ones(4))).astype(
        np.float32
    )
    frame = SceneFrame(contacts=contacts)
    original = contacts.copy()
    flags = {RenderFlag.CONTACTPOINT: True}
    debug = DebugDraw()
    publisher = OverlayPublisher(debug, flags)
    publisher.set_scene(source)
    layer = debug.layer("physics.contact.points")
    for shape, primitive in (
        ("cylinder", PrimitiveType.CYLINDER),
        ("sphere", PrimitiveType.SPHERE),
        ("point", PrimitiveType.POINT),
    ):
        publisher.contact_style = ContactStyle(shape=shape, scale=2)
        publisher._publish_contacts(frame)
        assert debug.primitives == 4 and len(layer._index) == 1
        assert layer.count_of(primitive) == 4
        packed = debug.build()
        assert packed.batch_count == 1
        store = layer._stores[primitive]
        assert store.colors[:4] == pytest.approx(np.tile(ContactStyle().color, (4, 1)))
        if shape == "point":
            assert store.sizes[:4] == pytest.approx(8)
        else:
            transforms = store.transforms[:4]
            assert transforms[:, :3, 3] == pytest.approx(contacts[:, :3])
            expected = (0.4, 0.4, 0.1 if shape == "cylinder" else 0.4)
            assert np.linalg.norm(transforms[:, :3, :3], axis=1) == pytest.approx(
                np.tile(expected, (4, 1))
            )
            if shape == "cylinder":
                assert transforms[:, :3, 2] == pytest.approx(normals * 0.1)
        publisher._publish_contacts(SceneFrame(contacts=contacts[:1]))
        assert debug.primitives == 1
    assert frame.contacts == pytest.approx(original)
    publisher.contact_style = replace(publisher.contact_style, use_model_color=True)
    publisher._publish_contacts(frame)
    assert layer._stores[PrimitiveType.POINT].colors[:4] == pytest.approx(
        np.tile((1, 1, 0, 1), (4, 1))
    )
    frame.contact_island_rgba = np.eye(4, dtype=np.float32)
    flags[RenderFlag.ISLAND] = True
    publisher._publish_contacts(frame)
    assert layer._stores[PrimitiveType.POINT].colors[:4] == pytest.approx(frame.contact_island_rgba)
    flags[RenderFlag.CONTACTPOINT] = False
    publisher._publish_contacts(frame)
    assert debug.primitives == 0
    flags[RenderFlag.CONTACTPOINT] = True
    publisher._publish_contacts(frame)
    publisher._publish_contacts(SceneFrame())
    assert debug.primitives == 0


@pytest.mark.parametrize("method", ["spheres", "cylinders"])
def test_invalid_solid_batch_retains_previous_draw(method):
    draw = DebugDraw()
    layer = draw.layer("test")
    paint = getattr(layer, method)
    matrices = np.eye(4, dtype=np.float32)[None]
    paint("same", matrices, (1, 0, 0, 1))
    before = draw.build().stream(DrawPath.SOLID).copy()
    with pytest.raises(ValueError):
        paint("same", matrices, np.ones((2, 4)))
    assert draw.build().stream(DrawPath.SOLID) == pytest.approx(before)


@pytest.mark.physics
def test_contact_marker_dimensions_follow_mujoco_visual_scale():
    import mujoco

    from mojive.adapters.mujoco import MuJoCoAdapter

    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <visual><scale contactwidth=".2" contactheight=".03"/></visual>
      <worldbody><body><geom size=".3"/></body></worldbody></mujoco>""")
    adapter = MuJoCoAdapter()
    try:
        adapter.load_model(model)
        diagnostics = adapter.scene_source().diagnostics
        assert diagnostics.contact_point_radius == pytest.approx(model.stat.meansize * 0.2)
        assert diagnostics.contact_point_half_height == pytest.approx(model.stat.meansize * 0.03)
    finally:
        adapter.release()
