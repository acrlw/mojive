"""Quiet ground changes display appearance without changing scene or physics data."""

from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from mojive import GeometryStyle, RenderProduct, SceneRenderer
from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.types import CameraView
from mojive.ui.scene_capture import SceneCapture

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.gpu

MODEL_XML = """
<mujoco>
  <asset>
    <texture name="checker" type="2d" builtin="checker" width="64" height="64"
             rgb1=".9 .85 .75" rgb2=".1 .12 .15"/>
    <material name="checker" texture="checker" texuniform="true" texrepeat="3 3"
              specular="0" shininess="0" reflectance="0"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .1" material="checker"/>
    <body name="subject" pos="0 0 .5">
      <freejoint/>
      <geom name="box" type="box" size=".4 .3 .5" material="checker"/>
    </body>
  </worldbody>
</mujoco>
"""


def _model_bytes(model):
    buffer = np.zeros(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return buffer


def _physics_state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec), np.float64)
    mujoco.mj_getState(model, data, state, spec)
    return state


def _interior(mask):
    """Exclude antialiased edges where a pixel also contains the adjacent ground."""
    for _ in range(2):
        inside = np.zeros_like(mask)
        inside[1:-1, 1:-1] = (
            mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        )
        mask = inside
    return mask


def test_quiet_ground_restores_color_preserves_geometry_and_matches_capture(backend_name):
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    adapter = MuJoCoAdapter()
    adapter.load_model(model)
    capture = SceneCapture()
    try:
        source = adapter.scene_source()
        frame = adapter.frame(FrameNeeds())
        original_source = deepcopy(source)
        original_frame = deepcopy(frame)
        original_materials = tuple(source.materials)
        original_textures = dict(source.textures)
        original_model = _model_bytes(model)
        original_physics = _physics_state(model, adapter.data)
        # The same authored material must remain textured on the ordinary object.
        assert source.geom_material[0] == source.geom_material[1]
        assert source.geom_infinite_plane.tolist() == [True, False]
        floor_id, subject_id = map(int, source.geom_object_id)
        assert floor_id != subject_id
        session = SimpleNamespace(source=source, frame=frame, structure_generation=1)
        camera = CameraView(
            eye=np.array((2.5, -4, 3), np.float32),
            target=np.array((0, 0, 0.3), np.float32),
            up=np.array((0, 0, 1), np.float32),
        ).with_aspect(256 / 192)

        def assert_source_unchanged():
            assert adapter.scene_source() is source
            assert len(source.materials) == len(original_materials)
            for material, identity, expected in zip(
                source.materials, original_materials, original_source.materials, strict=True
            ):
                assert material is identity
                for item in fields(material):
                    np.testing.assert_equal(
                        getattr(material, item.name), getattr(expected, item.name)
                    )
            assert source.textures.keys() == original_textures.keys()
            for name, texture in source.textures.items():
                assert texture is original_textures[name]
                for item in fields(texture):
                    np.testing.assert_equal(
                        getattr(texture, item.name),
                        getattr(original_source.textures[name], item.name),
                    )
            assert source.geom_material == original_source.geom_material
            assert source.geom_mesh == original_source.geom_mesh
            for item in fields(source):
                value = getattr(source, item.name)
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, getattr(original_source, item.name))
            for item in fields(frame):
                np.testing.assert_equal(
                    getattr(frame, item.name), getattr(original_frame, item.name)
                )
            np.testing.assert_array_equal(_model_bytes(model), original_model)
            np.testing.assert_array_equal(_physics_state(model, adapter.data), original_physics)

        with SceneRenderer(
            source, width=256, height=192, renderer=backend_name, camera=camera
        ) as view:
            view.update(frame)
            baseline = view.render().copy()
            baseline_ids = view.render(product=RenderProduct.OBJECT_ID).copy()
            baseline_depth = view.render(product=RenderProduct.METRIC_DEPTH).copy()
            subject = _interior(baseline_ids == subject_id)
            ground = (
                (baseline_ids == floor_id) & (baseline_depth > 0) & (baseline_depth < camera.far)
            )
            assert subject.sum() > 100
            assert ground.sum() > 1000
            peer = None
            try:
                for quiet in (False, True, False):
                    style = GeometryStyle(quiet_ground=quiet)
                    view.set_geometry_style(style)
                    image = view.render().copy()
                    if quiet:
                        difference = np.max(
                            np.abs(image.astype(np.int16) - baseline.astype(np.int16)), axis=2
                        )
                        assert np.count_nonzero(ground & (difference > 10)) > 1000
                        np.testing.assert_array_equal(image[subject], baseline[subject])
                    else:
                        np.testing.assert_array_equal(image, baseline)
                    np.testing.assert_array_equal(
                        view.render(product=RenderProduct.OBJECT_ID), baseline_ids
                    )
                    np.testing.assert_array_equal(
                        view.render(product=RenderProduct.METRIC_DEPTH), baseline_depth
                    )
                    with view._current():
                        captured = capture.read(view._backend, session, camera)
                    np.testing.assert_array_equal(captured, image)
                    assert capture._backend.get_geometry_style() == style
                    if peer is None:
                        peer = capture._backend
                    else:
                        assert capture._backend is peer
                    assert_source_unchanged()
            finally:
                with view._current():
                    capture.release()
    finally:
        adapter.release()
