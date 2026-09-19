"""Adaptive display geometry retains identity, exact opt-out and scene ownership."""

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mojive import CameraView, Light, LightSet, RenderProduct, Scene, SceneRenderer
from mojive.render.mesh import builtin_mesh
from mojive.types import MeshKey, MeshShape, MeshUpdate


def settled(renderer):
    deadline = time.monotonic() + 15
    while True:
        image = renderer.render()
        stats = renderer._backend.target.frame.statistics
        if not stats.lod_meshes_pending:
            return image
        assert time.monotonic() < deadline, "LOD preparation timed out"


def test_lod_products_selection_opt_out_resize_and_shared_deformation():
    scene = Scene()
    mesh = builtin_mesh(MeshKey(MeshShape.SPHERE))
    item = scene.mesh(mesh, color=(0.8, 0.3, 0.1, 1))
    camera = CameraView(eye=np.array([0, -20, 5]), far=50)
    with (
        SceneRenderer(scene.source, renderer="bgfx", width=320, height=240, camera=camera) as r,
        SceneRenderer(scene.source, renderer="bgfx", width=320, height=240, camera=camera) as peer,
    ):
        for renderer in (r, peer):
            renderer.update(scene.frame)
            renderer._backend._style.selection_fill = False
            renderer._backend._style.selection_outline = False
            renderer._backend.runtime.configure(
                renderer._backend._scene_handle, renderer._backend._style
            )
        exact = r.render()
        r.set_flag("mesh_lod", True)
        adaptive = settled(r)
        backend = r._backend
        stats = backend.target.frame.statistics
        assert stats.lod_meshes_ready > 0
        assert stats.lod_instances > 0
        assert 0 < stats.color_triangles < len(mesh.indices) // 3
        ids = r.render(product=RenderProduct.OBJECT_ID)
        depth = r.render(product=RenderProduct.METRIC_DEPTH)
        np.testing.assert_array_equal(ids > 0, depth < camera.far)
        assert backend.target.frame.statistics.data_triangles > 0
        # A peer that shares uploaded meshes keeps exact display when opted out.
        np.testing.assert_array_equal(peer.render(), exact)
        backend._style.selected_id = item.object_id
        backend.runtime.configure(backend._scene_handle, backend._style)
        np.testing.assert_array_equal(r.render(), exact)
        assert backend.target.frame.statistics.lod_instances == 0
        backend._style.selected_id = 0
        backend.runtime.configure(backend._scene_handle, backend._style)
        r.set_flag("mesh_lod", False)
        np.testing.assert_array_equal(r.render(), exact)
        r.set_flag("mesh_lod", True)
        r.resize(640, 480)
        r.render()
        assert backend.target.frame.statistics.color_triangles >= stats.color_triangles
        r.resize(320, 240)
        r.update(
            replace(
                scene.frame,
                mesh_updates={item.mesh_key: MeshUpdate(mesh.positions * 0.7, mesh.normals)},
            )
        )
        deformed = r.render()
        assert backend.target.frame.statistics.lod_instances == 0
        assert not np.array_equal(deformed, exact)
        np.testing.assert_array_equal(peer.render(), exact)
        r.set_scene(scene.source)
        r.update(scene.frame)
        settled(r)
        assert backend.target.frame.statistics.lod_meshes_ready > 0
        output = Path("output/mesh-lod/acceptance")
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate((exact, adaptive), axis=1)).save(output / "sphere.png")


def test_lod_jobs_survive_immediate_source_replacement_and_release():
    scene = Scene()
    scene.sphere()
    with SceneRenderer(scene.source, renderer="bgfx", width=80, height=60) as r:
        r.set_flag("mesh_lod", True)
        for _ in range(5):
            r.set_scene(scene.source)
            r.update(scene.frame)
            r.render()
        empty = Scene()
        r.set_scene(empty.source)
        r.update(empty.frame)
        r.render()
        assert r._backend.target.frame.statistics.lod_meshes_pending == 0


def test_shadow_lod_uses_light_resolution_and_opt_out_invalidates_cache():
    light = Light(direction=np.array([0.3, 0.5, -1]))
    scene = Scene(lights=LightSet(lights=(light,)))
    scene.plane(size=(100, 100, 0.01))
    scene.sphere(position=(0, 0, 1))
    camera = CameraView(eye=np.array([3, -4, 3]), target=np.array([0, 0, 1]))
    with SceneRenderer(scene.source, renderer="bgfx", width=320, height=240, camera=camera) as r:
        r.update(scene.frame)
        exact = r.render()
        exact_stats = r._backend.target.frame.statistics
        r.set_flag("mesh_lod", True)
        settled(r)
        adaptive = r._backend.target.frame.statistics
        assert 0 < adaptive.shadow_triangles < exact_stats.shadow_triangles
        assert adaptive.color_triangles == exact_stats.color_triangles
        r.render()
        assert r._backend.target.frame.statistics.shadow_reused
        r.set_flag("mesh_lod", False)
        np.testing.assert_array_equal(r.render(), exact)
        assert r._backend.target.frame.statistics.shadow_rendered


@pytest.mark.parametrize("last_owner_action", ["disable", "replace", "release", "deform"])
def test_lod_is_lazy_and_released_with_last_enabled_scene(last_owner_action):
    scene = Scene()
    mesh = builtin_mesh(MeshKey(MeshShape.SPHERE))
    item = scene.mesh(mesh)
    camera = CameraView(eye=np.array([0, -20, 5]), far=50)
    with (
        SceneRenderer(scene.source, renderer="bgfx", width=80, height=60, camera=camera) as r,
        SceneRenderer(scene.source, renderer="bgfx", width=80, height=60, camera=camera) as peer,
    ):
        runtime = r._backend.runtime
        for renderer in (r, peer):
            renderer.update(scene.frame)
        initial_uploads = runtime.resource_stats().mesh_uploads
        exact = r.render()
        for _ in range(3):
            peer.render()
            stats = peer._backend.target.frame.statistics
            assert stats.lod_meshes_pending == stats.lod_meshes_ready == 0
        assert runtime.resource_stats().mesh_uploads == initial_uploads

        r.set_flag("mesh_lod", True)
        settled(r)
        prepared_uploads = runtime.resource_stats().mesh_uploads
        assert prepared_uploads > initial_uploads
        np.testing.assert_array_equal(peer.render(), exact)
        assert peer._backend.target.frame.statistics.color_triangles == 0
        peer.set_flag("mesh_lod", True)
        settled(peer)
        assert runtime.resource_stats().mesh_uploads == prepared_uploads

        # Turning off one user must not evict an enabled peer's shared levels.
        r.set_flag("mesh_lod", False)
        np.testing.assert_array_equal(r.render(), exact)
        assert r._backend.target.frame.statistics.lod_meshes_ready == 0
        r.set_flag("mesh_lod", True)
        settled(r)
        assert runtime.resource_stats().mesh_uploads == prepared_uploads
        peer.set_flag("mesh_lod", False)

        if last_owner_action == "disable":
            r.set_flag("mesh_lod", False)
        elif last_owner_action == "replace":
            empty = Scene()
            r.set_scene(empty.source)
            r.update(empty.frame)
        elif last_owner_action == "release":
            r.close()
        else:
            r.update(
                replace(
                    scene.frame,
                    mesh_updates={item.mesh_key: MeshUpdate(mesh.positions * 0.7, mesh.normals)},
                )
            )
            prepared_uploads = runtime.resource_stats().mesh_uploads
        np.testing.assert_array_equal(peer.render(), exact)
        # An opted-out scene must not keep a dormant chain alive. Re-enabling
        # prepares new resources instead of reusing an unbounded hidden cache.
        peer.set_flag("mesh_lod", True)
        settled(peer)
        assert runtime.resource_stats().mesh_uploads > prepared_uploads


def test_disabling_pending_lod_does_not_install_late_results():
    scene = Scene()
    scene.sphere()
    with SceneRenderer(scene.source, renderer="bgfx", width=80, height=60) as r:
        r.update(scene.frame)
        exact = r.render()
        uploads = r._backend.runtime.resource_stats().mesh_uploads
        for _ in range(3):
            r.set_flag("mesh_lod", True)
            r.render()
            assert r._backend.target.frame.statistics.lod_meshes_pending > 0
            r.set_flag("mesh_lod", False)
            for _ in range(3):
                np.testing.assert_array_equal(r.render(), exact)
                stats = r._backend.target.frame.statistics
                assert stats.lod_meshes_pending == stats.lod_meshes_ready == 0
                assert r._backend.runtime.resource_stats().mesh_uploads == uploads


def test_geometry_view_switches_retain_prepared_lods_without_new_uploads():
    scene = Scene()
    scene.sphere()
    scene.box(size=(0.8, 0.8, 0.8))
    source = replace(
        scene.source,
        geom_role=np.array([1, 2], np.uint8),
        geom_group_visible=np.array([True, False]),
    )
    camera = CameraView(eye=np.array([0, -20, 5]), far=50)
    with SceneRenderer(source, renderer="bgfx", width=320, height=240, camera=camera) as r:
        r.set_flag("mesh_lod", True)
        r.set_geometry_view("both")
        r.update(scene.frame)
        images = {"both": settled(r)}
        assert r._backend.target.frame.statistics.lod_meshes_ready > 0
        runtime = r._backend.runtime
        uploads = runtime.resource_stats().mesh_uploads
        for mode in ("collision", "visual", "both", "default") * 2:
            r.set_geometry_view(mode)
            image = r.render()
            np.testing.assert_array_equal(image, images.setdefault(mode, image))
            assert r._backend.target.frame.statistics.lod_meshes_pending == 0
            assert runtime.resource_stats().mesh_uploads == uploads
