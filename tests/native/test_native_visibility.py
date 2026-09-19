"""Directional shadow visibility retains offscreen influence and cache coverage."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mojive import CameraView, Light, LightSet, Material, RenderProduct, Scene, SceneRenderer


@pytest.mark.parametrize("reflectivity", [0.0, 0.4])
def test_directional_culling_matches_complete_maps_and_expands_cached_coverage(reflectivity):
    scene = Scene(
        lights=LightSet(
            lights=(Light(direction=np.array([0.7, 0, -1])),),
            ambient=np.full(3, 0.15, np.float32),
        )
    )
    scene.plane(size=(25, 25, 0.01), material=Material(reflectance=reflectivity))
    for x in range(-18, 19, 6):
        for y in range(-18, 19, 6):
            if x or y:
                scene.box(position=(x, y, 0.5), size=(0.4, 0.4, 0.5))
    # This box is outside the narrow camera, but its shadow crosses the center.
    caster = scene.box(position=(-4, 0, 5), size=(0.4, 0.4, 0.4))
    narrow = CameraView(
        eye=np.array([0, 0, 40]),
        up=np.array([0, 1, 0]),
        orthographic=True,
        ortho_height=5,
        far=100,
    )
    wide = replace(narrow, ortho_height=100)
    with (
        SceneRenderer(scene.source, renderer="bgfx", width=240, height=240, camera=narrow) as r,
        SceneRenderer(scene.source, renderer="bgfx", width=240, height=240, camera=wide) as full,
    ):
        r.update(scene.frame)
        full.update(scene.frame)
        full.render()
        full_count = full._backend.target.frame.statistics.shadow_instances
        initial = r.render()
        assert r._backend.target.frame.statistics.shadow_instances < full_count / 3
        # Priming a complete map supplies an independent, unfiltered reference.
        full.update(scene.frame, camera=narrow)
        reference = full.render()
        assert full._backend.target.frame.statistics.shadow_reused
        np.testing.assert_array_equal(initial, reference)
        ids = r.render(product=RenderProduct.OBJECT_ID)
        assert caster.object_id not in ids
        r.set_flag("shadow", False)
        unshadowed = r.render()
        assert np.count_nonzero(np.any(initial != unshadowed, axis=2)) > 100
        r.set_flag("shadow", True)
        np.testing.assert_array_equal(r.render(), initial)

        # A changed projection can require new casters even with the same focus.
        # Conversely, contracting to a subset must reuse the larger cached map.
        for height in (20, 10, 100, 5):
            camera = replace(narrow, ortho_height=height)
            r.update(scene.frame, camera=camera)
            actual = r.render()
            stats = r._backend.target.frame.statistics
            assert stats.shadow_rendered if height in (20, 100) else stats.shadow_reused
            full.update(scene.frame, camera=camera)
            np.testing.assert_array_equal(actual, full.render())

        output = Path("output/shadow-visibility")
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate((initial, reference, unshadowed), axis=1)).save(
            output / f"offscreen-caster-{reflectivity}.png"
        )
