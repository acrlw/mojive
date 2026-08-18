"""Deterministic scene-light and shadow-slot scheduling."""

from forge_viewer.render.forge.passes.base import (
    LOCAL_SHADOW_SLOTS,
    MAX_SCENE_LIGHTS,
    schedule_lights,
)
from forge_viewer.types import Light, LightKind, LightSet


def test_light_schedule_caps_scene_and_local_shadow_slots():
    lights = tuple(Light(kind=LightKind.POINT) for _ in range(MAX_SCENE_LIGHTS + 5))
    schedule = schedule_lights(LightSet(lights=lights))

    assert len(schedule.lights) == MAX_SCENE_LIGHTS
    assert schedule.deferred_lights == 5
    assert schedule.local_shadows == tuple(range(LOCAL_SHADOW_SLOTS))
    assert schedule.selected_shadow_count == LOCAL_SHADOW_SLOTS
    assert schedule.deferred_shadows == len(lights) - LOCAL_SHADOW_SLOTS


def test_light_schedule_uses_source_order_and_skips_inactive_and_image_lights():
    lights = (
        Light(kind=LightKind.IMAGE),
        Light(kind=LightKind.POINT, active=False),
        Light(kind=LightKind.DIRECTIONAL),
        Light(kind=LightKind.DIRECTIONAL),
        Light(kind=LightKind.SPOT),
    )
    schedule = schedule_lights(LightSet(lights=lights))

    assert [light.kind for light in schedule.lights] == [
        LightKind.DIRECTIONAL,
        LightKind.DIRECTIONAL,
        LightKind.SPOT,
    ]
    assert schedule.directional_shadow == 0
    assert schedule.local_shadows == (2,)
    assert schedule.selected_shadow_count == 2
    assert schedule.deferred_shadows == 1
