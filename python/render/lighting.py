"""Backend-neutral light and shadow-caster selection policy."""

from __future__ import annotations

from dataclasses import dataclass

from mojive.types import Light, LightSet, LightType

LOCAL_SHADOW_SLOTS = 8
MAX_SCENE_LIGHTS = 100


@dataclass(frozen=True)
class LightSchedule:
    """Selected lights and shadow slots, plus deferred scene counts."""

    lights: tuple[Light, ...]
    active_count: int
    directional_shadow: int
    local_shadows: tuple[int, ...]
    shadow_candidate_count: int

    @property
    def deferred_lights(self) -> int:
        return self.active_count - len(self.lights)

    @property
    def selected_shadow_count(self) -> int:
        return (self.directional_shadow >= 0) + len(self.local_shadows)

    @property
    def deferred_shadows(self) -> int:
        return self.shadow_candidate_count - self.selected_shadow_count


def schedule_lights(lights: LightSet) -> LightSchedule:
    """Select active non-image lights and bounded shadow casters in scene order."""
    active = tuple(
        light for light in lights.lights if light.active and light.type is not LightType.IMAGE
    )
    selected = active[:MAX_SCENE_LIGHTS]
    directional = next(
        (
            index
            for index, light in enumerate(selected)
            if light.cast_shadow and light.type is LightType.DIRECTIONAL
        ),
        -1,
    )
    local = tuple(
        index
        for index, light in enumerate(selected)
        if light.cast_shadow and light.type in (LightType.POINT, LightType.SPOT, LightType.AREA)
    )[:LOCAL_SHADOW_SLOTS]
    shadow_candidates = sum(
        light.cast_shadow
        and light.type in (LightType.DIRECTIONAL, LightType.POINT, LightType.SPOT, LightType.AREA)
        for light in active
    )
    return LightSchedule(selected, len(active), directional, local, shadow_candidates)
