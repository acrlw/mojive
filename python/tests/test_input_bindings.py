from __future__ import annotations

import pytest

from mojive.ui.input_bindings import (
    DEFAULT_INPUT_BINDINGS,
    InputAction,
    InputBindings,
    key_choice,
)


def test_remap_swaps_the_displaced_action_without_duplicate_keys() -> None:
    remapped = DEFAULT_INPUT_BINDINGS.remap(InputAction.FRAME_SCENE, "g")

    assert remapped.key_id(InputAction.FRAME_SCENE) == "g"
    assert remapped.key_id(InputAction.GIZMO_TRANSLATE) == "f"
    identifiers = [remapped.key_id(action) for action in InputAction]
    assert len(identifiers) == len(set(identifiers))


def test_step_back_defaults_to_backspace() -> None:
    assert DEFAULT_INPUT_BINDINGS.key_id(InputAction.STEP_BACK) == "backspace"
    assert DEFAULT_INPUT_BINDINGS.label(InputAction.STEP_BACK) == "Backspace"


def test_dimensions_tool_is_unbound_by_default() -> None:
    assert DEFAULT_INPUT_BINDINGS.key_id(InputAction.GIZMO_DIMENSIONS) is None
    assert DEFAULT_INPUT_BINDINGS.label(InputAction.GIZMO_DIMENSIONS) == "Unbound"


def test_saved_input_bindings_restore_the_same_map() -> None:
    changed = DEFAULT_INPUT_BINDINGS.remap(InputAction.SNAP, "x").remap(
        InputAction.FLY_FORWARD, "digit_1"
    )

    restored = InputBindings.from_preferences(changed.preferences())

    assert restored.preferences() == changed.preferences()


def test_invalid_saved_input_bindings_are_ignored() -> None:
    restored = InputBindings.from_preferences(
        {
            InputAction.FRAME_SCENE.value: "not-a-key",
            InputAction.SNAP.value: 42,
        }
    )

    assert restored == DEFAULT_INPUT_BINDINGS


def test_unknown_remap_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported viewport key"):
        DEFAULT_INPUT_BINDINGS.remap(InputAction.FRAME_SCENE, "not-a-key")

    with pytest.raises(ValueError, match="Unsupported viewport key"):
        key_choice("not-a-key")


def test_action_can_be_explicitly_unbound_and_restored() -> None:
    changed = DEFAULT_INPUT_BINDINGS.remap(InputAction.FLY_FORWARD, None)

    assert changed.key_id(InputAction.FLY_FORWARD) is None
    assert changed.label(InputAction.FLY_FORWARD) == "Unbound"
    assert InputBindings.from_preferences(changed.preferences()) == changed


@pytest.mark.parametrize(
    "preset,buttons,keys,expected",
    [
        ("Mojive", (0,), (), "camera.orbit"),
        ("Mojive", (0,), ("shift",), "camera.pan"),
        ("Mojive", (0,), ("ctrl",), "perturb.translate"),
        ("Blender", (2,), (), "camera.orbit"),
        ("Blender", (2,), ("shift",), "camera.pan"),
        ("Blender", (2,), ("ctrl",), "camera.dolly_drag"),
        ("Unity", (0,), ("alt",), "camera.orbit"),
        ("Unreal", (2,), ("alt",), "camera.pan"),
        ("MuJoCo", (2,), (), "camera.dolly_drag"),
    ],
)
def test_pointer_navigation_presets_use_physical_chords(preset, buttons, keys, expected):
    from mojive.ui.pointer_bindings import PointerAction, PointerFrame

    bindings = DEFAULT_INPUT_BINDINGS.navigation_preset(preset)
    frame = PointerFrame(buttons=frozenset(buttons), keys=frozenset(keys))
    assert bindings.pointer_match(PointerAction(expected), frame) is not None
    assert InputBindings.from_preferences(bindings.preferences()) == bindings


def test_mouse_combinations_conflicts_and_modifier_remaps_are_atomic():
    from mojive.ui.pointer_bindings import PointerAction, PointerChord, PointerFrame

    original = DEFAULT_INPUT_BINDINGS
    changed = original.remap_pointer(PointerAction.PAN, ("left+right",))
    both = PointerFrame(buttons=frozenset((0, 1)))
    assert changed.pointer_match(PointerAction.PAN, both) == PointerChord.parse("left+right")
    assert changed.pointer_match(PointerAction.ORBIT, both) is None
    assert InputBindings.from_preferences(changed.preferences()) == changed
    with pytest.raises(ValueError, match="already used"):
        original.remap_pointer(PointerAction.PAN, ("left",))
    with pytest.raises(ValueError):
        original.remap_pointer(PointerAction.PAN, ("ctrl+banana",))
    assert original.pointer_chords(PointerAction.PAN) != changed.pointer_chords(PointerAction.PAN)
    remapped = original.remap(InputAction.PERTURB, "q")
    assert remapped.pointer_match(
        PointerAction.PERTURB_TRANSLATE,
        PointerFrame(buttons=frozenset((0,)), keys=frozenset(("q",))),
    )
    assert not remapped.pointer_match(
        PointerAction.PERTURB_TRANSLATE,
        PointerFrame(buttons=frozenset((0,)), keys=frozenset(("ctrl",))),
    )


def test_custom_focus_and_timeline_gestures_drive_their_hints():
    from mojive.ui.panels.keyframes import timeline_status_hints
    from mojive.ui.pointer_bindings import PointerAction, PointerFrame
    from mojive.ui.viewport_widgets import default_tool_hints

    bindings = DEFAULT_INPUT_BINDINGS.remap_pointer(PointerAction.TIMELINE_PAN, ("alt+middle",))
    bindings = bindings.remap_pointer(PointerAction.ORBIT, ("alt+left",))
    bindings = bindings.remap_pointer(PointerAction.FOCUS, ("middle:double",))
    pan = next(
        hint
        for hint in timeline_status_hints(str, bindings=bindings)
        if hint.hint_id == "keyframes.pan"
    )
    assert (pan.control, pan.modifier) == ("middle", "Alt")
    orbit = next(
        hint for hint in default_tool_hints("camera", bindings) if hint.hint_id == "camera.orbit"
    )
    assert (orbit.control, orbit.modifier) == ("left", "Alt")
    assert not bindings.pointer_match(PointerAction.FOCUS, PointerFrame(buttons=frozenset((2,))))
    assert bindings.pointer_match(
        PointerAction.FOCUS, PointerFrame(buttons=frozenset((2,)), doubled=frozenset((2,)))
    )
    assert not bindings.remap_pointer(PointerAction.TIMELINE_PAN, ()).pointer_chords(
        PointerAction.TIMELINE_PAN
    )
