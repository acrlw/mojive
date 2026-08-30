from __future__ import annotations

import pytest

from forge_viewer.ui.input_bindings import (
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
