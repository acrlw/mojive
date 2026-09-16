"""Shared editor refusal messages, independent of panels and input tools."""

GIZMO_REFUSAL_RUNNING = "Physics is running; pause to move things"
GIZMO_REFUSAL_DRIVEN = "This link is joint-driven; use its joint gizmo or the Joints panel"


def gizmo_refusal_reason(
    paused: bool,
    posable: bool,
) -> str | None:
    if not paused:
        return GIZMO_REFUSAL_RUNNING
    if not posable:
        return GIZMO_REFUSAL_DRIVEN
    return None
