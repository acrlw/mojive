# Viewer construction

The composition helpers create the native window, selected renderer backend, session, scene
adapter, debug bridge, and editor UI as one owned `Viewer`. The default is the full desktop
experience, with all built-in panels enabled; reducing features requires an explicit user choice.

The [configuration reference](../reference/configuration.md#choose-the-amount-of-ui-to-load)
distinguishes the UI-free offscreen `Renderer` from the interactive Viewer and documents optional
feature reduction for embedding applications.

## Viewer lifecycle and capture

::: mojive.app.composition

## Passive MuJoCo viewing

For MuJoCo-style calls, see the [viewer migration guide](../tutorials/mujoco-viewer.md).

::: mojive.viewer

The process-based extension remains a separate API:

::: mojive.app.passive

## Backend discovery

::: mojive.app.backends

## Built-in programmatic scenes

::: mojive.app.demos
