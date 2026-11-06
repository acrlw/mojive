# Electron workspace menus

The native Mojive inventory in `python/ui/workspace/menu_catalog.py` defines the
group order and command meanings. Electron's supported subset lives in
`main/menu-catalog.json`, shared by the macOS application menu and the in-window
menus on other platforms. This is not a complete port of every native command.

| Native group | Electron mapping |
| --- | --- |
| File | Open Scene and Save Scene As retain their scene-document meaning. Open Model contains MJCF and URDF entry points. Project folder open/close are Electron additions. |
| Edit | Undo, Redo, object visibility and deletion use the same command boundaries as the inspector and scene tree. Native text editing remains owned by the focused field. |
| Add | Box, sphere, cylinder, capsule and plane precede the separator for point light and camera. |
| View | Frame All, Frame Selection and scene capture precede projection, transform tool and body/world frame submenus, followed by transform-handle visibility. |
| Simulate | Play/Pause, Go to Start and simulation recording precede history stepping, playback speed and Physics Options. Physics Options opens Environment; it does not open the actuator panel. |
| Window | Sidebar and inspector visibility precede panel selection, Settings and Physics Options. The second Physics Options entry follows the native inventory. |
| Help | Documentation, Interaction Guide and About use distinct actions. macOS owns its Help search field and other system menu roles. |

Samples belong to the asset browser. They are not presented as native File menu
commands. Shared labels use Camera consistently in English and 相机 in Chinese.
Open Scene uses Command/Ctrl+O; Save Scene As uses Command/Ctrl+Shift+S.

## Capability boundaries

Unavailable native features are omitted instead of being relabeled as unrelated
Electron commands:

- New Scene, in-place Save, Reload Model, Add/Remove Models and resource-directory
  management need complete source and unsaved-document transactions.
- Cone, ellipsoid and site creation are not implemented by the current creator.
- Viewport-UI/window capture and live recording of those surfaces, including their
  recording settings, are not represented by timeline video export.
- Layout presets, sensor plots, statistics and a physics-adapter selector are not
  exposed by the current workspace.

Add/Delete availability reflects actual command capabilities, including loading,
recording and physics-owned scene data. Modal state blocks scene commands while
native Undo/Redo/Delete can still edit a focused text field. A hidden sidebar has
no checked panel. Focus shortcut remapping accepts the same directional keys as
the input settings and translates them to Electron accelerator names.

When the timeline owns keyboard focus, its Focus, Delete and clipboard shortcuts
remain local. Native clipboard menu entries are temporarily unavailable in that
scope; explicit menu commands for timeline clipboard operations are not yet
implemented. Text fields within the timeline retain native text editing.

## Verification

`tests/native-menu.test.mjs` checks catalog groups, separators, Chinese labels,
capabilities, modal text editing, panel state and keyboard ownership.
`tests/menu-interactions.test.mjs` checks real fallback menus with pointer and
keyboard navigation, nested popovers, checked actions, platform hints, and scene
open/save shortcuts. The latter produces
`output/electron/verification/workspace-menu-projection.png` from the production
components for visual inspection.

Native menu icon generation runs before desktop development startup and during
production asset packaging, including icons referenced by nested menu entries.
