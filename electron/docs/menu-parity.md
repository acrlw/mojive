# Electron workspace menus

The native inventory in `python/ui/workspace/menu_catalog.py`, its dispatch in
`python/ui/workspace/menu_actions.py`, and the receiving components define the
reference. Electron's supported subset lives in `main/menu-catalog.json`, shared
by the macOS application menu and the in-window menus on other platforms. A
matching label or menu entry alone does not establish behavior parity.

## Command inventory

The following inventory covers every native top-level menu command and submenu.
“Available” means that the Electron command has a backing implementation; it does
not claim identical rendering, serialization formats, or all backend capabilities.

| Group | Native command | Electron implementation and remaining differences |
| --- | --- | --- |
| File | New Scene | Missing. Requires a new document transaction with unsaved-change handling. |
| File | Open Scene | Available through `file.openScene`, loading an Electron scene document. Native and Electron document formats are not interchangeable by this menu mapping. |
| File | Save | Missing in-place save. The current save command always opens the destination picker. |
| File | Save As | Available through `file.save`, including a destination picker and scene serialization. |
| File | Open Model | Available as MJCF and URDF children of Open Model; replaces the current document through the loading coordinator. |
| File | Add Models / Remove Model | Missing model composition and per-model removal. Removing an authored object is not an equivalent command. |
| File | Reload Model | Missing. Reload must retain the actual source and honor the unsaved-document transaction. |
| File | Resource Directories: Add / Remove | Missing user-managed resource roots. Resolving a model's relative assets does not supply this feature. |
| File | Quit | Electron application close uses its document-close transaction. macOS places Quit in the application menu. A browser page has no equivalent application Quit command. |
| Edit | Undo / Redo | Available through document history; focused native text fields retain their own edit history. |
| Edit | Visible | Available; applies to the selected object's visibility. |
| Add | Box / Sphere / Cylinder / Cone / Ellipsoid | Available as authored scene geometry. Shape dimensions and placement require domain checks in addition to catalog checks. |
| Add | Plane | Available as authored scene geometry, labeled Ground. Native also inserts a `geom:plane` into the physics world's topology when supported; Electron's authored ground does not add a collision plane to the compiled model. |
| Add | Capsule | Available as authored scene geometry. Native `create:Capsule` instead inserts a `geom:capsule` into the selected model/body through its topology-editing command; Electron's authored capsule does not add a collision geom to the compiled model. |
| Add | Point Light / Camera | Available as authored scene entities. |
| Add | Site | Missing. Native Site creation is an adapter topology edit, not just a marker icon. |
| View | Frame All | Available; uses the viewport scene framing command. |
| View | Capture: Scene Image | Available on desktop through the current-view capture and PNG save path. Browser image saving is not implemented by this command. |
| View | Capture: Viewport with UI / Entire Window | Missing. Scene-only capture excludes the interface. |
| View | Record: Scene Only / Viewport with UI / Entire Window | Missing live surface recording. Timeline WebM export renders an existing recorded take; it does not record live camera interaction or these UI surfaces. |
| View | Cancel / Pause / Resume / Stop live recording | Missing with the live recording workflow above. Simulation take recording is a separate feature. |
| View | Recording Settings | Missing live recording configuration. |
| View | Projection: Perspective / Orthographic | Available through viewport projection commands. |
| View | Transform tool: Select / Move / Rotate / Scale | Available; Move and Rotate are labeled Position and Rotation. Joint is an additional Electron menu entry. |
| View | Gizmo orientation: World / Body | Available through the shared frame setting. |
| View | Transform handles | Available through viewport gizmo visibility. |
| Simulate | Play / Pause | Available for the MuJoCo simulation. |
| Simulate | Go to Start | Available through simulation reset. |
| Simulate | Record / Stop simulation | Available through simulation take recording, distinct from live video recording. |
| Simulate | Playback speed: 0.25 / 0.5 / 1 / 2 | Available through simulation speed. |
| Simulate | Physics adapter | Native dispatch is `panel:Control`, opening actuator and equality controls; it is **not a backend selector**. Electron has actuator controls, but Control parity is partial; see below. |
| Simulate | Physics Options | Available through the Environment inspector. This is distinct from the Control panel. Environment property coverage is a separate physics-domain contract. |
| Window | Show / Hide left and right panels | Available through sidebar and inspector visibility. |
| Window | Panels: Scene / Pose / Control / Assets / Cameras / Layers / Output | Corresponding panels exist. Pose is labeled Joints; Control is labeled Actuators and is only partially implemented. Project is an Electron addition. |
| Window | Panels: Sensors / Statistics / plugin-provided panels | Missing. FPS and render counters in the status bar do not replace the Statistics panel. |
| Window | Panels: Inspector / Timeline | Available; the timeline supports scene-state keyframes, recorded takes, and snapshots. |
| Window | Inspector position: Float / Dock right / Group with Scene | Missing docking and floating positions. The resizable right inspector is not a docking system. |
| Window | Save / Restore / Reset Layout | Missing named layout transactions. Persisting panel dimensions is not equivalent. |
| Window | Show Viewport tab | Missing; Electron has a fixed central viewport. |
| Window | Settings | Available. This does not establish that every native setting has an Electron counterpart. |
| Window | Physics Options | Available through Environment; the repeated entry also exists in the native inventory. |
| Help | Documentation / Interaction Guide / About | Available as distinct actions. Documentation opens the same repository README. Guide and About are Electron content. |

Native contextual menus additionally include camera-panel navigation, Solid /
Wireframe shading, previous/next frame, and loop playback. Electron has camera
navigation and history stepping, and the timeline has loop playback; it does not
currently expose the native Wireframe shading mode. History stepping also appears
in Electron's Simulate menu even though native places it in the playback context.

The shared catalog preserves the top-level File, Edit, Add, View, Simulate, Window,
Help order. Supported commands must retain their native semantic groups and
relative order. Platform roles and explicit Electron additions are exceptions,
not evidence of complete native menu parity. Native OS text-editing/window roles
and the macOS Help search field are supplied by the platform.

The retained panel order is Scene, Joints, Actuators, Project, Assets, Cameras,
Layers, Output, followed by a separator and Inspector, Timeline. Project is an
explicit Electron addition between the native Control and Assets entries. The
Joint viewport tool is also an Electron addition to native Select / Move /
Rotate / Scale; it does not replace any of those tools.

Samples belong to the asset browser. They are not presented as native File menu
commands. Shared labels use Camera consistently in English and 相机 in Chinese.
Open Scene uses Command/Ctrl+O; Save Scene As uses Command/Ctrl+Shift+S.

## Functional gaps and scope

The native `ControlPanel` (`python/ui/panels/control.py`) provides actuator input,
equality enable/disable, search and sorting, and copying control/activation values.
Electron's `physics/ActuatorControls.tsx` writes actuator control values, supports
joint navigation and clearing an input, but has no equality controls or Control
search, sorting, or control/activation copying. Its URDF transmission cards show
metadata; they are not a simulation control implementation.

The smallest remaining viewer improvements are source-aware Reload, New Scene and
in-place Save with proper document transactions, then Control search/sort/copy and
equality toggles at the physics command boundary. Wireframe mode and the absent
capture surfaces are independent follow-up capabilities. Multi-model composition,
resource-root management, sensor plots, statistics, and docking/layout workflows
are larger missing features. They must remain documented as gaps rather than
being silently described as deferred by the viewer scope.

The adopted viewer scope intentionally defers detailed per-body physics-property
authoring and joint/topology construction. Site insertion and the native
Plane/Capsule physics-topology insertion paths depend on that authoring scope; ordinary visual
primitives do not. Environment simulation parameters, model interaction,
actuator input, asset browsing/previews, and timeline replay remain in scope.
Plugin-dependent MuJoCo models are outside the current WASM compatibility target.
This boundary does not excuse asset, material, environment, or replay failures in
otherwise supported models.

Unavailable features are omitted instead of being relabeled as unrelated
commands. In particular, a timeline export must not stand in for live recording,
and the native Physics adapter entry must not be mistaken for a backend picker.

## Availability and input ownership

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
These tests verify the shared menu contract and interactions. They do not verify
all the domain behavior in this inventory or establish complete native parity.
