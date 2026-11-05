# Viewport implementation map

This implementation uses Three.js for scene meshes and the adopted native **flat**
gizmo style for editor overlays. It does not use Three.js TransformControls or its
OutlinePass. Scene materials use Three.js PBR lighting; the native lighting renderer
and optional native solid gizmo style are not selected by this application.

| Contract                       | Native source                                                                                    | Electron implementation                                                                                                                                                                                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Selected albedo/emission       | `python/render/opengl/shaders/scene_body.glsl`, `passes/opaque.py`                               | `pipeline.ts`: albedo mix after texture/color, `(1,.82,.45)`, `.35`; additive `.16 * albedo`                                                                                                                                                                                            |
| Selection mask/outline         | `passes/outline.py`, `shaders/outline.frag`                                                      | `shaders.ts` contains the original shader body, including diagonal connected border closure and complementary AA coverage; silhouette mask deliberately has no depth test, matching native                                                                                              |
| Both geometry comparison       | `passes/comparison.py`, `shaders/comparison.frag`, `types.py`                                    | Two independent depth-writing nearest-role targets, no transparent overdraw, display-space `.65/.35` weighted sum; collision shells do not enter the visual shadow pass                                                                                                                 |
| Flat overlay AA                | ImGui draw-list triangle/fringe behavior                                                         | `viewcube.ts` NativeRaster uses WebGL triangles, explicit native centered/outward fringes, original-contour triangulation, and one-piece round-capped arc ribbons                                                                                                                       |
| G3 arrows, planes, scale boxes | `python/geometry2d/curves.py`, `interaction/gizmo.py`                                            | `geometry.ts` preserves the cubic curvature ramp, 8-point Gauss quadrature, `.025` chord tolerance and `.618` smoothing                                                                                                                                                                 |
| Projection/fade                | `interaction/gizmo.py`                                                                           | 88pt extent; hide `.08`, fade `.20`; `.05` hit threshold; analytic perspective half-ring extrema; far-to-near planes/axes                                                                                                                                                               |
| Transform gestures             | `ui/gizmo/dragging.py`                                                                           | Projected-axis travel; ray-plane translation/rotation; angle unwrapping; 8pt cardinal escape; near-edge linear mode; Shift 0.1m/5° snap; pointer ownership, cancel, single edit transaction                                                                                             |
| Active drag feedback           | `ui/gizmo/guides.py`, `overlay.py`, `workspace/gizmo_readout.py`, `shaders/debug_drag_link.frag` | Native clipped/center-masked translation ruler, 5° tick hierarchy, perspective radial-cut sector ribbons, neutral-basis joint sectors and original mouse-down anchoring, dashed hinge direction axis, source shader G3 connector, native numeric anchors/palette/units/snap/turn counts |
| Precise numeric entry          | `ui/gizmo/precise_input.py`, `ui/app/gizmo_input.py`                                             | Scalar-handle double-click; deferred Relative/Absolute entry; degrees/radians and U conversion; Enter commits atomically; Escape/outside cancels; body/screen rotations remain relative                                                                                                 |
| View cube                      | `ui/viewcube.py`, `ui/workspace/viewcube.py`                                                     | Source-derived six-ball layout, back fade, labels, lollipop profiles, native raster, hit priorities, 4pt click slop and origin projection toggle                                                                                                                                        |
| Camera motion                  | `ui/camera.py`, `math3d.blend_projection`                                                        | Z-up, 0.35°/px orbit, ±89.9° poles, native AABB frustum support fit, .35s axis alignment, .28s normalized perspective/orthographic matrix morph                                                                                                                                         |

`tests/gizmo-geometry.test.mjs` compares production geometry with independently
exported native coordinates. `tests/gizmo-interactions.test.mjs` executes the
production browser gizmo through held input, snapping, cancellation and oblique
rotation. `tests/viewport-pipeline.test.mjs` uses a real WebGL context to verify
four-edge outline closure, visibility ownership and Both order/overdraw invariance.
`tests/gizmo-precise.test.mjs` mounts the real Viewport and checks pointer double-click,
keyboard submit/cancel, mode conversion and degree/radian conversion.
`tests/drag-guides.test.mjs` compares 52 native projected/clipped/faded ruler, sector, radial-cut, endpoint-cap and hinge-arrow fixtures; `tests/gizmo-feedback.test.mjs` captures held drags, snapped feedback and oblique/edge views and verifies the connector shader equations remain verbatim. Representative held-input captures and view-cube raster comparisons are recorded under `../output/electron/verification/`.

Physics perturbation remains capability-gated: this URDF editor has no physics
backend and cannot apply physical translation/rotation perturbations. Control-drag
is therefore reserved instead of performing an unrelated scene edit. URDF joint
manipulation is kinematic.

`tests/gizmo-configuration.test.mjs` drives the original `assets/joint_gizmo.xml`
through ball/free quaternion and free/multi-joint position edits, custom latched snap
steps, cancellation, and authored-frame preservation. Held-input captures are
`../output/electron/verification/native-joint-{ball,free,free-rotation,multi}-drag.png`. Camera controls
and actual display-setting ownership are checked in the mounted Viewport tests.

The Electron Scale tool edits object scale as requested by the Inspector contract.
It is not the native primitive-specific Dimensions editor (which couples radii
and shape parameters); position, rotation, and joint interaction use the native
flat contracts mapped above.

Default geometry mode preserves authored role-group visibility and original
materials; explicit modes override only presentation filters and restore them
after rendering. Camera projection is controlled and idempotent; imperative
menu requests consume the latest controlled value before the next render. Snap
steps are shared by scalar joints, planar/axial motion, trackball rotation,
precision rails, guide ticks, and labels. Snapping retains NumPy nearest-even
ties. Axial world-per-point conversion is frozen at mouse-down, and slides retain
the native 8pt cardinal escape and limit-overtravel rebasing.

Readout numeric fonts and surfaces follow current settings; light-theme marker
ink uses the native 3:1 contrast correction. Three.js scene lighting and canvas
text rasterization are not pixel-identical to the native renderer/ImGui fonts.
Shader equations and native overlay geometry are independently source-checked;
this is not a claim that every rendered scene pixel matches OpenGL.

Viewport content layers follow the native `ui/layers.py` Editing aids and
Selection feedback groups. The Electron panel also exposes its supported scene
geometry, grid and view-cube channels; it does not advertise absent physics/debug
publishers. Layers affect render passes, helper picking and gizmo input, without
changing authored object visibility. Object `gizmoEnabled` inherits the nearest
authored ancestor flag independently of the global gizmo layer.

`ViewportStats` reports measured CPU frame/submission wall time and Three.js
scene/compositor/helper pass counters. FPS is the actual number of rendered
frames in the trailing second; a render-on-demand viewport reports idle and
0 FPS after 500 ms without a frame. UI raster canvases use separate contexts
and are excluded from draw-call/triangle counters. Sampling is throttled to
4 Hz and does not schedule rendering. `viewport-metrics.test.mjs` and mounted
Viewport tests check idle behavior and actual layer visibility;
`viewport-layers.test.mjs` checks shared switches, reset, focus ownership and
narrow-view scrolling.

The trackball fill uses the native complete 64-segment disk. Its readout preserves
`dragging.py`'s two signed screen-axis angles on one line; it is not an Euler XYZ
readout. Canvas numeric surfaces and ordinary DOM popovers use the current shared
G2 cubic UI corner policy, including readouts, precise inputs, and the joint
chooser. Canvas surfaces use the same path generator as DOM surface paint. Segments still divide one zero-padding, zero-gap frame and keep square
internal corners. World-space gizmo silhouettes and native G3 arrow/ring geometry
retain their independently specified curves.

The precise editor inherits the current shared palette for idle, hovered, selected
and focused states, including light mode. Its numeric field follows native
`borderless_numeric_input`; shortcuts remain Enter/Escape/U, exposed in the field's
hover hint rather than a permanent small-text footer. The compact joint precision
rail also resolves field and primary colors from the current theme. Viewport
readouts observe theme changes even when system appearance changes without a new
scene or settings object.

`joint-picker.ts` ports the contextual chooser from `ui/app/gizmo_input.py`: a
multi-joint body's selection opens beside the scene click (+14pt), or at the
viewport's top-left (+18pt) when selected in a panel. Geometry selection resolves
the owning body. The chooser can be moved and closed; closing preserves the chosen
joint and suppresses reopening for that body selection. Single-click chooses;
double-click chooses, focuses, and dismisses. Selecting a concrete joint bypasses
the chooser. `gizmo-precise.test.mjs` checks all three native compound-body joints,
click/tree placement, dismissal memory, pointer drag, focus, and real rich-scene
trackball cancel in both themes. `gizmo-feedback.test.mjs` checks full disk coverage
in all eight octants and the native label. Representative captures are
`../output/electron/verification/trackball-rich-{light,dark}.png`,
`../output/electron/verification/precise-popup-{light,dark}.png`, and
`../output/electron/verification/contextual-joint-picker-click.png`.

Hinge-axis picking follows `python/ui/gizmo/core.py` and the original solid
`_hinge_axis_projection` outline in `guides.py`: the head, shaft, center and rear
dash gaps share the Rotate-Z joint handle with a 4-point halo. The full hit
polygon is retained separately from the unchanged dashed paint polygons. Axis
ownership wins over an overlapping outward endpoint tick. Axis and ring fade
independently; a hidden edge-on ring does not remove the visible axis's operation.
At a center or parallel-ray press, the native joint basis supplies the initial
radial direction and starts linear rotation without an initial angle jump.
`hinge-axis-interactions.test.mjs` exercises these cases, normal release and
cancellation in perspective and orthographic views. `drag-guides.test.mjs`
compares both paint geometry and full hit silhouettes against Python production
fixtures. Captures: `../output/electron/verification/hinge-axis-hover-{oblique,edge-on}.png`.

Scene-camera View and Pilot leave the editor orbit state independent, matching
`ui/app/navigation.py::_leave_model_camera`. Entering the first scene camera
saves the editor pivot, yaw/pitch/distance, navigation up, projection/optics and
paused transition state. Switching cameras or View/Pilot modes does not replace
that saved editor state. Explicitly returning or removing the viewed camera
restores it; Pilot writes only its authored scene camera. Manual navigation that
exits View instead adopts the currently displayed eye, pivot and projection,
matching native input ownership. The first pan uses that displayed screen basis.
`camera-navigation.test.mjs` exercises real pointer orbits and pans, chained
perspective/orthographic cameras, rolled scene-camera up vectors, top/bottom
editor views, Pilot transactions and active-camera deletion.

Scene-camera handoffs use `ui/camera.py::CameraViewTransition`: 280 ms with
`1 - (1 - t)^3`, shortest quaternion interpolation (normalized linear interpolation
above dot 0.9995), interpolated eye and orbit distance, and a target reconstructed
from the interpolated orientation. FOV, clipping planes, orthographic height,
normalized focal lengths, principal offsets and perspective/orthographic blend
follow the native equations. Destination camera poses are resolved every frame.
A subsequent switch starts from the current rendered sample; returning transitions
to the saved editor state. A real navigation gesture stops the handoff and adopts
the visible pose. Native orbit then levels its basis around world Z; a Pilot
interruption remains one undo transaction and Escape restores its authored start.

`camera-transition.test.mjs` checks 72 samples from live Python production across
rolled and calibrated cameras, opposite orientations, both mixed-projection
directions, moving destinations and interrupted transitions. View and projection
matrices match within 2e-5. `scene-camera-transitions.test.mjs` checks actual
Viewport integration at intermediate times, switching, complete editor restoration,
real pointer interruption and Pilot cancellation. Captures
`../output/electron/verification/camera-handoff-{70,140,210,280}ms.png` show the sampled progression.
Run these and the camera consumers with `make camera-review`.

Initial framing and adopting an exact focus pose do not announce an unchanged
projection mode. Imperative projection commands retain one pending notification
until the controlled prop acknowledges it, so repeated commands within one React
commit cannot emit duplicate events or let a stale mode replace the requested one.
The original strict gizmo projection test and a controlled-focus interruption/
reversal regression cover both uncontrolled and controlled consumers.

Pilot is an Electron authoring operation, using the same world-Z orbit calculation
as the editor and native `ui/camera.py::OrbitCamera`. Camera view `up` describes its
actual screen basis; it must never become a fixed orbit pole. Adopting a view derives
yaw/pitch from eye minus target, clamps pitch to +/-89.9 degrees, and uses the native
screen-up heading recovery at exact +/-Z poles. Each new Pilot gesture adopts the
current authored pose so intervening Inspector or animation edits cannot leave a
stale pivot. Entering View/Pilot alone preserves authored roll; orbit reconstructs
the level world-Z basis, while dolly retains the exact authored orientation. Escape
restores the complete pre-gesture camera and closes its authoring transaction.

`scripts/export-camera-orbit-fixtures.py` exports 37 cases directly from the live
native controller, including 30-degree roll, zero/small accumulated drags, four
headings at each exact pole and pitch clamping. The gesture suite compares real
browser pointer moves to these fixtures, and compares Editor/Pilot eye, target,
screen-up and FOV after identical multi-gesture paths and mode re-entry. Captures
`../output/electron/verification/{editor,pilot}-orbit-{oblique,top,bottom}.png` show corresponding poses.

Entity markers use the production 24-point master and 13-point pick radius from
`interaction/scene_icons.py` and `ui/scene_entities.py`, both multiplied by the
current UI scale. Display pixel ratio is applied once by the renderer. Source
ink bounds are camera21.5×17.5pt and light13.5×21.5pt inside the 24pt canvas.
`entity-markers.test.mjs` measures actual WebGL pixels at UI scales1/1.5/2 and
DPR1/2, and checks the scaled pick boundary. Evidence:
`../output/electron/verification/entity-marker-sizes.json`, `../output/electron/verification/entity-marker-scales.png`,
`../output/electron/verification/editor-camera-{before-view,after-return}.png`.

Selection ownership follows `session/core.py::selection_highlight_object_id` and
the renderer's single object-ID equality, rather than hierarchy traversal. In the
Electron scene contract each geom/primitive has an explicit render identity. A
body highlights only its own geometry; a joint resolves its authored body, excluding
child bodies. Imported model roots (`kind=robot` in this frontend) and world nodes
are hierarchy containers, so they do not highlight descendant meshes. Camera/light
selection colors remain in their own marker pass. The mask and surface-highlight
passes share `scene-selection.ts::selectionGeometry`, including inherited viewport
selection exclusions. Neither focus bounds nor the original outline shader changes.

Native adapter IDs remain adapter-owned: legacy imported GEOM nodes can have ID0,
while authored geometry can own a positive ID; native WorldInstances MODEL nodes
can own moving instance IDs even though imported MuJoCo MODEL containers do not.
The Electron adapter exposes explicit geometry identities and logical own-body
selection. `export-selection-ownership-fixtures.py` records all these distinctions
from 30 actual native selections rather than inferring ownership from node names.

`selection-ownership.test.mjs` checks semantic ownership, actual GPU pixels in all
four geometry modes, and the production hierarchy at the reported anchor/child
selection. Native outlines intentionally use a depth-independent silhouette; a
child's narrow outline may cross its occluder, but their surface highlight interiors
must not overlap. Evidence: `../output/electron/verification/selection-owner-pixels.json`,
`../output/electron/verification/selection-scope-app.json` and `selection-scope-{anchor,child,model}.png`.

`scene-selection.ts` adds the requested per-object viewport selection preference;
this is an Electron authoring extension to native `ui/app/viewport.py::_selectable`,
which currently excludes only the world/root identities. A persisted
`Object3D.userData.viewportPickable = false` disables viewport picking and the
selection fill/outline for that entire subtree. Both mesh rays and camera/light
marker picking resolve the same ancestor chain; disabled hits do not prevent
selecting eligible geometry behind them or clearing selection on empty space.
Body selection also omits disabled owned geometry from its selection mask. The
outline/highlight shader itself is unchanged. Tree selection, Inspector editing
and transform gizmos remain available, so a disabled floor can be edited without
covering the viewport in its selected surface color. `createPrimitive('ground')`
and catalog XML `type="plane"` initialize this preference from their authored type,
never from their name. Imported URDF geometry keeps its existing default.

Project and undo snapshots retain the Boolean in object metadata. `encodeDocument`
and `decodeDocument` detach serialized state because Three's JSON conversion and
ObjectLoader retain userData references; subsequent live or restored edits must
not mutate earlier history entries. `viewport-selection.test.mjs` exercises real
viewport pointer routing, all four geometry modes with pixel comparisons, helper
picking, inherited Inspector state, actual undo/redo, type-derived defaults, file
round trips and independent metadata snapshots. Run via `make viewport-review`;
representative captures are `../output/electron/verification/nonselectable-floor-{inspector,gizmo}.png`.

## Hierarchy focus and framing

Tree double-click follows `workspace/navigation.py::select_scene` and
`app/navigation.py::_apply_pending_node_focus`. `focus-bounds.ts` selects one
geom, one body's own geometry, all descendant geometry for an imported model,
or all finite scene geometry for the world. Hidden visual/collision geometry
still contributes, as in native bounds; infinite planes do not. Camera/light
attachments do not inherit body geometry. Geometry-free nodes use their world
position and a radius of 2.5% of the finite scene diagonal. Local geometry bounds
are expressed in the owning body frame, and complete transforms are composed
before taking axis-aligned extents. Electron's `robot` document root corresponds
to the native MODEL container, rather than a native ROBOT body.

`camera-focus.ts::CameraFocusTransition` ports `OrbitCamera.focus_bounds`:
the eye-to-new-center azimuth is retained, elevation is at least 30 degrees,
and four AABB support inequalities fit the retained lens and viewport aspect.
Calibrated focal/principal offsets and orthographic extent remain meaningful.
The default 0.36 s smoothstep is applied once to yaw/pitch/roll; reciprocal-scale
remapping controls translation and distance so the eye travels a straight path.
This is distinct from the 0.28 s quaternion scene-camera handoff. View-cube framing
uses native `look_from_target` with a selected bounds sphere, 1.2 padding and 0.35 s
quartic ease-out. Frame All uses native `frame_scene`: retained yaw, a 30 degree
pitch floor, sphere framing, 1.15 padding, 0.35 s quadratic ease-out and adaptive clips.

`joint-focus.ts` separately ports native hinge readability, marker clearance,
slide midpoint/range and nearby occlusion candidate selection; joint framing uses
`focus_target` spherical/log-distance travel. Its 40 live-native direction cases
avoid treating equal-score lattice ordering as a specified behavior.
`scripts/export-camera-focus-fixtures.py` supplies 138 native camera samples,
including rolled/calibrated lenses, portrait aspect, exact poles, distance limits,
all focus easings, view-cube and frame-scene paths. `export-focus-bounds-fixtures.py`
independently supplies native local/world bounds for 8 hierarchy entities.
`camera-focus.test.mjs` compares poses and full view/projection matrices;
`camera-focus-integration.test.mjs` checks actual viewport travel, interruption,
paused editor motion, joint focus and production tree double-click.
Run these checks with `make camera-review`.

Native `request_node_focus` and `request_joint_focus` refuse requests while a
scene camera is active. Electron applies the same guard to tree/node/viewport
object focus in both View and Pilot, including an unfinished camera-entry handoff.
Selection may still change, but the request does not exit that camera, interrupt
its transition or author a camera edit. Frame All and ViewCube alignment retain
their separate navigation paths; explicit return and manual navigation are unchanged.

Open circular strokes preserve `geometry2d.curves.arc_ribbon_mesh` strip indices.
Only a negatively wound offset triangle invokes the native exterior-repair branch.
In particular, an authored range such as Atelier's ±3.14159 radians is an open,
almost complete turn: overlapping end caps must not remove the ring's interior.
Both idle hinge ranges and active rotation guides use the indexed mesh plus an
AA fringe around its boundary. `export-arc-ribbon-fixtures.py` supplies 16 native
topology cases; `gizmo-ribbon.test.mjs` renders Atelier at −90.045 degrees from
front, oblique and shallow perspectives and checks the empty ring interior.

Native `_viewing_selected_camera` also yields gizmo drawing and input while the
selected authored camera is the active View or Pilot camera. The same predicate
guards Electron drawing, hover, drag acquisition and precise input. Other selected
objects retain their handles, and returning to Editor restores the camera handles.
The separate Inspector preview does not suppress editor controls. Entity icons and
influence helpers remain excluded from active scene-camera views. The mounted
viewport checks in `camera-self-gizmo.test.mjs` cover these ownership rules and
actual View/Pilot orbit gestures.
