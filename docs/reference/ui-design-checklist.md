# UI design review checklist

Use this checklist to diagnose and prevent recurring UI failure patterns. It applies to the
production workspace, shared controls, design guide, workbench, and viewport overlays.
Read only the sections relevant to the change. The [UI design rules](../guides/ui-design.md)
define the contract; the [verification matrix](../guides/testing.md#change-mapping) determines
required gates. The checks below are focused starting points, not an additional mandatory suite.

Identify the affected owner and consumers and choose representative states. For visual
acceptance, inspect the actual production result; a static review or proposal should distinguish
code findings from behavior that still needs reproduction. A symptom suggests where to
investigate; it does not prove a cause. Fix the violated constraint rather than adding a special
case to one screenshot.

## Production ownership and design tools

| Failure pattern | Review criterion | Focused evidence |
|---|---|---|
| A specimen looks correct but the running workspace differs. | Share production measurement, painting, hit geometry, and native item state. Compare the same specification, fonts, scale, and scene state. | `make ui-native-capture`; `tests/gpu/test_design_audit_draw.py` |
| The UI runs only in one development checkout. | Track and package runtime resources. Keep disposable exports under `output/`; runtime code must not depend on examples or temporary files. | `tests/test_ui_design_runtime.py`; `tests/test_layering.py`; installed-runtime checks in the development guide |
| Changing one group unexpectedly changes unrelated controls. | Resolve the nearest semantic owner, preserve explicit child overrides, and restore scope between siblings. Global settings must be visibly identified as global. | `tests/test_group_controls.py`; `tests/test_panel_groups.py`; `make ui-native-pick-check` |
| A parameter changes paint but not layout, picking, reset, or export. | All consumers resolve the same effective value. Caches include scope, revision, and typography inputs. Reset removes the override and exposes inheritance. | `tests/test_design_parameter_audit.py`; `tests/test_numeric_layout_style_cache.py`; `make ui-native-audit` |
| Tuning requires scrolling away from the result, or a preview silently replaces defaults. | Keep controls and their actual-size context visible together. Separate preview, export, and adoption; export all authored parameters. | Design guide Group laboratory; `tests/gpu/test_ui_optical.py`; `tests/gpu/test_ui_icon_alignment.py` |
| A visual migration loses actions or panel capabilities. | Compare action inventories, command routes, search, sorting, copying, and registered panels as well as appearance. Basic editor operations work with minimal adapters. | `tests/test_native_add_menu.py`; `tests/test_native_menu_routes.py`; `tests/test_panel_action_menus.py` |

## Typography, layout, and numeric stability

| Failure pattern | Review criterion | Focused evidence |
|---|---|---|
| A changing number moves neighboring text or appears to shake inside a fixed box. | Reserve stable format budgets and digit cells; anchor decimal points and unit ends. Try `51.1 → 59.0`, sign changes, unit promotion, and long values. Display formatting must preserve stored precision. | `tests/test_status_metrics.py`; `tests/test_status_phrase_geometry.py`; `tests/test_status_metric_interaction.py` |
| A fixed metric leaves excessive whitespace, or adjusting one gap moves both sides of a divider. | Distinguish text padding, label/value/unit gaps, numeric widths, and each divider-side clearance. Stable layout still needs compact, independently adjustable budgets. | `tests/test_status_metric_settings.py`; `tests/test_status_bar_layout.py` |
| XYZ/RGBA fields reflow when values change or form partial rows. | Choose layout from available width, style, and static format budgets. Use complete rows or complete stacks. Verify full-precision text editing after reflow. | `tests/test_numeric_layout_policy.py`; `tests/test_ui_design_numeric.py`; `tests/test_ui_design_color.py` |
| A Boolean row wraps despite enough space for its caption and switch. | Use the actual Boolean control's width, independent of numeric/vector minima. Check the shared grid as well as manual rows. | `tests/test_inspector_switch_layout.py`; `tests/test_switch_fit.py` |
| Body labels protrude beyond section headings. | Use one owned leading content column for headings, manual rows, tables, and checkbox grids. Keep heading hover padding separate and restore enclosing bounds. | `tests/test_native_property_style.py`; `tests/test_panel_groups.py`; guide `--capture-alignment` |
| Cards, selections, scrollbars, or controls touch container edges. | Resolve content insets, selected-row bounds, scrollbar gutters, and control padding explicitly. Inspect sibling consumers of the shared owner. | `tests/test_ui_design_components.py`; `tests/test_native_settings_layout.py`; production panel captures |
| Text scales twice, icons drift from captions, or CJK glyphs/descenders are missing. | Measure and paint with the same resolved font and baseline. Apply UI scale once and framebuffer density separately. Check painted native text after live changes, including fractional scales below 1; do not multiply resolved sizes by atlas-relative DPI again. Test real captions and fallback fonts. | `tests/gpu/test_hidpi.py`; `tests/test_ui_design_fonts.py`; `tests/test_status_phrase_geometry.py` |
| Disabled controls lose semantic color, or hover and press look identical. | Apply opacity once at its intended layer. Preserve axis meaning and readable switch thumb/track contrast through relevant states. | `tests/gpu/test_value_rails.py`; `tests/test_ui_design_controls.py` |

## Corners, menus, and floating surfaces

| Failure pattern | Review criterion | Focused evidence |
|---|---|---|
| Nested corners cease to match their spacing at fractional scale. | Derive `inner = max(0, outer - actual_inset)` after resolving native insets and quantization. Keep parent constraints local. Check painted edges as well as arithmetic. | `tests/test_native_menu_quantization.py`; `tests/test_menu_corner_offsets.py`; `tests/test_settings_corner_chain.py` |
| The navigation rail fails to match the native window edge under UI zoom. | Keep OS-facing geometry in native window points. Derive the inner corner from actual equal leading/bottom insets. Inspect the desktop window; a framebuffer does not include the compositor edge. | `tests/test_rail_control_corners.py`; actual-window procedure in the testing guide |
| A floating panel appears to have a cut corner despite correct content geometry. | Inspect native window chrome and resize decoration separately. Preserve resize, docking, and autofit behavior when correcting the painted layer. | `tests/gpu/test_native_resize.py`; floating Timeline inspection |
| Menus elide titles unnecessarily, checks move shortcuts, or columns waste space. | Measure each popup's own entries and separate check/icon/title/shortcut/arrow columns. Reserve the check column by menu inventory, not current selection. Include the final row in quantized bounds. | `tests/test_menu_columns.py`; `tests/test_native_menu_quantization.py`; `tests/test_menu_bar_interaction.py` |
| Hints have inconsistent styles or notices clip content and corners. | Use shared tooltip/notice surfaces and explicit bounds. Align wrapped file-action and pending-edit buttons to the content right edge. Verify even authored viewport spacing separately from measured glyph or scaled pixel bounds. Keep persistent status distinct from transient feedback. | `tests/test_workspace_tooltips.py`; `tests/test_status_tooltip.py`; `make ui-notice-review` |
| Dismissing a chooser loses selection, blocks the gizmo, or breaks the next edit. | Use shared floating surfaces and input ownership. Preserve joint selection on close; focus may dismiss the chooser while leaving the gizmo usable. | `tests/gpu/test_floating_surfaces.py`; `tests/test_floating_corners.py`; `make ui-floating` |

## Icon geometry and meaning

Follow the [icon authoring guide](../how-to/ui-icons.md) for the complete geometry workflow.

| Failure pattern | Review criterion | Focused evidence |
|---|---|---|
| A small icon becomes too heavy, or dots/gaps disappear. | Scale every authored dimension together on the family grid. Keep antialiasing policy separate; do not add a pixel floor to one feature. Inspect actual production sizes. | [Canonical grid contract](../how-to/ui-icons.md#author-one-canonical-grid); multi-size production galleries |
| Shafts and heads have seams, round caps protrude, or inner lobes appear. | Inspect expanded contours, tangents, and internal openings. Union compatible opaque coverage and antialias the final boundary. Correct source geometry before raster workarounds. | `tests/test_packaged_icon_meshes.py`; `tests/test_ui_design_icons.py`; junction gallery |
| Equal canvases still produce inconsistent weight or placement. | Author optical placement in the shared grid and review beside real neighbors and text. Production workspace glyphs must not normalize to individual bounds. Candidate alignment tools remain explicit comparisons. | `tests/test_ui_design_icon_roles.py`; [candidate scope](../how-to/ui-icons.md#feasibility-candidate-library) |
| Different actions look identical or familiar actions become obscure. | Reuse a master for the same meaning; distinguish camera objects/projection, snapshots/image capture, Settings/General, and Layers/content visualization. Preserve recognizable directional and joint cues. | `tests/test_ui_design_icons.py`; `tests/test_camera_primitive_icons.py`; adopted SVG manifest |
| A specimen change restores an experimental icon or misses scene helpers. | Locate the adopted master before editing. Share semantic masters across UI and scene contexts while retaining each context's size, picking, and selection behavior. | `tests/test_scene_icon_masters.py`; `make scene-entities`; workspace/candidate distinction in the icon guide |

## Input, transactions, and navigation

| Failure pattern | Review criterion | Focused evidence |
|---|---|---|
| Dismissing input also selects the scene, or a shortcut leaks through a modal. | One input owner consumes events through dismissal/release. Preserve queued positions and physical modifiers. Use the binding layer for shortcuts. Test modifier-first release and the next gesture. | `tests/gpu/test_input_ownership.py`; `tests/test_glfw_input.py`; `tests/gpu/test_input_mapping.py` |
| Status changes on every hover or navigation clears selection. | Activate panel context by click; keep navigation and selection independent. Text/modal ownership can temporarily take priority. Avoid duplicating selection information in transient notices. | `tests/gpu/test_ui_interaction.py`; `tests/test_viewport_widgets.py` |
| A clamped drag ends early, reversal lags, or one edit creates multiple undo entries. | Keep the gesture active until release/cancel even at limits. Commit once through Session; Escape restores gesture-start state. Verify cancellation and subsequent editing. | `tests/test_gizmo.py`; `tests/test_ui_design_numeric.py`; `tests/gpu/test_ui_design_gizmos.py` |
| Timeline gestures conflate modes, overflow, or misrepresent deleted sample gaps. | Keep simulation/replay and range/Loop distinct. Bound view ranges; share one time mapping for samples, cells, endpoints, and playhead. Basic local gestures work without physics or takes. | `tests/test_ui_design_timeline.py`; `tests/test_ui_design_timeline_details.py`; `tests/test_native_timeline_edits.py` |
| Focus misses moving entities or a preview jumps away from its drag pointer. | Resolve current world transforms and camera-mode intent. Keep smooth focus and readable joint ticks. Preserve the initial pointer offset and all preview margins. | `tests/test_joint_focus.py`; `tests/gpu/test_joint_focus_readability.py`; `tests/gpu/test_ui_design_camera.py` |

## Presentation, performance, and coverage

| Failure pattern | Review criterion | Focused evidence |
|---|---|---|
| Startup exposes blank or unresolved docking frames. | Reveal only after a successfully submitted frame and resolved docking. Reuse normal preparation frames, without hidden physics ticks or fixed sleeps. | `tests/test_startup_initialization.py`; `tests/gpu/test_startup_presentation.py`; `make startup-profile` |
| Resizing produces black margins until dragging stops. | Derive render extent and projection from current layout. Keep a valid target on allocation failure; preserve intentional fixed-output letterboxing. | `tests/test_viewport_resize.py`; `tests/gpu/test_viewport_resize.py`; `make viewport-resize` |
| Repeated selection/tab/disclosure changes get slower, or optimization changes appearance. | Profile the production workload and recovery to idle. Separate UI/render/init cost; bound caches and invalidate explicitly. Preserve geometry, typography, hit bounds, and transactions. | `tests/test_property_diagnostics.py`; `tests/test_inspector_profile.py`; `tests/test_numeric_paint_visibility.py`; `make inspector-profile` |
| A capture omits diagnostics or differs from the actual editor. | Select scene/viewport/window scope explicitly. Match the specification, fonts, scale, density, scene state, and viewport size; inspect results before replacing maintained media. | `tests/gpu/test_scene_capture.py`; `make readme-media`; [capture scope](../guides/ui-design.md#interaction-and-continuity) |
| Documentation text is too small, or scaling hides Timeline tracks. | Check readability at the page's display width. Resize dock splits after choosing UI scale so wrapped toolbars and all showcased tracks remain contained; inspect whether the resulting viewport height changes tool orientation. | `make readme-media`; [README media](../guides/development.md#readme-media) |
| UI cleanup moves fields into unrelated Inspector tabs. | Keep geometry in Properties, contact/mass in Physics, and appearance in Material. Shared presentation does not merge domain ownership. | `tests/gpu/test_native_workspace.py`; `make ui-inspector-tabs` |
| A shared fix covers one specimen but leaves another consumer broken. | Enumerate affected consumers, inspect representative production states, and update the owning guide for any new reusable constraint. Geometry assertions complement visual and interaction review. | Root `AGENTS.md`; [visual review policy](../guides/testing.md#visual-review-and-baselines); the focused checks for the affected component |

## Keeping the checklist useful

Add an entry only for a reusable failure pattern or a changed contract. Write the symptom,
the constraint that prevents it, and a reproducible check. Link existing code, tests, or public
technical sources; keep private discussions and local incident details out of project documents.
Merge overlapping entries and remove stale checks when ownership changes. Keep current defaults
in the shared specification and specialist details in their owning guide.
