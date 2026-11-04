# Electron UI scope and native reference audit

This inventory compares the Electron editor against current production Python UI. It does
not treat retained design experiments as adopted behavior. Current references:

- `python/design/resources/design-spec.json`, `themes.json`
- `python/ui/workspace/settings_panel.py`, `settings_layout.py`, `typography.py`
- `python/ui/panels/settings.py`, `settings_pages/{catalog,camera,interaction,rendering,recording,visuals}.py`
- `python/ui/workspace/output_toolbar.py`, `output_panel.py`, `python/ui/panels/output.py`
- `python/ui/localization.py`, `python/ui/workspace/localization.py`

## Typography and palette

The native adopted specification uses 14 points for UI/body and output, 16 for titles and
numeric values. The actual `input_number` painter resolves text RGBA(230,232,235,255) in
the dark theme; `property_style.readonly_value` resolves secondary RGBA(169,175,183,255).
The current Electron correction intentionally uses 14-pixel default numeric text and native
secondary ink, rather than claiming that the native editable field already used muted ink.
Output remains 14 points. Numeric and output roles
independently choose the body or monospace family and design-default or even sizes from
10 through 22. Electron uses the requested MiSans body and JetBrains Mono monospace fonts.
MiSans CSS uses its normal em sizing; the native ImGui loader's 1326/1000 size conversion
compensates for its own font metric convention and must not be applied a second time in CSS.

The exact native dark/light role palettes are copied into `src/locales/themes.json`.
`applySettings` publishes every role as `--ui-<role>` and maintains the existing token aliases.
System appearance follows `prefers-color-scheme` without changing geometry or typography.
UI scale is applied by the owning shell; controls must not apply a second scale.

Shared buttons and surfaces currently use ordinary CSS circular corners. Their resolved
border radii preserve layout, padding, shadows, keyboard focus and native pointer ownership
without resize observers or contour clip masks. Unknown icon names throw instead of silently using a
cube glyph; each rendered icon exposes its semantic master in `data-icon`.

The settings surface uses outer 16 / inset 12 / inner 4, 28-point controls and an independent
12-point label/control gap. Its category rail reserves `max(120px, 8.5em)` in either language;
the shared label column follows measured content widths and controls consume the remainder.
Search spans that column; field and segmented-control right edges align. The dialog stays at 560×590 points across all
categories, clamps to the viewport, and scrolls its content internally.
The category rail stacks on narrow windows. Segments have equal-width connected cells,
no inset or gap, outer corner inheritance and square interior joins. Inspector sections share
one intrinsic property-page label column, so projection and output-resolution fields have the
same left and right edges. Vector rows reserve a stable signed numeric format budget before
placing complete XYZ rows beside their labels; narrow groups reflow together. Boolean settings
use a full-width hover/press target with their switch right-aligned and one keyboard stop.

## Settings inventory

| Native group          | Electron implementation                                                                                                                        | Scope difference                                                                                                                                                                                                                        |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| General               | Chinese default / English, exact light/dark palettes plus system mode, UI scale, MiSans UI, independent numeric/output font families and sizes | Native deferred model rebuilding is not applicable to the direct frontend scene model. Panel visibility is owned by the main editor.                                                                                                    |
| Camera                | FOV, orbit/pan/zoom speed, focus distance margin, transition duration                                                                          | Native transition-curve choice, linear/proportional zoom mode and configurable distance bounds are not yet exposed. They require corresponding camera behavior, rather than inert settings.                                             |
| Interaction           | Translation/rotation snap steps, rotation tick scale, selection outline/highlight and outline width                                            | Native input-binding remapping, per-interaction enable switches, selection labels/frames/bounds and alternate gizmo render styles are not yet exposed. Perturbation force/torque settings require physics and are outside this version. |
| Rendering             | Grid, orientation widget, native camera/light entity icons, selected camera frustum/light range                                                | Native backend/device diagnostics, render-flag/debug-view matrix, shadow quality controls are not yet exposed. The Electron renderer is Three.js/WebGL.                                                                                 |
| Recording             | Three lanes for keyframes, full-scene takes and snapshots; local recording and WebM export                                                     | Physics simulation recording remains outside scope. The editor owns scene poses and records authored state without a physics adapter.                                                                                                   |
| Content Visualization | Default/visual/collision/both is owned by the viewport toolbar                                                                                 | Native contact, BVH and physics-specific overlays remain outside scope.                                                                                                                                                                 |

Only supported settings are displayed. All displayed values persist under
`mojive.electron.settings.v1`, validate on load, and update their owning UI/viewport behavior.
A reset affects the current category, preserving independent settings in other groups.

## Output inventory

The output toolbar retains native semantic icon masters: severity `info`/`warn`/`err`,
`search`, `copy`, `follow`, `delete`, and collapse `down`. A `sort` control adds explicit
oldest/newest ordering. Counts remain visible independently of search results. Every search
token must match the timestamp, severity or message; severity toggles compose with search.
Hiding search clears the query, matching native ownership.

Rows preserve multiline message content, a 14-point design-default output font and stable
timestamp columns. Selection supports Shift ranges, Command/Control toggles, keyboard
navigation, Select All and Copy. Copy selection yields message text; Copy shown includes
timestamp/severity. Following new output can be disabled manually and stops when the user
scrolls away. Clearing history clears selection. Native path reveal is not exposed because
it requires a constrained desktop reveal API; arbitrary text is not treated as a trusted path.

## Localization and verification

`src/locales/zh.json` combines the complete native Chinese catalogs with explicit Electron
captions in `electron-zh.json`. Regenerate catalog and palette copies with
`python electron/scripts/sync-localization.py`. `t` is available to shared controls,
and the language subscription rerenders settings and output immediately.

`node --test tests/settings-output.test.mjs` runs real browser checks for sanitized persistence,
default Chinese, switching language, exact native light colors, independent font roles,
Chinese search terms while the UI is English, live numeric updates, output search/severity,
sort order, range selection, copy formatting and clearing. The main editor acceptance remains
responsible for integration and panel-layout captures. A 200% shell-scale check verifies the settings window and close
action remain within the viewport, and a wheel-driven history check verifies manual browsing
suspends output following.

## Native shared controls

`components.tsx` keeps the native drag-field ownership contract from
`workspace/numeric_fields.py`: pointer capture for scrubbing, double-click/keyboard text entry,
full stored precision while editing, stable three-decimal inactive captions, normalized
negative zero, one begin/end transaction, and Escape rollback. Axis reset badges use the
native 18-point width and the exact axis palette from `ui/theme.py`, mixed with field ink at
56%, 72% and 88% for idle/hover/held. Units remain in the label in parentheses.

`CustomControls.tsx` replaces platform selects and checkboxes with keyboard-accessible
choice popovers and switches. The adopted specification defines 54×24 switches with an
18-point neutral thumb; `interaction_paint.py` supplies track feedback colors. Choice
popovers use outer 8 / inset 4 / inner 4, and rendered borders are inset ink rather than extra
layout pixels. The settings dialog uses outer 16 / inset 12 / inner 4. Search controls are
capsules, their clear actions are circles, and mouse/keyboard input focus belongs to a
neutral compound contour with no inner green rectangle.

`ColorPicker.tsx` follows `workspace/color_editor.py`: Original/Current comparison,
Copy/Paste, SV plane with 14-point hue/alpha bars, Hex/RGB255/RGB1 entry, original-color
restore, and a single popup transaction with Cancel/Done or Escape. Its outer 12 / inset 8 / inner 4
geometry is derived from actual padding. Explicit clipboard actions use the Electron
preload bridge; the browser preview uses the browser clipboard API. The parser preserves
the native hexadecimal, byte, normalized and CSS-style alpha grammar.

The existing `Squircle` surface and `useSquircleSurfaces` names remain API-compatible, but
the current policy uses ordinary CSS border radii and no global corner observer. Canvas
readout backgrounds also use circular round rectangles. True circles and capsules retain
round caps. Spatial gizmo G3 geometry remains independently owned and unchanged.

`tests/shared-controls.test.mjs` verifies real pointer scrubbing, precise entry, invalid
entry rollback, captured-drag cancellation, axis reset, keyboard choice/switch controls,
color preview and one-transaction cancellation/commit. `tests/color-picker-desktop.test.mjs`
verifies Copy/Paste through the real Electron preload/IPC route with an isolated clipboard
sink, plus circular corners and uncut overflow. The settings browser test covers
actual mouse focus and a custom choice popup inside the 200% modal. Reproduce screenshots
with `node scripts/review-editor-panels.mjs`; inspect `shared-color-picker.png`,
`shared-select.png`, `settings-search-mouse-focus.png` and the dark/light settings captures.

## Confirmation and final control contracts

The unsaved-scene confirmation uses one complete localized sentence with a filename
placeholder, a 16-point heading, 14-point supporting text, and equal-width Cancel / Don't
Save / Save actions. The parent owns saving and continuing the pending open/close command;
a failed or cancelled save retains the confirmation. The custom HTML alert dialog owns
focus trapping, Escape / Command-Period, pending-operation input and safe initial Cancel
focus. There is no repeated Save icon or button-explaining helper text.

[Apple's alert guidance](https://developer.apple.com/design/human-interface-guidelines/alerts)
supports concise situation-specific copy, useful complete-sentence information, explicit
action verbs, no more than three choices and an alternative cancellation gesture. Cancel
initial focus is this editor's requested safe-action choice, not a claim about Apple's
recommendation for assigning a default alert button. Mojive's modal 12-point inset and
outer 16 / inner 4 corners come from the local design guide, not Apple.
[Refactoring UI's published guidance](https://refactoringui.com/) supports separating
content with spacing and tonal contrast instead of adding redundant borders; the
confirmation uses a single surface and a clear committing action under that principle.

Switch tracks and thumbs have an explicit round semantic shape. UI surface
styling must never turn a switch thumb into a rounded rectangle, including when a previously hidden
Settings category becomes visible. The Output search belongs between the severity filters
and actions in one toolbar row, wrapping only when the panel cannot fit that row. Every
Output toolbar glyph uses the 20-point native master at 28-point hit height; inline message
glyphs use 16 points. Tests exercise actual Settings in both themes, toolbar wide/narrow
layout, confirmation keyboard ownership, failed/pending saves, 200% scale and numeric
scrubbing after a search field held text focus.

Dialog surfaces have one corner painter: a transparent, square HTML dialog hosts one
circular-corner surface with its own inset contour and shadow. A second mask or opaque
wrapper must not cut that contour or leave mismatched corner blocks.
Native CSS rounded menus preserve overflowing shadows; `AppMenu.tsx` owns outer 8 / inset 4 /
row 4 geometry and positions the popover in the top layer.

Semantic committing buttons use the native `primary_action_ink` palette in every consumer.
The shared Button maps explicit primary/secondary variants and the existing primary-button
class to one state owner. Generic hover rules must not replace primary ink with a neutral
field. Dark primary rest/hover/held are #9CBF8D / #67875A / #B8D2AC; light states are
#4C7040 / #45683A / #3E6135. Foreground remains `on_primary`; disabled controls keep their
semantic hue at reduced opacity. Secondary states follow `secondary_action_ink`, including
its light-theme 12% text blend. `tests/button-palette.test.mjs` verifies source-derived values.
`review-production-controls.mjs` captures actual Electron Settings and FileDialog states in
both themes, without writing a scene file; evidence is in `../output/electron/verification/app-settings-*`,
`../output/electron/verification/app-save-*`, and `../output/electron/verification/production-controls-review.json`.

## Shared interaction review

| Failure pattern                                                  | Current invariant                                                                                                                 | Production evidence                                                                 |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Scrollbars remain visible or lose drag feedback.                 | One-second activity expiry, separate theme states and held protection; transparent track keeps native scrolling and hit geometry. | `tests/scrollbars.test.mjs`, `make scrollbar-review`                                |
| Resizing a pane steals scrollbar drags.                          | Resize hit strips stay on the viewport side of the painted boundary.                                                              | Actual thumb drag and splitter/cancel checks at 100/125/200% scale                  |
| A disclosure requires clicking a small arrow.                    | The complete joint heading is one keyboard-accessible toggle; independent edits do not collapse it.                               | `tests/joint-heading.test.mjs`                                                      |
| Actuators diverge from joint cards or appear raised. | Joint and actuator panels share `EntityCard` geometry and state paint; embedded cards have no shadow or bottom edge. Transmission details retain full metadata without adding a fake drive control. | `make actuator-review`, `../output/electron/verification/actuator-cards-review.json` |
| Empty text appears detached or duplicates the navigation glyph.  | Center captions and wrapped descriptions within symmetric insets; omit repeated panel illustrations.                              | `../output/electron/verification/empty-states.json`, `empty-material-zh.png`, `empty-actuators-zh.png` |
| Opening a project loses the selected root or cannot be reversed. | Confirm the visible directory, lazy-load children and expose Close Project independently of scene state.                          | `make project-review`                                                               |
| Recording stops at the initial ruler boundary. | Recording ends only on manual Stop; the displayed extent grows without changing pixel scale or the cursor. Undo preserves enough extent for the retained cursor and later captures. | `make timeline-review` |
| A near-complete hinge range becomes a filled disk. | Preserve native indexed ribbon topology and its empty interior; only folded offset triangles invoke exterior repair. | `tests/gizmo-ribbon.test.mjs`, native arc-ribbon fixtures |
| An active scene camera shows or picks its own transform handles. | View and Pilot yield both drawing and input for the selected active camera; other object handles and editor-camera restoration remain functional. | `make camera-review`, `make camera-self-review` |
