# Electron UI contract

The adopted Python design specification remains the palette and spacing authority. Generated
`design-tokens.css`, bundled MiSans, and icon masters are synchronized by `scripts/sync-design.mjs`.
The Electron-specific layout and interaction rules live here; they do not change native defaults.

- Use MiSans for UI and JetBrains Mono for numeric fields, time values, and keyboard badges.
- Mojive G2 rectangle smoothing at 60% and native G3 capsule smoothing at 38.2% are the editor UI defaults. Keep authored radii unchanged. Derive child corners from their
  actual inset: `inner = max(0, outer - inset)` (the tool palette uses 8 outer / 4 inset / 4 inner).
- Numeric fields default to scrubbing. Double-click or Tab enters full-precision text editing;
  unchanged display, focus and no-op pointer clicks must not round stored values or add history.
- Render only ruler ticks and keyframes intersecting the visible time window, retaining a held
  marker until release. Dense marker hit widths must not capture the neighboring marker center.
- Labels and keyframe lanes share one vertical timeline scroll owner. Keep their headers sticky;
  horizontal navigation may move tracks under the pinned label column.
- Hierarchy indentation expresses link ownership; a joint wrapper does not consume an additional
  depth step. Double-click focuses without selecting browser text.
- Scene selection, transforms, animation and history remain usable with no physics adapter.
- Click activates the application menu bar; hovering another trigger then switches the open menu.
  Escape, an outside click, a second click on the active trigger, or an action ends that mode.
  Each menu uses its own trigger for placement and focus return; Left/Right switches groups.
- Scene-tree right-click and Shift+F10 target the same row and expose focus, copy name,
  visibility, viewport picking and deletion through existing scene/history operations.
- Resolve keyboard labels and command modifiers from the desktop preload platform. Windows/Linux
  use Ctrl, Alt, Shift and Delete; macOS uses Command/Option glyphs. Menus, timeline actions and
  the interaction guide share the formatter. Browser previews use the browser host platform.
- The interaction guide uses native status-mouse masters and separate keycaps, with short action
  captions and context notes. Follow the native primary-input rows; show one primary mouse
  chord per action and current keycaps. Keep alternatives in Settings and contextual control
  hints. Wrap complete gestures together and show only implemented bindings.
  Pressed mouse fills remain opaque. The divider stays inside the pressed half so its visible
  edge ends at the x=12 midline; do not let stroke thickness enlarge that half.
- Renderer hints use `TooltipHost` and `data-tooltip`, not native DOM `title` bubbles. Keep accessible
  labels independent of visual hints. Shortcut hints share `InputHints` with the guide: body-size hint
  text, 14px keycaps, 4px keycap radius, 6px inset and 10px outer radius. Plain hints use 4px corners.
  The keycap height follows larger body text so equal insets and the derived corner remain valid.
  Delay hover by 450ms; dismiss on input, scrolling, blur or owner removal. Popovers inherit the
  owner's theme, zoom and modal dialog, and remain within the viewport.
- Output severity glyphs scale with the output font and center on first-line ink metrics,
  including the localized fallback font. Multiline messages retain first-line alignment;
  fixed pixel offsets must not drift when font family, size or UI scale changes.
- Interaction settings record actual keyboard chords and mouse-button/wheel gestures. Display
  recorded values as keycaps and mouse glyphs. Only named navigation presets use a choice list;
  bindings never require scrolling a key list or typing a gesture grammar. Keep reset icon-only
  at the heading's trailing edge. Avoid permanent explanatory paragraphs below controls; use
  contextual tooltips and transient validation feedback at the normal body size.
- Native camera movement uses W/S forward/back, A/D left/right and Q/E world-up/down at
  `distance × 1.6` units per second. The focused viewport owns held movement; text, dialogs,
  menus, gizmo drags and precise editing take priority. Camera mouse navigation can accompany
  held fly keys, with one transaction lasting until both input owners release. Release, Escape, blur and hidden
  windows stop movement. Scene-camera View stays locked; Pilot edits one cancelable transaction.
- Default tool keys match native Mojive: V selects, G translates and R rotates. Scale has no
  default key and remains available in the palette. Fly and tool actions share one exclusive
  map, including conflict swaps in Settings. Old Q/W/E/R factory preferences migrate; explicit
  custom keys remain intact and conflicting new movement defaults stay unbound.
- Persist one input map for dispatch, tool hints, menus, status hints and the interaction guide.
  Assigning an occupied keyboard chord swaps its existing action; conflicting mouse gestures
  leave the map unchanged. Focused recording fields consume their input before editor commands.
  Escape cancels recording, blur exits, and a clear icon unbinds the action.
  Reject application/window accelerators such as Alt+F4 and macOS Close/Hide/Quit when recording
  or restoring bindings; native menu roles retain ownership of those combinations.
- Light activation has one Inspector control: Light source > Enabled. The tree eye edits the
  same visibility state. Do not repeat it as Display > Visible. Camera visibility and preview
  remain separate controls because they have independent effects.
- Section title hit areas use 4px corners and 8px horizontal text padding. Extend their hover
  surface into the section gutter to preserve the common label edge. Reset/disclosure controls
  remain independent targets. Inspect the complete title under hover, not only its icon.
- Floating menus and choices use one subtle neutral edge and one broad theme-derived shadow.
  Do not stack opaque dark outline rings or a second tight shadow. Color comparison swatches
  are 20px inside 28px controls with equal 4px leading/top/bottom clearances. The color editor
  uses a 16px shell, 8px inset and 8px corner-adjacent comparison/footer controls; comparison
  swatches derive 4px inner corners. Interior channel, hex and segmented fields retain 4px corners.
  The standalone trigger follows 8px outer / 4px inset / 4px inner geometry as well.
  Verify all three layers in the actual Inspector: component-specific selectors must survive
  the later generic button stylesheet. A standalone fixture with a different CSS order is
  insufficient. Opaque swatches use one fill; transparent swatches flatten their color into
  the two checker colors before painting one rounded gradient, preventing antialias fringes.
  XYZ fields and connected geometry/appearance/projection/color-format choices paint through
  their outer segments. Their frames remain transparent and do not clip the same corners
  again. Use opaque theme fills for each segment and keep internal joins square.
  RGB/RGBA reuse the XYZ joined badge/value component. Keep every channel on one horizontal
  row in both RGB 255 and RGB 1; alpha adds a fourth field and a wider popup. Channel badges
  reset to the color transaction's original value without starting a separate undo entry.
- Open/closed menus, focused fields and held pointer gestures retain native DOM ownership.
- Viewport shapes are separate from the UI corner system. Preserve production contour sampling,
  angular opacity, hit thresholds, pointer release/cancel, color, and painter order.

Review at 1540×980, 1120×800 and 125% scale. Inspect field contents, both geometry layers,
end-on gizmos, selected silhouettes touching viewport boundaries, and the next gesture after
Escape. `make ui-review` captures the actual Electron renderer and exercises the production UI.
`node scripts/bundle-smoke.mjs` separately verifies the final packaged application.

## Workspace review checklist

- Use the native semantic icon master for each action. Unknown icon IDs fail visibly during
  development; never substitute a generic cube for an unavailable action glyph.
- Scene filters and search use the same node identities as picking and Inspector. Flat filtered
  rows have no expander. Only rows with reachable children expose expansion.
- Geometry display presents Default, Visual, Collision and Both as one group at the top of the viewport. Default preserves authored roles; it is not an alias for an arbitrary preset.
- Do not add branding or explanatory captions to occupy spare workspace space. Put details in
  tooltips, settings or the interaction guide; keep persistent captions meaningful to the editor.
- All panel boundaries resize with one pointer owner. Escape or cancellation restores the
  initial size, and completed widths/heights survive restart. Review at125% UI scale.
- Theme changes resolve every surface and text role, including timeline sticky labels/rulers,
  popups, held readouts and output. Inspect both complete workspace captures.
- Native macOS menus and the fallback application menu invoke the same editor operations.
  Typing, clipboard operations and text Undo retain native input ownership.
- UI/body14, title16 and numeric14 are CSS em sizes. Do not copy the ImGui font-loader
  compensation into CSS. Verify the actual loaded font faces and complete negative values.
- Output search, severity, ordering, follow and selection are functional controls; test together
  with multiline messages.

## Shared control and frame invariants

- Icon-only buttons expose the caller's semantic hover hint. Preserve an explicit title;
  otherwise the shared Button may reuse its accessible label. Never infer an action from an
  icon asset name or add redundant caption hints to icon-and-text buttons. Review actual
  consumers, since native HTML buttons do not inherit the shared Button behavior.
- `smooth-corners.ts` applies the shared G2 cubic Bézier generator to rectangular surface paint.
  Keep layout, input and overflow on the DOM owner; paint each contour once in an SVG layer.
  Derive nested contours by normal offsets from one reference. Check whole-chain footprint
  budgets and curvature extrema before fitting. Canvas readouts share the generator.
  Capsules use the native Mojive G3 180-degree end profile through the same path entry point;
  a half-height radius must not force a long capsule back to circular ends. True circles retain
  their native circle shape. General Settings exposes independent persisted 0–100% rectangle
  and capsule smoothing preferences, defaulting to 60% and 38.2%; component radii and insets
  remain design-owned.
  See the [corner guide](../../docs/how-to/ui-corners.md#electron-surfaces).
- An inset card header derives its hover radius from the actual card inset. A header touching
  the card edges inherits the outer top corners and keeps square lower joins while expanded;
  when that header is the entire collapsed card, all outer corners belong to it.
- All search fields use `SearchField`, including file-dialog filtering. It owns the 28px
  capsule, 16px leading icon, body typography and placeholder ink. Consumers set placement
  and available width only. Clearing retains input focus. The 20px clear action is circular. Field focus belongs to the
  compound surface; never draw a separate rectangle around its text input. Use a neutral focus
  boundary while preserving keyboard navigation.
- Tree labels alone consume remaining width. Disclosure slots have fixed width for both branches
  and leaves. Search/type results are flat; a placeholder cannot introduce indentation.
- A timeline key owns the entire scene state at one timestamp: transforms and all joint
  configurations. The three lanes are Keyframes, Take, and Snapshots. Keyframes are named presets:
  selecting moves only the cursor; double-click or Load applies the stored pose. Take playback
  uses recorded frames without interpolation (nearest on scrub, floor during playback).
  Local scene recording is an Electron extension; it does not claim physics simulation.
- Joint controls use distinct cards with a name/range header and a custom rail beside its numeric
  value. Ball/free coordinates fold independently. Their numeric fields share the scrub and
  double-click text interaction below, including cancellation and one Undo edit per gesture.
- Joint and actuator cards use the shared `EntityCard` shell: 12px outer corners, 8px inset,
  4px inner header corners derived from the shell radius and inset, a 28px title row and 12px content gap. Embedded cards have neither
  shadows nor a simulated bottom edge; floating menus and dialogs retain their own elevation.
  Title hover and held paint are distinct and cannot be overridden by generic button states.
  Actuator headings select the linked joint and toggle transmission details across the entire
  row; double-click focuses that joint. The compact summary keeps joint navigation and reduction
  visible. URDF metadata does not imply an actuator control channel; do not invent a drive slider.
- Use custom Select, Switch, ColorPicker and FileDialog surfaces. macOS owns the application menu;
  file selection, overwrite confirmation and unsaved-change decisions belong to the renderer.
- Keep one frame-all action in viewport chrome. Activity icons are22px, tool icons20px; avoid
  duplicating the activity glyph in its panel title. Bottom tabs have explicit hit insets and gaps.
- Settings and Layers share the same grid, view-cube and entity-helper flags. Telemetry reports
  actual renderer activity and labels CPU submission time; idle FPS is zero.

DOM measurement observers belong to the mounted section. Bind and disconnect them during the layout commit, before removed rows can deliver resize or mutation callbacks. Verify rapid selection, folding, removal and Undo through the native menu bridge as well as direct pointer events.

## Numeric input ownership

- A single left click stays in scrub mode. The entire compound field, including its unit,
  acquires pointer capture and moves focus away from any text input. Show distinct rest,
  hover and held paint immediately on press; editing begins only after 3px of horizontal motion.
  A click or subthreshold movement starts no edit transaction.
- Double-click and keyboard Tab enter native text editing with the complete stored precision.
  Set the draft synchronously before native focus/selection; deferred selection must not overwrite
  the first typed character. Enter or valid blur commits once. An unchanged edit adds no history.
  Escape, invalid blur, pointer cancellation, lost capture and window blur roll back their active
  edit, release ownership and allow the next gesture. Local editing keys cannot edit the scene.
- Scrubbing uses the caller's step, multiplied by 0.01 with Alt and by 10 with Shift; combined
  modifiers multiply together. Integer fields accumulate fractional movement before rounding
  the published value, so fine movement is not discarded event by event. Clamp the accumulator
  at limits so reversing direction responds immediately.
- The stable idle caption uses the configured precision without changing stored values. Disabled
  fields reject pointer and keyboard editing. Preserve one transaction per completed change and
  retain the native caret/selection behavior while text editing owns the field.

## State and popup acceptance

- Hidden tree entities retain their closed-eye glyph after the pointer and selection leave. Search
  results preserve the same state; keyboard focus reveals an otherwise idle visible-eye action.
- Primary buttons keep their primary semantic ink/fill on hover and press in both themes. Neutral
  hover styles cannot override the primary variant. Selected viewport tools and Snapping use
  that same solid variant through idle, hover and held states. Disabled controls do not respond to hover.
- Wide XYZ rows share one label column and three equal numeric columns. Label length and value
  changes cannot move one row's axis starts independently from its siblings.
- Settings uses one fixed viewport-clamped shell across categories with scrolling inside. Only
  one surface paints its corners, border and shadow. Switch thumbs remain circles; capsule tracks use the shared constrained corner profile.
- Menus enter the browser top layer, clamp to the viewport, and retain unclipped shadows. An 8px
  outer corner with 4px inset has 4px row corners. Equal segmented choices have zero inset/gap and
  square internal joins; they are not a second row of floating pills.
- Camera and Layers navigation are real panels. Camera view/pilot/edit/capture operate on authored
  camera identities. Layers and viewport quick controls share one state; selected camera/light
  influence helpers are independent of their always-visible entity icons.
- While looking through the selected scene camera, View and Pilot yield both drawing and
  input for that camera's transform handles. Other selected objects keep their handles;
  returning to Editor restores the camera handles without changing the display preference.
  An independent preview does not suppress editor handles.
- Export renders an isolated scene copy with the shared lighting and geometry pipeline. It must
  preserve the editor's document, selection, playhead and undo state, and dispose temporary GPU
  resources on success, cancellation or encoder failure.

- A popup close event belongs to the menu generation that opened it. Switching menus with one
  click cannot let the old popover's delayed close dismiss the new menu.
- Every modal owns keyboard focus and makes the editor inert, including informational Help/About
  views. Both renderer shortcuts and native-menu commands honor modal ownership. Backspace,
  Delete and Undo in a modal never edit the scene behind it.

- Material color, roughness and metallic controls share one label column and identical control
  edges within each material group. Resizing the Inspector cannot leave the color control at
  a different width from the numeric fields; preserve this at normal and fractional UI scale.
- Scene rows retain each entity type's semantic hue in icon, text and selection fill, in both
  themes and filtered results. A selected row does not erase the entity type with a generic
  selection color.
- Apply application zoom once. The shell's logical pixel width and height equal the viewport
  dimensions divided by UI scale; panel layout and splitter bounds use those same dimensions.
  Do not combine a percentage shrink with CSS zoom, which compensates twice.

## Workspace chrome and repeated cards

- Activity targets use equal clearance to the rail's top, bottom and side edges. Its divider is
  painted inside the rail rather than consuming content width. Verify the first and last target
  at both normal and fractional UI scale.
- Geometry choices retain equal widths, a full 12px text inset and 8px icon gap when the viewport
  has room. Remove redundant icons before compressing captions; use icon-only choices only in
  the narrow viewport fallback, retaining accessible names and all four actions.
- Camera cards use two compact rows: name/More, then projection/resolution and View/Pilot.
  Hover is inset from the card; the ordinary 12px outer radius and 8px inset derive 4px inner
  corners. Keep View, Pilot, Focus, Align, Capture and Delete reachable in narrow panels.
- Native and frontend menus preserve the same ordered semantic groups and localized nouns.
  Separators must survive rebuilding menus after language or selection changes.

## Timeline toolbar and playhead

- Keep every control directly in one horizontal toolbar: transport, current time, Record,
  Export/Stop, snapshot, keyframe capture/navigation, speed, Zoom/Fit, Follow and
  range endpoints. Export sits immediately beside Record and keeps the same target when it
  changes to Stop. Starting export reveals that Stop target within the scrolling toolbar.
  There is no generic More/settings popover for these controls.
- Use compact semantic groups and accessible hover hints. Scroll intact groups horizontally
  when the complete width does not fit; do not wrap a lone control onto another row or shrink
  numeric text. Follow always stays in the visible fixed tail. At 480 logical pixels and above,
  both range endpoints stay there too; at smaller widths, range editors join the scrolling
  controls and each complete field remains reachable.
- Range endpoints are inline numeric editors, not a dropdown summary. With a recorded take they
  edit its playback sample range; without one they edit the visible time window. Preserve those
  owners instead of inventing duplicate View/Loop ranges. Recording timestamp growth cannot
  change field or neighbor widths.
- Recording continues until manual Stop. There is no End control, automatic stop at the initial
  view boundary, fixed duration/frame-count cap or implicit reset to zero on reaching that boundary.
  The persisted `duration` is only a finite content extent: recording, seeking, view edits and
  keyframe edits extend it as needed. It does not constrain authored timestamps or Take loops.
- Content growth preserves `88 × zoom` pixels per second. Follow changes only horizontal
  position, extending the content to cover its target window before scrolling. Stop retains
  that view; the visible range does not shrink back to the final recorded sample.
- Take loop endpoints remain actual sample indices. View endpoints edit only the visible time
  window when there is no recorded Take. Range editing is disabled during recording/export.
- Timeline commands retain 28px targets. Numeric fields and connected Follow cells use the shared
  28px field height and align vertically with the commands. Verify English and Chinese in both
  themes at normal and fractional scale.
- Time and range fields reserve eight numeric characters plus input/unit insets. Keep three
  decimals while they fit, then reduce idle-caption decimals or use complete scientific notation
  within that same budget. Text editing retains the complete stored precision. Zoom reserves
  `8000.0%`; speed uses a compact multiplication field. Values never choose another field width.
- Focused toolbar controls own local editing keys; timeline edit commands operate from the
  track. Global Undo/Redo, Open, Save and Settings shortcuts still reach the application. Three
  44px tracks retain one scroll owner and real one-pixel separators. The label column ends at
  `28 + 3 × 44 = 160px`; it must not paint an extra block below the last track.
- Show metadata details only for an actual keyframe/snapshot selection or selected keyframes.
  An empty or merely active Take lane has no footer. Keep New/Clear/Delete take and selected
  sample loop/delete commands in the Take header/track context menu, with the same shortcuts
  and transaction ownership. Take name and sample count remain available in the header hint.
- The playhead uses native `theme.danger`, independently of keyframe/selection hues. Keep its
  opaque 8×9px head in the sticky ruler, and its 1px/.85-alpha line above the ruler but below the
  sticky label column. Check actual painted pixels before and after vertical scrolling; computed
  color and size alone do not reveal an opaque sibling covering the marker.

## Operation notices

- Anchor routine operation feedback to the viewport's lower-left corner with a 12px inset,
  bounded width and the normal body font. Resolve severity icons, ink and surfaces from the
  native theme. Translate the message when rendered; preserve authored entity names.
- A notice publication has its own identity, including repeated text. Expiry and dismissal carry
  that identity so an old callback cannot clear a replacement. Pause the remaining lifetime
  while the pointer or keyboard focus is inside; do not reset the timer on unrelated renders.
- Undo/Redo actions belong to one committed history entry. The document owner rechecks that
  exact entry at invocation and rejects pending edits, recording or incompatible documents.
  Starting a new edit removes the old action even if that edit later cancels. Replacing the
  document clears its notices. The presentation component never operates on history itself.
- Notices do not take focus when appearing. Their buttons own editing keys; Escape dismisses
  the notice. Keep a 28px content height after an action disappears, with an 8px vertical inset
  and ordinary nested radii of 12px outside and 4px on controls.

## Camera alignment semantics

- Align to view uses the independent `camera-align` master, not Reset. Camera, direction and
  destination view remain recognizable as one 24-unit, 1.5-stroke outline. Both Inspector and
  the camera action menu use the same generated source and semantic hover hint.
- Local Electron masters are registered in `design/icons/sources.json` and imported by the
  design sync script. They cannot replace a native identifier. Check 16/20/24-point controls,
  fractional scale, 1×/2× density and actual hover states; lens apertures remain transparent.

## Inspector and settings alignment

- Section titles take the remaining width; actions and disclosure own fixed 28px targets.
  Reset sits immediately left of disclosure with a 4px gap. Reset does not fold the section,
  and folding never resets its content. Verify hover geometry on the actual section header.
- Inspector fields retain the shared 28px base height. Leave 8px between adjacent property
  frames and between the heading target and the first row. Section padding is 12px above and
  below, with one separator between groups. Do not add heading margins, content bottom padding
  or taller invisible rows on top of these distances. Wrapped labels may increase row height.
- Inspector properties share one intrinsic label column across the active property page,
  followed by a 12px gap. Transform, camera projection, output resolution, light and read-only
  sections inherit that grid instead of measuring separate label widths. Controls share both
  edges across sections. Long captions may wrap while retaining the control budget.
- Boolean settings use one full-width switch row: label on the left, circular-thumb switch
  on the right. The entire row owns hover, press, click and keyboard focus, including its
  blank space. Use one semantic switch and one Tab stop; the visual track must not be a
  nested button. One activation produces one change and one undoable transaction. Disabled
  rows cannot toggle, and keyboard activation must not leak into viewport or timeline commands.
  Keep 8px horizontal insets inside the row's feedback surface; expand that surface into the
  surrounding inset without moving the shared text/field edges or adding horizontal scrolling.
- XYZ layout uses a fixed signed three-decimal format budget (`-999.999`) in the resolved
  numeric font, including axis badges, input padding and field gaps. Move the complete label
  above its vector when needed, then stack all three axes only when the full row cannot fit.
  Inspector ball/free joint coordinates inherit the section's measured columns, including
  vectors inside a pose wrapper. Card-specific stacked captions must stay scoped to joint cards.
  Changing values cannot change the layout. Check English/Chinese, both themes, 280/480px panes
  and 100/125/200% scale with `tests/inspector-spacing.test.mjs`.
- Scalar joints in the Inspector retain an explicit Joint angle or Joint position caption.
  Narrow panes place the complete rail/value pair below its caption; joint cards retain their
  existing compact heading context. Timeline lane labels fill their parent's content width,
  leaving its right border visible in selected and unselected states.
- Settings search belongs to a global header region spanning navigation and content, above
  their two-column layout. It stays fixed while results scroll and searches every category.
  The dialog width grows with the body font within the viewport budget, preserving complete
  appearance captions and aligned property columns when users increase text size.
  Results identify their source category and group; typing a query clears the active category
  indicator without changing navigation geometry. Clearing search restores the previous category;
  choosing a category clears the query. Search Reset affects only matching fields and is disabled
  for no results, never silently resetting hidden categories or unmatched typography settings.
- One intrinsic label column spans every section of the active category or search result,
  including the Numeric and Output Messages groups.
  Its width is the longest rendered label, followed by the adopted 12px label gap; the field
  column consumes the remainder. Do not allocate labels a percentage of the page or fit each
  row separately. A bounded label column may wrap long text to retain usable controls.
- Settings navigation reserves the same `max(120px, 8.5em)` width in both languages, with
  explicit icon/text insets; short captions must not collapse the rail. With enlarged text, omit
  redundant appearance icons before clipping Dark/Light/System names. All input and segmented
  fields share left and right edges. Narrow windows stack labels and retain full-width fields.
  Category changes retain the viewport-clamped 560×590 dialog size; only content scrolls.
  This follows `property_style.py::label_column` and the UI guide's proximity rule: unused
  label space must not make a caption appear detached from its value.
- Search capsules use one 28px shared frame and a 20px circular clear target. Vertical clearance
  is 4px; the horizontal center follows `capsuleEndPadding` (14.2453px at 38.2% smoothing),
  matching mean end clearance to the straight sides. Compound focus never adds a second outline.
- Camera handoff and object focus are separate native motion contracts. Explicit return
  restores the independent Editor state; manual interruption uses the rendered pose. Verify
  intermediate view/projection matrices, not just final screenshots. Focus bounds distinguish
  a body's own geometry, model descendants and the complete scene. Scene-camera View/Pilot
  blocks node/joint focus requests, as in the native command owner.

- Hierarchy membership is not render selection ownership. A body selection includes its own
  geometry but excludes child bodies; joints resolve the owning body. Imported model roots
  and world containers do not select a descendant mask. Explicit geom identities remain
  selectable individually. Verify these scopes independently of focus bounds and preserve
  the native depth-independent silhouette policy.

- A file picker's confirmation uses its visible location. Resolve edited paths before committing,
  even without Enter; changing directories clears the old selection. No explicit selection means
  the current directory, while a selected child means that child. Loading and invalid paths never
  commit an older directory. Keep project-folder state independent of scene resources: closing a
  project clears its explorer and search without changing the scene, dirty state or undo history.
  Load directory children on expansion; a large first subtree must never hide its root siblings.
  Search unopened directories independently and disclose bounded search results.
- Commit snapshots synchronize authored transforms before serialization, independently of the
  renderer's frame cadence. Fast text edits and keyboard confirmation must not disappear from Undo.

## Native scrollbar ownership

- Scroll containers retain browser scrolling, text selection, keyboard behavior and scrollbar
  capture. Visibility changes only thumb paint; the existing 7px Electron gutter is unchanged.
  This gutter differs from the native application's production 19px track and is not a geometry
  parity claim. Track and corner backgrounds remain transparent.
- Reveal the actual owner on scrolling, wheel input or pointer movement over its complete track.
  Both axes share visibility; only the hovered or held axis changes state color. Resolve normal,
  hover and held paint from `field_active`, `primary_dim` and `primary`, as in `theme.py`.
- The default native delay is one second and opacity is binary. The adopted Electron behavior
  expires stationary hover after the same idle delay; native `scrollbar_visibility.py` instead
  renews a stationary track hover every frame. Held input never expires. Release starts a fresh
  delay; cancellation, blur, hidden owners and unmount clear capture state.
- Delegate by actual DOM scroll owner so newly mounted panels and top-layer dialogs participate.
  Content hover alone does not reveal scrollbars. A panel splitter cannot cover a scrollbar's
  native hit region: keep its 7px hit strip on the viewport side of the preserved painted boundary.
  Verify `tests/scrollbars.test.mjs` with Playwright's `--hide-scrollbars` default disabled, including
  real native thumb drags, idle pixels, both axes, modal ownership and fractional UI scale.

## Empty states and disclosure targets

- Empty states use one concise status and, when needed, one specific next-step sentence at the
  normal readable text size. Center them within explicit symmetric insets; do not add repeated
  navigation glyphs, blank detail blocks or decorative explanations to fill space.
- Offer an action only when the current owner can perform it: Open folder for a closed project,
  Create camera for an empty camera list, or Clear search/filters for no matches. An unselected
  Inspector directs selection in the viewport or scene tree. A material-less entity explains
  that a mesh or link with visual geometry is needed; it must not invent an Assign/Create
  operation that the editor cannot execute. Empty directories need no fake creation control.
- A collapsible joint card owns one heading button spanning its icon, name, context and
  disclosure indicator. The indicator describes the state; it is not a second nested button.
  Pointer, Space and Enter activate the same toggle. A heading selection echo must not reopen
  a card the user just collapsed. Independent fields and reset actions do not toggle the card.
  Double-click focuses once after its first activation without a second collapse toggle.

## Nested surface acceptance

Measure painted border boxes after CSS zoom, including first/last rows and hover surfaces.
Insets adjacent to the same rounded corner must match before deriving its inner radius.
Menu/select popovers use 8/4/4, entity cards use 12/8/4, corner-adjacent color actions use
16/8/8, and modal forms use
16/12/4 (outer/inset/inner, CSS pixels). Color comparison swatches use 8/4/4 within their
controls. Joint and camera search clear targets are circular
with 4px vertical clearance and optical end placement from the actual G3 cap. Independent interior controls, circles and
zero-inset connected segments retain their own geometry.

`make menu-review` exercises menu activation, scene context actions and platform labels,
checks nested geometry in the production components at 100%/125% scale, and captures the
Electron desktop. Inspect hovered card headings, menu first/last rows, form actions and
clear targets under `output/electron/verification/`; arithmetic checks accompany visual review.

## Localization

The renderer uses `i18next` with `react-i18next` subscriptions. It merges the imported
`src/locales/zh.json` catalog with `src/locales/electron-zh.json` at startup. Edit Electron
wording directly in the latter; new entries do not require a separate synchronization step.
Run `python scripts/sync-localization.py` when refreshing the native catalog and palettes. Native menu
copy lives in `main/menu-en.json` and `main/menu-zh.json`, included in packaged runtime inputs.

Use `t('Material {{index}}', { index })` for dynamic captions. Store `message(key, values)`
for output and notifications, translating when displayed so existing entries change with the
language. Do not parse English prefixes or interpolate before translation. Entity names and
paths remain literal arguments; nested message arguments are explicitly marked. Translate
known enum captions and built-in scene titles without rewriting scene data. Keep source
English in copied diagnostics. Set the document language on switch and preserve the setting.

`make localization-review` checks catalog coverage, native menu parity, interpolation, persisted
language switching, existing Output messages, and Chinese Inspector/timeline captures.

## Hint and card hierarchy

- Suppress redundant hints on obvious close/search actions and fully visible captions.
  Name-only hints on tree rows, Inspector headers and cards require actual label overflow.
  Preserve additional context, keyboard bindings and disabled reasons.
- Form numeric inputs, XYZ/RGB/RGBA channels, Select triggers and segmented choices use the
  shared `--field-height`: max(28px, max(body font, numeric font) + 8px). Badges and values share
  the numeric font and size. Badge width is field height minus cap height plus character advance,
  balancing the type cell without altering individual glyphs. Hover/press retain channel hue.
- Timeline and Output toggles share the activity rail. Status feedback excludes rendering
  submission times, draw-call counts and triangle counts.
- Camera names use primary ink; projection and resolution use secondary ink at body size.
  Metadata wraps only when its measured content cannot fit. Actuator empty states describe
  the scene source, independently of its file format or physics adapter.

## Shared field inventory

| Family                 | Production consumers                                                                    | Geometry owner                                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Numeric channels       | Inspector transforms, joint cards and properties, camera vectors, RGB/RGBA picker       | `NumericChannel`: 4px outer radius, square internal seam; no consumer radius override                |
| Segmented choices      | Geometry display, Timeline Follow, Settings Appearance, Camera projection, Color format | `Segmented`: 4px outer radius, first/last cells only; one state palette and field height             |
| Search capsules        | Scene, Project, Joints, Cameras, Output, Settings, file dialog                          | `SearchField`: 28px shell, 16px icon, 20px circular clear action                                     |
| Capsule icon placement | Search, Switch/SwitchRow, scene type chips                                              | `data-capsule-layout` resolves `--capsule-end` from the painted profile before measuring descendants |

Consumer styles may set available width, responsive captions and horizontal content padding.
They do not replace shared radius, height, font or state colors. Interior fields do not inherit
popup corner radii merely because they are descendants. True corner-adjacent surfaces retain
the explicit outer/inset/inner relationship. `make corners-review` includes a production audit
of all five segmented consumers, field heights, optical centers and numeric typography.
