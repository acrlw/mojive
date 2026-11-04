# Mojive Electron

The maintained Electron desktop frontend for Mojive, built with React, TypeScript and
Three.js. Source, runtime assets, regression fixtures and build scripts live here. Generated
captures, reports and application bundles live in `../output/electron/`.

## Run

Requires Node.js 22+ and npm. Python and native renderer builds are not required.

```sh
cd electron
npm ci
npm start
```

`npm run dev` serves the renderer for browser development. `MOJIVE_DEV_URL=http://127.0.0.1:5173 npm run desktop`
uses that renderer in Electron. `npm run package` builds `../output/electron/release/Mojive.app` on macOS (arm64 or x64).
The bundle is unsigned; signing, notarization, and Windows/Linux distribution are not yet configured.
The packaged application runs without Python, Vite, or a terminal.

## Available workflows

- Open a local URDF or project directory; resolve relative and `package://` STL, OBJ and DAE resources.
- Close or replace the project folder independently of the open scene and its undo history.
- Browse and search the hierarchy; single-click to select, double-click or `F` to focus.
- Switch Default, Visual, Collision and Both geometry display; Default retains authored visibility and materials.
- Edit position, rotation, scale, material, joint positions, cameras and lights; add box, sphere,
  cylinder, capsule and ground entities.
- Browse scalar, ball and free joint controls, choose direct joints on a compound body, and inspect URDF transmission/actuator metadata.
- Open the native joint example, URDF arm, geometry comparison or scene-object gallery.
- Use native macOS menus, localized settings, Light/Dark/System appearance and resizable panels; other platforms retain the application menu in the window.
- Use three timeline lanes: named full-scene keyframes, recorded takes, and snapshots. Scrubbing a take selects recorded frames; keyframes and snapshots load explicitly. Record local edits, step frames, set a loop range, pan/zoom, and export a take to WebM.
- Disable viewport picking/highlighting per object while retaining tree editing; new ground is disabled by default.
- Use corner notices with transaction-bound Undo/Redo actions.
- Undo/redo edits; save and reopen `.mojive.json` documents, including embedded textures and animation.
- Search, filter, sort and copy Output; inspect viewport layers and real rendering statistics.
- Manage scene cameras in the navigation panel; edit projection, optics, clipping and image resolution, toggle View/Pilot, retain the editor orbit state, preview live camera images, align to the view and export PNG images.
- Switch cameras with native nonlinear motion; focus body geometry, joints and whole scenes using their respective native framing rules.
- Scrub numeric values with the left mouse button; use custom selection, color and file dialogs.

The joint example reproduces the maintained `assets/joint_gizmo.xml`: 45 entities, 19 geometries and
9 joints, including compound and precision-limit cases. These are local kinematic edits.

The left mouse button selects or orbits; right/middle drag pans; wheel zooms. `Q/W/E/R/J` select tools,
`Shift` snaps transform drags, `Esc` cancels, `K` captures a pose, and `Space` controls playback.
The orientation widget follows Mojive's native projection, axis navigation and drag behavior.
The in-app Interaction Guide lists the bindings.

## Verification and source synchronization

```sh
make check             # TypeScript, production bundle and focused automated tests
make acceptance        # Actual Electron interactions and screenshots
make viewcube-review   # Native/browser orientation-widget comparison
make project-review    # Custom folder picker, native project commands and scene preservation
make design            # Re-import current design tokens, icons and bundled UI fonts
```

See [verification](docs/verification.md) for prerequisites, focused review targets and artifact
locations. Native numerical references are maintained under `tests/`; temporary captures are
never runtime inputs.
Browser drag-and-drop supports self-contained URDFs; desktop mode supplies project
resource access. URDF floating/planar joints and mesh formats other than STL/OBJ/DAE fail explicitly.
No MuJoCo, physics simulation, force actuator control, or physical perturbation backend is connected.

See [architecture](docs/architecture.zh.md), [UI contract](docs/ui-contract.md),
[design sources](docs/design-sources.md) and [native UI source map](docs/ui-source-map.md).
Bundled font and icon licenses are retained under `public/fonts/` and `public/icons/`.

## Project layout

| Directory  | Ownership                                                          |
| ---------- | ------------------------------------------------------------------ |
| `main/`    | Electron lifecycle, sandboxed preload, menus and file access       |
| `src/`     | React controls, scene model, timeline and Three.js viewport        |
| `public/`  | Bundled demos, fonts, SVG icons and their licenses                 |
| `design/`  | Electron-specific semantic icon masters                            |
| `tests/`   | Regression tests, reviewed native references and browser harnesses |
| `scripts/` | Resource synchronization, capture, validation and packaging        |
| `docs/`    | Architecture, UI contracts and verification instructions           |

`node_modules/`, `dist/`, generated `public/menu-icons/`, and packaging staging directories
are ignored. `make clean` removes build products and `../output/electron/`, retaining source,
fixtures and installed dependencies. It also removes any application bundles and captures in
that generated output directory.
