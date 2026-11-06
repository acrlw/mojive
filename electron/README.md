# Mojive Electron

The maintained Electron desktop frontend for Mojive, built with React, TypeScript and
Three.js. Source, runtime assets, regression fixtures and build scripts live here. Generated
captures, reports and application bundles live in `../output/electron/`.

## Run

Requires Node.js 22+ and npm. Python and native renderer builds are not required.

From the repository root, `make electron` installs or refreshes dependencies when the
package manifest or lockfile changes, then builds and launches the editor. Unchanged
installations are reused. Use `make electron-install` to force a clean dependency install.

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

- Open a local URDF, MJCF, saved workspace or project directory; resolve relative resources and
  `package://` URDF meshes. Browser uploads can include the model and its resource files together.
- Close or replace the project folder independently of the open scene and its undo history.
- Browse and search the hierarchy; single-click to select, double-click or `F` to focus.
- Switch Default, Visual, Collision and Both geometry display; Default retains authored visibility and materials.
- Edit position, rotation, scale, material, joint positions, cameras and lights; add box, sphere,
  cylinder, capsule and ground entities.
- Browse scalar, ball and free joint controls, choose direct joints on a compound body, and inspect
  URDF transmission/actuator metadata. Physical models expose real actuator inputs through the same cards.
- Discover models in the Asset Browser through samples, project files, recent models and favorites.
  Preview in an independent scene and camera before explicitly opening a model.
- Run, pause, single-step and reset MJCF simulations using official MuJoCo WASM in a dedicated Worker.
  Change simulation speed; apply translation or rotation perturbations to dynamic bodies.
- Select Environment to edit timestep and gravity, or expand advanced solver and contact options.
  Model values are the defaults; simulation reset and restoring environment defaults are separate actions.
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

The default joint example loads the maintained `assets/joint_gizmo.xml` directly through MuJoCo WASM,
the same MJCF used by native `make joint-gizmo`. Its nine joints include scalar, ball, free,
compound and precision-limit cases. Startup, the example menu and asset previews use the same
loader; joint edits, simulation and history operate on engine state. The asset library also includes
live MJCF pendulum and drop examples. URDF viewing remains independent of physics.
Physical recordings, snapshots and saved workspaces retain the engine state, environment, control
inputs and model resources needed to continue simulation; recording has no preset end time.

The left mouse button selects or orbits; right/middle drag pans; wheel zooms. In the focused
viewport, `W/S` moves forward/back, `A/D` left/right and `Q/E` up/down. `V/G/R/J` selects the
selection, position, rotation and joint tools; scale is available in the tool palette. `Shift`
snaps transform drags, `Esc` cancels, `K` captures a pose, and `Space` controls playback.
Settings > Interaction records keyboard and mouse bindings directly, with conflict handling.
The orientation widget follows Mojive's native projection, axis navigation and drag behavior.
The in-app Interaction Guide lists the bindings.
Hold `Control` and drag a dynamic body with the left button for translation perturbation or the
right button for rotation perturbation. Releasing or cancelling the gesture releases the spring.

## Verification and source synchronization

```sh
make check             # TypeScript, production bundle and focused automated tests
make acceptance        # Actual Electron interactions and screenshots
make physics-review    # Real WASM, isolated preview, long recording and workspace roundtrip
make viewcube-review   # Native/browser orientation-widget comparison
make project-review    # Custom folder picker, native project commands and scene preservation
make design            # Re-import current design tokens, icons and bundled UI fonts
```

See [verification](docs/verification.md) for prerequisites, focused review targets and artifact
locations. Native numerical references are maintained under `tests/`; temporary captures are
never runtime inputs.
URDF floating/planar joints and mesh formats other than STL/OBJ/DAE fail explicitly in the
kinematic loader. The physics adapter uses pinned `@mujoco/mujoco` 3.17.0 in both Electron and
the browser; the default single-thread runtime does not require SharedArrayBuffer. Rendering uses
Three.js, not MuJoCo's native renderer. The scene bridge maps rigid primitives, meshes, heightfields,
dynamic flex/skin surfaces, RGB/RGBA surface textures, skyboxes, fog, model lights and cameras.
Base color, opacity, emission, roughness and metallic values are applied; additional texture roles
and MuJoCo's legacy specular, shininess and reflectance behavior are not reproduced. Three.js
appearance is therefore not claimed to match the native renderer exactly.

The corpus review checks model compilation, scene construction, stepping and state restoration;
it is not an exhaustive visual or timeline review of every model. Plugin-bearing entries are
excluded from that review, and some non-plugin inputs still fail engine schema or mesh validation.
The asset browser provides live 3D previews; automatic thumbnail caching is not implemented.
Model topology and per-body mass/inertia authoring are outside the viewer scope.

See [architecture](docs/architecture.zh.md), [UI contract](docs/ui-contract.md),
[design sources](docs/design-sources.md) and [native UI source map](docs/ui-source-map.md).
The [viewer and backend design](docs/viewer-backends.zh.md) describes capability and state ownership.
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
