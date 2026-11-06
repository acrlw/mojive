# Electron verification

Run commands below from `electron/` unless a repository-root command is shown. Reports,
profiles, captures and recordings are written to `../output/electron/verification/`.
They are disposable evidence, not checked-in baselines or application resources.

## Prerequisites

- Node.js 22+ and npm; install the lockfile with `npm ci --no-audit --no-fund`.
- An installed Google Chrome for browser tests (`channel: 'chrome'`).
- A graphical desktop session for Electron icon generation and desktop acceptance.
- Native comparison targets additionally require the repository Python environment,
  native extension and an available OpenGL display (`make setup`).

Electron's postinstall script verifies/repairs runtime installation if the package installer
exits before its archive has finished downloading. It verifies the upstream archive checksum.
URDF viewing and editing do not require a running physics backend. The default test suite
also exercises the pinned official MuJoCo WASM package in Node and a browser Worker.

## Required checks

```sh
make check          # TypeScript, native menu assets, production bundle, serial tests
make acceptance     # The checks above plus actual Electron workflows and screenshots
```

The equivalent root entry points are `make electron-check` and
`make electron-acceptance`. The full Node suite runs serially because browser/Electron
checks share desktop and GPU resources. Preserve failure logs and inspect the corresponding
screenshots before changing a fixture or accepting a new baseline.

`npm run acceptance` reuses a completed production build when only desktop acceptance needs
to be repeated. It exercises loading, selection, transforms, joints, cameras, project access,
timeline capture, persistence, settings, and light/dark appearance in the actual application.
Inspect its images and JSON results under `output/electron/verification/` from the repository root.

## Focused checks

| Target                       | Coverage                                                                                                                                     |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `make menu-review`           | Menu hover/keyboard mode, tree context, direct key/mouse recording, shared hints, themed tooltips and nested corners                         |
| `make corners-review`        | G2 joins, analytic nested offsets, G3 capsules and optical icon placement, shared field/segment geometry, focus outlines and Canvas readouts |
| `make controls-review`       | Numeric drag/type states, custom controls and palette                                                                                        |
| `make layout-review`         | Inspector, settings, empty states, switches and responsive panels                                                                            |
| `make project-review`        | Custom file dialogs, project open/close, native menus and resources                                                                          |
| `make camera-review`         | Camera cards, transitions, focus, orbit restoration and preview                                                                              |
| `make camera-self-review`    | Actual desktop camera/gizmo visibility                                                                                                       |
| `make viewport-review`       | Joint picking, geometry, feedback and selection ownership                                                                                    |
| `make viewcube-review`       | Native/browser geometry, raster output and interaction comparison                                                                            |
| `make timeline-review`       | Three-lane state, recording, editing and unbounded navigation                                                                                |
| `make timeline-wheel-review` | Native desktop wheel normalization and cursor anchoring                                                                                      |
| `make scrollbar-review`      | Hover, held, idle and two-axis scroll ownership                                                                                              |
| `make switch-review`         | Whole-row toggles and pointer cancellation                                                                                                   |
| `make actuator-review`       | Actuator/joint cards and actual desktop captures                                                                                             |
| `make physics-review`        | Real WASM, Environment edits/reset/undo, independent asset preview, recording beyond ten seconds and physical workspace roundtrip             |

Browser harness HTML files are maintained under `tests/harness/` and served by Vite. They
exercise production components; the production bundle starts only from `index.html`.
Native geometry references under `tests/fixtures/` and `tests/*-reference.json` are retained
source fixtures. Regenerate them deliberately with the relevant `scripts/export-native-*.py`
command and review the numerical/visual differences; ordinary tests must not rewrite them.

`make physics-review` writes production captures and its report to `output/electron/physics/`
from the repository root. The backend tests cover invalid-command atomicity, model defaults,
full-state restoration and continuation, resource identity, Worker lifecycle, and engine-free
viewer operation. The workspace test also checks translation and rotation perturbation ownership
under perspective and orthographic cameras. Inspect the dark/light Environment, actuator,
asset-preview and three-lane recording captures; a successful build alone is not visual acceptance.

## Packaging acceptance

On macOS arm64 or x64:

```sh
make package
make bundle-smoke
```

The root equivalent is `make electron-package electron-bundle-smoke`. Packaging copies an
explicit runtime allowlist into `../output/electron/release/Mojive.app`, checks the official
Electron archive checksum, and writes `package-manifest.json`. Source harnesses, development
dependencies, private filesystem paths and test captures are excluded. Font, icon, Electron,
Chromium and runtime dependency licenses accompany the app.

The bundle smoke target launches the packaged executable with an isolated profile and checks
sandboxing, fonts, loading, geometry modes, cameras and joint controls. It also runs the physical
workspace workflow against the packaged WASM Worker. Inspect `bundle.json` and its screenshots
in the verification directory, plus `output/electron/physics/packaged/review.json` and captures.
Packaging is local and unsigned;
notarization and other-platform distribution remain outside this configuration.

## Design synchronization

`make design` imports adopted tokens, semantic icons and fonts from repository sources.
Review the resulting diff and run the affected UI tests. It is intentionally separate from
installation/building so a normal build cannot silently change tracked design resources.
See [design sources](design-sources.md), [UI contract](ui-contract.md),
[timeline contract](timeline-parity.md) and [viewport contract](viewport-parity.md).
