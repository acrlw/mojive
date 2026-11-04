# Electron action icon masters

These original masters extend the adopted Mojive family for Electron-specific actions.
The shared source remains `examples/ui_design/native/svg/sources.json`; local identifiers
must not shadow its semantic masters. `node scripts/sync-design.mjs` imports both inventories,
copies the editable SVG unchanged to `public/icons`, and generates `src/icons.ts` and provenance.

All dimensions use the native 24-unit grid and 1.5-unit round stroke/caps/joins. The renderer
scales the entire canvas uniformly. There are no per-size stroke floors, fitted bounds,
background-colored holes or runtime optical offsets.

`camera-align` means aligning an authored camera to the current editor view. Its camera and
open lens identify the object, the northeast arrow expresses the operation, and the detached
destination corner identifies the view. It is intentionally distinct from the circular Reset
arrow, viewing a camera, piloting, focusing, and image capture. The camera aperture and the
destination/arrow clearance remain open at small sizes. This local extension does not change
native renderer, world helpers or G3 gizmo geometry.

Run `node --test tests/camera-align-icon.test.mjs` for generated-source equality, canonical
geometry, transparent aperture coverage at 16/20/24 points, fractional scales and 1×/2× density.
The test produces shared-control comparison galleries and actual Inspector/menu hover captures
under `../output/electron/verification/camera-align-*`. Inspect the small controls beside Camera, Fit, Reset and
Perspective; enlarged artwork alone is not visual acceptance.
