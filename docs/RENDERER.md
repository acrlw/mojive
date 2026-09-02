# Renderer design

Mojive provides the OpenGL backend and the cross-platform wgpu backend. Both consume
backend-neutral scene data and implement the same linear-light, bucketed render pipeline.

## Frame pipeline

```text
shadow
reflect
opaque
id
export
skybox
tendon
transparent
outline
debug
gizmo
present
```

This is the superset, not a fixed cost paid by every call. A backend compiles a pass plan from a
backend-neutral `RenderRequest` and schedules only the work needed for the requested products:

- viewport: resolved color plus object IDs for picking and outlines
- color capture: resolved color only
- metric depth: an `R32F` export target only
- segmentation: an `RG32I` semantic-ID export target only

The ID pass shares scene visibility with opaque rendering and supplies picking and selection
outlines. Metric depth and MuJoCo `(object_id, object_type)` pairs are written directly by the GPU;
they do not require full-frame CPU depth conversion or an ID lookup. Debug and gizmo passes
consume generic commands and UI state.

The wgpu backend (`MOJIVE_BACKEND=wgpu`) runs the same pass order with WebGPU
constructions for the GL-only pieces: picking/segmentation/depth readback comes from a
single-sampled export MRT pass that re-rasterizes the scene instead of an MSAA blit resolve
(WebGPU cannot resolve integer or depth MSAA), wireframe carries barycentrics in a lazily
built vertex attribute instead of a geometry shader, and reflection clipping is a fragment
discard on a plane equation instead of `gl_ClipDistance`. The `msaa` render flag switches the
active targets between 1× and the configured multisample count; sample-count-dependent resources
are rebuilt through the backend's normal option path.

Both backends separate instance preparation from pass execution. The reflection planner writes a
small, pass-owned routing value for each reflective surface; it never overwrites material
reflectance in `RenderScene`. The backend resolves this metadata before executing the planned
passes, so render passes cannot silently mutate persistent scene state.

## Data lifetime and upload policy

`SceneSource` owns topology and immutable resources. `RenderScene` carries independent structure,
pose, visual, and identity revisions. OpenGL and WebGPU store instance data in three matching GPU
streams: 64-byte transforms, 64-byte visual/material records, and 16-byte identity/reflection
records. A normal animated frame uploads only changed transform ranges instead of retransmitting
the full 144-byte record. Visual, identity, and reflection changes update their own contiguous
ranges. Revision zero remains a conservative compatibility path for custom sources, with bytewise
dirty-range detection preserving correctness without unconditional GPU writes.

Shadow maps and planar-reflection colors are persistent pass products. Both backends reuse them
while their complete dependency keys are unchanged: scene lifecycle revisions, camera, lights,
mesh and texture resources, render flags, selection/debug state, target size, and shader generation.
This removes the expensive offscreen passes from static frames without lowering update frequency or
introducing temporal lag. Any dependency change invalidates the relevant product immediately;
revision-zero custom scenes deliberately take the conservative render-every-frame path.

The MuJoCo adapter derives `FrameNeeds` from visible scene options. Contacts, tendons, deformables,
diagnostics, islands, and BVH data are prepared only when a visible feature consumes them. Pose
data remains mandatory. This keeps optional simulation extraction out of ordinary frames without
coupling the adapter to a specific graphics backend.

## Readback boundary

OpenGL color capture reads packed RGB directly into a persistent staging allocation or a supplied
output array. Metric depth and segmentation likewise read their typed export targets directly.
The current synchronous API deliberately returns a ready NumPy array before `render()` completes.

`Renderer.render_async()` is the explicit pipelined alternative. On wgpu it submits a texture copy
into a lazily-created, three-slot staging ring and returns a standard `Future`. GPU completion,
mapping, row-padding removal, RGB packing, orientation, and optional destination-array copying run
on a dedicated worker. Saturating all three slots applies bounded backpressure instead of growing
GPU memory without limit. Resize, sample-count changes, and release drain outstanding tickets
before destroying their textures. A supplied output array belongs to its ticket until the future
completes.

The synchronous API and its dtype-casting/strided-output behavior remain unchanged. OpenGL exposes
the same asynchronous method as an already-completed future until a PBO-backed queue is justified;
no background GL context is introduced implicitly. On an Apple M5, a 40-frame 1920×1080 wgpu RGB
capture measured about 98 FPS synchronously and 197 FPS through the staging ring, with pixel-exact
results. This is a throughput measurement, not a promise that single-frame latency is halved.
Metric-depth and segmentation synchronous readback is already inexpensive on this device; their
asynchronous form is provided for non-blocking composition, not assumed to improve throughput.

The high-level video recorder uses the same three-frame pipeline when the active target supports
it, preserving submission and encoding order. Camera preview requests color rather than the full
viewport contract, so ordinary previews no longer render an unused picking buffer; identity debug
views still add their required product through the backend's plan compiler.

## Color pipeline

Texture sampling uses sRGB formats. Mojive-native scene sources combine lighting, reflection, fog,
haze, emission, and selection highlight in linear light, then apply a soft tone-mapping knee at
0.8 before display encoding.

MuJoCo scene sources select the `mujoco-classic` shading model. It reconstructs display-domain
material and texture values, combines the classic renderer's light contributions in display
space, and bypasses OpenGL tone mapping. Other adapters retain the linear pipeline by default.

`make calibrate` measures individual terms. `make parity` evaluates the complete scene.

## Reference-aligned behavior

- Shininess maps to a Phong exponent through `shininess * 128`.
- Default headlight terms are diffuse 0.4, specular 0.5, and ambient 0.1.
- Headlight and active scene-light ambient terms add together.
- MuJoCo render and visualization flags retain their public names.
- MuJoCo horizon haze is enabled by default, matching `mjRND_HAZE`. Its truncated-cone geometry
  uses the model's `quality.numslices`, and its color is blended in the classic display domain
  over the sky without depth-fogging scene geometry.
- Infinite MuJoCo planes reconstruct the classic renderer's 200-cell grid and triangle-interpolate
  spotlight attenuation, avoiding the hard per-fragment boundary produced by a two-triangle proxy
  plane.
- Classic fixed-function specular is modulated by the surface texture together with ambient and
  diffuse lighting; Mojive-native scenes retain the untextured specular path.
- Texture surfaces receive lighting.
- Image lights sample cube-map diffuse radiance and roughness-aware specular mip levels. An
  intensity of 5000 maps to unit radiance in the OpenGL lighting model.
- Tendons use their model material, RGBA, width, texture, and transparency.

## Renderer diagnostics

| Feature | Implementation |
|---|---|
| Selection highlight | Linear-light tint plus emission |
| Selection outline | ID edge detection with x-ray visibility |
| Picking | `R32UI` object IDs |
| Instancing | Mesh/material/transparency buckets |
| Shadows | Three-cascade directional atlas and local-light shadows |
| Reflections | Mirrored camera, oblique clipping, and surface sampling |
| Wide lines | Screen-space triangle strips |
| Text | GPU glyph atlas shared with UI font configuration |
| Timing | CPU measurements on both backends; GPU measurements on OpenGL and optional asynchronous timestamp queries on wgpu |

## Transparency

Opaque and transparent instances use separate buckets. Transparent objects skip the shadow map,
blend in the transparent pass, and preserve depth testing. Tendons follow the same material and
transparency model.

## Dynamic geometry

`SceneSource` owns stable topology. `MeshUpdate` provides frame-local positions and normals.
This supports flex surfaces, skinned meshes, and custom deformable backends while preserving the
static mesh upload path.

MuJoCo flex modes map to independent instance filters:

- flex face: flat element surfaces
- flex skin: smooth shell surfaces
- skin: skinned meshes
- static: world-welded geometry

## Debug draw

Debug primitives use world anchors and screen-space widths. Supported commands include points,
lines, arrows, frames, boxes, spheres, sectors, polylines, and text. Occlusion modes select depth,
ghost, or always-visible presentation. Batches support local scripts, socket clients, remote
viewers, and snapshot replay.

## Transform gizmos

Native 2D and 3D gizmos share interaction state, colors, sizing, snapping, labels, and drag
feedback. Their pixel footprint remains stable with camera distance. Axis depth ordering follows
camera-space depth.

Position snapping uses a projected axis ruler. Rotation snapping uses an outer tick ring. The
default increments are 0.5 m and 5 degrees.

## Measurement commands

```bash
make bench
make parity
make calibrate
make showcase
make gallery
make gizmo-gallery
make renderer-benchmark
make renderer-benchmark-full
```

On an Apple M5 at 1920×1080 with 4× MSAA, the architecture benchmark records OpenGL RGB capture at
3.19–3.40 ms, metric depth at 2.02–2.13 ms, and segmentation at 2.74–3.03 ms across the bundled
workloads. These measurements are evidence for regression tracking rather than fixed performance
requirements; use the benchmark commands on the target machine when changing pipeline structure.
