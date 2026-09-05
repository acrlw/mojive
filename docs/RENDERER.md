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

Opaque draw buckets follow actual resource bindings: instances with the same mesh and texture share
one draw even when their logical materials differ. Their shading values and texture coordinates are
already carried by the visual instance stream. Transparent instances stay separate because their
camera-dependent draw order is semantically observable. This boundary keeps authoring identity out
of the command graph and lets both backends benefit from the same batching policy.

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

WebGPU render-target views and resource bind groups follow the same ownership model. Target views
are created with their textures, while scene, tendon, outline, identity-presentation, and RGB-pack
bindings persist until one of their buffers or views is replaced. All synchronous output formats
share one map-readable staging buffer that grows to the largest transfer and survives target-view
replacement. Resize, MSAA changes, and instance-buffer growth rebuild only the affected resources
rather than allocating wrappers or staging buffers every frame.

The MuJoCo adapter derives `FrameNeeds` from visible scene options. Contacts, tendons, deformables,
diagnostics, islands, and BVH data are prepared only when a visible feature consumes them. Pose
data remains mandatory. This keeps optional simulation extraction out of ordinary frames without
coupling the adapter to a specific graphics backend.

## Readback boundary

OpenGL color capture reads packed RGB directly into a persistent staging allocation or a supplied
output array. Metric depth and segmentation likewise read their typed export targets directly.
The current synchronous API deliberately returns a ready NumPy array before `render()` completes.

WebGPU cannot copy an `rgba8unorm` texture directly into tightly packed three-channel rows. Before
mapping, a persistent compute pass therefore converts each four RGBA pixels into three `u32` words.
The mapped payload is exactly three bytes per pixel (apart from at most nine terminal padding bytes),
instead of a four-byte texture copy followed by a strided CPU channel extraction. The shader,
pipeline, storage buffer, texture binding, and dispatch geometry survive across frames; target
resize invalidates only the size-dependent buffers and binding. Synchronous capture records the
packing dispatch and copy through the target's persistent map-readable staging buffer in one command
encoder, then maps that buffer directly. This avoids the second submission and transient staging
allocation hidden by `queue.read_buffer()`. The conversion is byte-exact for the normalized render
target. If an unusually large target exceeds a device's storage-binding limit, it falls back to
the texture-copy path rather than failing capture. The generic row decoder retains Pillow's
compiled RGBA conversion for that path and other texture-copy callers.

`Renderer.render_async()` is the explicit pipelined alternative. On wgpu it records the same GPU
packing pass and a buffer copy into a lazily-created, three-slot staging ring, then returns a
standard `Future`. GPU completion, mapping, orientation, and optional destination-array copying run
on a dedicated worker. The decoder reads mapped storage directly and completes its owned result
before unmapping, avoiding an intermediate CPU copy. Saturating all three slots applies bounded
backpressure instead of growing GPU memory without limit. Resize, sample-count changes, and release
drain outstanding tickets before destroying their textures. A supplied output array belongs to its
ticket until the future completes.

Metric depth, object ID, and segmentation use the same staging owner and generic row decoder.
Caller-owned typed arrays are filled directly from mapped rows rather than through a second full
image allocation. Shape, orientation, dtype-casting, and strided-output behavior remain unchanged.
OpenGL exposes the same asynchronous method as an already-completed future until a PBO-backed queue
is justified; no background GL context is introduced implicitly.

The high-level video recorder uses the same three-frame pipeline when the active target supports
it, preserving submission and encoding order. Camera preview requests color rather than the full
viewport contract, so ordinary previews no longer render an unused picking buffer; identity debug
views still add their required product through the backend's plan compiler.

Native vectorized camera/world rendering is a separate concern from asynchronous single-target
capture. Its current limits and proposed resource, scene, camera, and GPU-result contracts are
documented in [Batch rendering for reinforcement learning](BATCH_RENDERING.md).

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
| Instancing | Opaque mesh/texture-binding buckets; individually sorted transparent instances |
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

The primitive Dimensions mode reuses the same projection, hit testing, input routing, edit
transactions, and `Draw2D` overlay. Its square endpoints expose only independent authored
parameters; it does not introduce transform scale or another backend render pass.

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

On an Apple M5 at 1920×1080 with 4× MSAA, the isolated public-API benchmark records the following
RGB update-and-render medians after the lifecycle-stream, persistent-pass, stable-binding, and
single-submission RGB readback changes:

| Workload | MuJoCo | Mojive OpenGL | Mojive wgpu |
|---|---:|---:|---:|
| 256 static objects | 6.37 ms | 3.17 ms (0.50×) | 2.43 ms (0.38×) |
| 1,024 animated objects | 10.05 ms | 3.59 ms (0.36×) | 2.73 ms (0.27×) |
| textured, transparent, reflective materials | 4.32 ms | 2.97 ms (0.69×) | 2.15 ms (0.50×) |
| 256 untextured material variants | 8.12 ms | 3.19 ms (0.39×) | 7.19 ms (0.89×) |

The ratio is Mojive time divided by MuJoCo time, so lower is faster. These measurements are evidence
for regression tracking rather than fixed performance requirements; use the benchmark commands on
the target machine when changing pipeline structure. Metal readback on this host has distinct
power-state modes, so renderer order can change absolute wgpu latency. In alternating same-machine
A/B rounds, moving packing from the native CPU kernel to the persistent GPU pass reduced the
materials median from 3.43–3.50 ms to 2.50–2.60 ms. The earlier strided NumPy path was about 10.3 ms
under the same load. Combining GPU packing and staging copy into one submission then reduced four
interleaved-process rounds from a 2.541 ms median to 2.186 ms, about 14 percent, while preserving
byte-exact output. The measured process RSS delta also fell from roughly 62.5 to 56.6 MiB,
consistent with replacing the temporary `queue.read_buffer()` path by persistent staging. Compare
repeated rounds rather than selecting one favorable run.

The same persistent staging design now covers typed outputs. In two alternating 100-frame A/B
rounds at 1920×1080, WebGPU depth fell from a 2.752 ms combined median to 2.199 ms (20.1 percent),
and segmentation from 3.073 ms to 2.466 ms (19.7 percent). Sync and async paths remained
pixel-identical, including direct caller-owned output arrays.

The benchmark report separates the backend command graph from synchronous readback under
`backend_render`. On the same materials scene, color-only WebGPU submission used 0.309 ms of CPU
and completed at 0.590 ms/frame throughput; the interactive viewport request, including the object
ID export, used 0.316 ms of CPU and completed at 0.669 ms/frame. This keeps renderer architecture
decisions tied to command-graph measurements even when Metal mapping latency changes power state.

The `material_variants` workload isolates draw-bucket scalability: it uses 256 distinct logical
materials over a small set of built-in meshes while leaving their texture binding identical. On the
same 1920×1080 setup, resource-binding batching reduced the scene from 321 buckets to 6. Median
render-graph CPU time fell from 1.406 to 0.319 ms on OpenGL and from 2.301 to 0.589 ms on wgpu;
optimized and baseline RGB captures were byte-identical on both backends. OpenGL end-to-end time
fell from 4.00 to 3.19 ms. Metal's variable readback power state can outweigh command-encoding
savings in an individual wgpu run, so its end-to-end row is a captured sample, while the isolated
CPU result is the appropriate regression signal for this change.
