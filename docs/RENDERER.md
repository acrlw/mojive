# Renderer design

Forge is an OpenGL 4.1 renderer for simulation inspection and tooling. It uses a linear-light
pipeline, bucketed instancing, backend-neutral scene data, and explicit diagnostic passes.

## Frame pipeline

```text
shadow
reflect
opaque
id
skybox
tendon
transparent
present
outline
debug
gizmo
```

The ID pass shares scene visibility with opaque rendering and supplies picking, segmentation,
and selection outlines. Debug and gizmo passes consume generic commands and UI state.

The wgpu backend (`FORGE_VIEWER_BACKEND=wgpu`) runs the same pass order with WebGPU
constructions for the GL-only pieces: picking/segmentation/depth readback comes from a
single-sampled export MRT pass that re-rasterizes the scene instead of an MSAA blit resolve
(WebGPU cannot resolve integer or depth MSAA), wireframe carries barycentrics in a lazily
built vertex attribute instead of a geometry shader, and reflection clipping is a fragment
discard on a plane equation instead of `gl_ClipDistance`. MSAA sample counts are fixed at
construction in both backends.

## Color pipeline

Texture sampling uses sRGB formats. Lighting, reflection, fog, haze, emission, and selection
highlight combine in linear light. Tone mapping applies a soft knee at 0.8 before display
encoding.

MuJoCo's classic fixed-function renderer combines lighting in display space. The difference is
measurable in scenes with multiple light contributions. On the parity floor, Forge/reference
brightness measures about 0.774. Isolated diffuse, ambient, headlight, and texture calibration
remain aligned.

`make calibrate` measures individual terms. `make parity` evaluates the complete scene.

## Reference-aligned behavior

- Shininess maps to a Phong exponent through `shininess * 128`.
- Default headlight terms are diffuse 0.4, specular 0.5, and ambient 0.1.
- Headlight and active scene-light ambient terms add together.
- MuJoCo render and visualization flags retain their public names.
- Texture surfaces receive lighting.
- Image lights sample cube-map diffuse radiance and roughness-aware specular mip levels. An
  intensity of 5000 maps to unit radiance in the Forge lighting model.
- Tendons use their model material, RGBA, width, texture, and transparency.

## Forge diagnostics

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
| Timing | CPU and GPU measurements per named pass |

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
```
