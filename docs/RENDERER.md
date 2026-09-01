# Renderer design

Mojive provides the OpenGL backend and the cross-platform wgpu backend. Both consume
backend-neutral scene data and implement the same linear-light, bucketed render pipeline.

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

The wgpu backend (`MOJIVE_BACKEND=wgpu`) runs the same pass order with WebGPU
constructions for the GL-only pieces: picking/segmentation/depth readback comes from a
single-sampled export MRT pass that re-rasterizes the scene instead of an MSAA blit resolve
(WebGPU cannot resolve integer or depth MSAA), wireframe carries barycentrics in a lazily
built vertex attribute instead of a geometry shader, and reflection clipping is a fragment
discard on a plane equation instead of `gl_ClipDistance`. MSAA sample counts are fixed at
construction in both backends.

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
| Timing | CPU measurements on both backends; GPU measurements on OpenGL when timer queries exist |

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
