# Platform measurements

Measurements come from `tools/probe_gl.py` on macOS 26.6.1, Apple M5, OpenGL 4.1 core over
Metal, and GLSL 410. Run `make probe` to refresh the report.

## Capability summary

| Capability | Measurement | Design consequence |
|---|---:|---|
| OpenGL version | `4.1 Metal - 90.5` | GLSL 410; vertex and fragment pipelines |
| Maximum samples | 8 | 4x MSAA target |
| Vertex attributes | 16 | 10 used by mesh and instance data |
| Fragment texture units | 16 | Material, shadow, skybox, and ID bindings fit |
| Multisampled `R32UI` | incomplete FBO | Separate single-sample ID pass |
| Multisampled RGBA8 and depth | complete FBO | Main color path uses MSAA |
| Integer clear through ModernGL | float bit pattern | Full-screen integer clear shader |
| `glClearBufferuiv` | available | Optional native clear path |
| GPU elapsed query | available | Frame totals and approximate pass timing |
| KHR debug | absent | Debug groups compile to empty scopes |
| Line width range | `[1.0, 1.0]` | Wide lines use triangle expansion |
| Geometry shaders | available | Barycentric wireframe path |
| HiDPI scale | 2.0 | Input and target coordinates use framebuffer scale |

## Render-target layout

`RenderTarget` selects one of two layouts during creation:

| Layout | Color and ID storage | Geometry work |
|---|---|---|
| `SHARED` | Multisampled color and integer ID attachments | Opaque geometry once |
| `SPLIT` | Multisampled color target plus single-sample `R32UI` target | Lightweight ID pass |

Apple M5 selects `SPLIT`. Picking, outline, and segmentation access the selected layout through
`RenderTarget.id_texture` and `RenderTarget.id_samples`.

## Instance-buffer offsets

ModernGL 5.12 omits byte offsets from its instance layout API. Forge creates the VAO
through ModernGL and rebinds instance pointers with `glVertexAttribPointer`,
`glVertexAttribIPointer`, and `glVertexAttribDivisor`. This work occurs during scene setup.

The portable path allocates one instance buffer per bucket. Both paths preserve bucket identity,
draw counts, and frame behavior.

## Timing on tile-based GPUs

Apple GPUs defer tile work until a flush. `GL_TIME_ELAPSED` can attribute later passes to an
earlier open query. Whole-frame totals remain representative; individual pass values provide an
approximate distribution.

`make bench` reports this condition when active queries contain a mixture of zero and nonzero
pass times. It leaves GPU scheduling asynchronous.

## Shadow cost

At 1280x720 with a 4096-square three-cascade atlas:

| Work | Time |
|---|---:|
| Shadow GPU | 0.53 ms |
| Shadow CPU | 0.081 ms |
| Cascade matrices | 0.051 ms |

The pass is fill-rate limited on Apple M5. Large ground geometry dominates the atlas workload.

## Integer attributes

Object IDs use an independent `uint32` instance attribute and `glVertexAttribIPointer`. The
Apple driver returns identical values through the floating-pointer variant, so tests protect the
declared integer attribute type. The implementation follows the OpenGL integer-input contract.

## MuJoCo reference rendering

MuJoCo's classic renderer uses a legacy OpenGL context. Forge uses a core context. Parity and
calibration launch the reference renderer in a subprocess with its own context:

```bash
make parity
make calibrate
```

## Planar reflections

ModernGL 5.12 exposes pure depth attachments through its framebuffer API. Forge renders a
mirrored camera into an offscreen texture and samples it from reflective surface fragments.
Oblique clipping and winding reversal preserve the reflected scene boundary.
