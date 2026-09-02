# Implementation decisions

This document records choices that affect public contracts, rendering results, and platform
behavior. Platform measurements live in [PLATFORM.md](PLATFORM.md). Reference-rendering behavior
lives in [RENDERER.md](RENDERER.md).

## Scene data

### MuJoCo array copies

MuJoCo Python exposes `geom_xpos` and `geom_xmat` as contiguous arrays. The adapter copies each
array into a preallocated frame buffer. Apple M5 measurements:

| Geoms | Bulk copy | Element loop |
|---:|---:|---:|
| 401 | 0.9 us | 252 us |
| 1,501 | 2.8 us | 1,145 us |
| 5,001 | 7.5 us | 3,889 us |

### Object identity

Robot and link nodes carry stable nonzero object IDs. Child geom, joint, and site nodes carry a
body index and object ID zero. Geometry instances use the parent object ID in the ID buffer.
Session lookup therefore resolves one selectable node per object.

### Bucket identity

Draw buckets represent GPU binding state rather than logical material identity. Opaque instances
share a bucket when they use the same mesh and texture binding; color, emission, specular,
shininess, reflectance, and texture coordinates remain per-instance data. A bucket retains one
representative material ID so backends can resolve its shared texture without adding another public
table. Transparent instances remain individually bucketed for stable back-to-front sorting. Public
scene inspection presents the mesh and representative material while opaque and transparent bucket
lists expose pass membership.

### Geometry color

An explicitly authored geom RGBA overrides its material. Default geom color resolves through the
material. Alpha follows the same path.

### Renderer ownership

`RenderBackend.set_scene()` creates the renderer-side scene builder. `update()` consumes frame
data. This keeps scene conversion inside the rendering boundary and gives protocol clients a
complete setup and frame path.

## Picking and overlays

GPU picking and physics raycasts use the same visual-group mask. World-body hits resolve to the
scene root. Object IDs use a dedicated `uint32` attribute and `R32UI` target.

Overlay passes mask the shared ID attachment when the platform uses a combined color and ID FBO.
This preserves picking data under text, debug drawing, outlines, and gizmos.

## Texture coordinates

Per-instance texture coefficients carry repeat, offset, and primitive mapping metadata.
Primitive boxes generate object-space coordinates for all three face-axis pairs and apply
MuJoCo `texuniform` scaling. Meshes use their authored UV coordinates.

## Debug primitives

`Layer.arrow()` represents a screen-width debug arrow. `ARROW_SHAFT` and `ARROW_HEAD` remain mesh
shapes for perspective-scaled scene geometry. A command keeps one width model throughout its
lifetime.

Sector radius defaults to the reference vector length. `radius_px` selects a screen-space radius.
Frame axis length uses world units; frame line width uses the theme's screen-space width.

Perturbation feedback uses the always-visible overlay tier. Translation feedback joins its start
ring, connector, and target into one stroked shape. Rotation feedback uses a rounded projected
silhouette and a depth-sorted reference frame.

## Axis colors

X, Y, and Z use a luminance-balanced palette across transform gizmos, perturbation feedback,
orientation widgets, and scene frames. The native gizmo implementation owns the complete style.

## Planar reflection

The reflection pass renders a mirrored camera into an offscreen texture. Reflective fragments
sample that texture in screen space. Oblique clipping removes geometry behind the plane and
winding reversal preserves front faces. Coplanar surfaces share a target; up to four distinct
planes receive independent mirrored views. Reflective boxes use their local positive-Z face.

## Reference calibration

MuJoCo 3.11 measurements establish these lighting rules:

| Configuration | Reference result |
|---|---:|
| Headlight ambient 0.4 | 97 |
| Scene-light ambient 0.4 | 97 |
| Both ambient terms at 0.4 | 194 |

OpenGL adds active ambient terms and uses an ambient gain of 1.0. A 0.6 texture sample scales to
38, 77, and 153 under ambient values 0.25, 0.5, and 1.0.

MuJoCo image-light intensity 5000 maps to unit cube radiance. Diffuse lighting samples the
coarsest cube mip. Specular lighting selects a mip from material shininess.

## Verification design

Regression assertions target observable behavior and fail under a focused mutation. Examples:

- vertical camera movement checks full commanded displacement and zero horizontal drift;
- gesture ownership locks on the press frame;
- gizmo axis reversal changes screen-space direction;
- rebuild debounce spans a representative human pause;
- integer object IDs verify declared integer attribute types.

`make reverse` applies the registered mutations and confirms the corresponding checks fail.
