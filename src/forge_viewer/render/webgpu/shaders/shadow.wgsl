// Depth-only shadow-map vertex shader for the webgpu backend.
//
// Port of forge's shadow.vert/shadow.frag: rasterizes scene depth into one
// atlas tile.  The pipeline has no fragment stage and no color attachments —
// only depth is written (forge's shadow.frag is an empty main).

struct ShadowDraw {
    view_proj: mat4x4f,         // cascade view-projection (WebGPU clip z in [0,1])
    light: vec4f,               // unused here; keeps one layout with spot_dist.wgsl
};

// Keep in sync with instances.py (144-byte stride).
struct Instance {
    model: mat4x4f,
    color: vec4f,
    material: vec4f,
    texcoef: vec4f,
    cubecoef: vec4f,
    object_id: u32,
};

@group(0) @binding(0) var<uniform> u_draw: ShadowDraw;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;

@vertex
fn vs_shadow(
    @location(0) position: vec3f,
    @builtin(instance_index) instance_index: u32,
) -> @builtin(position) vec4f {
    let world = instances[instance_index].model * vec4f(position, 1.0);
    return u_draw.view_proj * world;
}
