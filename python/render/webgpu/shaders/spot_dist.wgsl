// Local-light distance-map shaders for the webgpu backend.
//
// Port of opengl's spot_dist.vert/spot_dist.frag: writes the world-space
// distance to the light into one r16float layer of the local shadow array.
// Fragments beyond the light range discard, and the pipeline blends with MIN
// (there is no depth attachment) so the nearest occluder distance survives.

struct ShadowDraw {
    view_proj: mat4x4f,         // spot matrix or one point-cube-face matrix
    light: vec4f,               // xyz u_light_pos, w u_light_range (spot_dist.frag)
};

// Transform-only lifecycle stream; keep in sync with instances.py.
struct InstancePose {
    model: mat4x4f,
};

@group(0) @binding(0) var<uniform> u_draw: ShadowDraw;
@group(0) @binding(1) var<storage, read> instance_pose: array<InstancePose>;

struct DistOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
};

@vertex
fn vs_dist(
    @location(0) position: vec3f,
    @builtin(instance_index) instance_index: u32,
) -> DistOut {
    let world = instance_pose[instance_index].model * vec4f(position, 1.0);
    var out: DistOut;
    out.clip = u_draw.view_proj * world;
    out.world = world.xyz;
    return out;
}

@fragment
fn fs_dist(in: DistOut) -> @location(0) f32 {
    let dist = length(in.world - u_draw.light.xyz);
    if u_draw.light.w > 0.0 && dist > u_draw.light.w {
        discard;
    }
    return dist;
}
