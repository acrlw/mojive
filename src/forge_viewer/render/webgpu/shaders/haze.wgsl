// Horizon-haze ring shaders for the webgpu backend.  Direct port of forge's
// haze.vert/haze.frag: a two-layer ring band positioned in world space around
// the camera on the infinite ground plane, alpha-blended (0 -> 1 -> 0 across
// the two layers).  CPU-side ring vertices are generated once by
// passes/skybox.py.
//
// Expects common.wgsl prepended (finish_color, srgb_to_linear3).

struct HazeUniforms {
    view_proj: mat4x4f,         // WebGPU-clip view-projection
    eye: vec4f,
    basis_x: vec4f,
    basis_y: vec4f,
    normal: vec4f,
    geometry: vec4f,            // skybox distance, elevation, radius, transition height
    color: vec4f,               // raw sRGB
    params: vec4f,              // x exposure, y tonemap on, z MuJoCo classic
};

@group(0) @binding(0) var<uniform> haze: HazeUniforms;

struct HazeOut {
    @builtin(position) clip: vec4f,
    @location(0) alpha: f32,
};

@vertex
fn vs_haze(@location(0) in_haze: vec3f) -> HazeOut {
    let layer = in_haze.z;
    var height = 1.0;
    if layer < 0.5 {
        height = 0.0;
    } else if layer < 1.5 {
        height = haze.geometry.w;
    }
    let radial = 1.0 - haze.geometry.z * (1.0 - height);
    let world = haze.eye.xyz
        + haze.geometry.x * radial * (in_haze.x * haze.basis_x.xyz + in_haze.y * haze.basis_y.xyz)
        + haze.geometry.y * (height - 1.0) * haze.normal.xyz;
    var out: HazeOut;
    out.alpha = select(0.0, 1.0, layer > 0.5 && layer < 1.5);
    out.clip = haze.view_proj * vec4f(world, 1.0);
    return out;
}

@fragment
fn fs_haze(in: HazeOut) -> @location(0) vec4f {
    let color = select(
        finish_color(srgb_to_linear3(haze.color.xyz), haze.params.x, haze.params.y > 0.5),
        haze.color.xyz,
        haze.params.z > 0.5,
    );
    return vec4f(color, in.alpha);
}
