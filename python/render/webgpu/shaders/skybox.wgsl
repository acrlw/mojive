// Mojive-native scenes use a fullscreen skybox at the far plane. MuJoCo classic
// scenes use the reference renderer's depth-writing closed cylinder at 70% of
// the far distance. Both paths sample the same Z-up cubemap convention.
//
// Expects common.wgsl prepended (finish_color).

struct SkyboxUniforms {
    inv_view_proj: mat4x4f,     // inverse of the WebGPU-clip view-projection
    view_proj: mat4x4f,
    eye_distance: vec4f,        // xyz camera eye, w classic cylinder distance
    params: vec4f,              // x exposure, y tonemap on, z MuJoCo classic
};

@group(0) @binding(0) var<uniform> skybox: SkyboxUniforms;
@group(1) @binding(0) var skybox_tex: texture_cube<f32>;
@group(1) @binding(1) var skybox_sampler: sampler;

struct SkyboxOut {
    @builtin(position) clip: vec4f,
    @location(0) dir: vec3f,
};

@vertex
fn vs_skybox(@builtin(vertex_index) vertex_index: u32) -> SkyboxOut {
    let p = vec2f(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    let ndc = p * 2.0 - 1.0;
    let near = skybox.inv_view_proj * vec4f(ndc, 0.0, 1.0);
    let far = skybox.inv_view_proj * vec4f(ndc, 1.0, 1.0);
    var out: SkyboxOut;
    out.dir = far.xyz / far.w - near.xyz / near.w;
    out.clip = vec4f(ndc, 1.0, 1.0);
    return out;
}

@vertex
fn vs_classic_skybox(@location(0) position: vec3f) -> SkyboxOut {
    var out: SkyboxOut;
    out.dir = position;
    let world = skybox.eye_distance.xyz + skybox.eye_distance.w * position;
    out.clip = skybox.view_proj * vec4f(world, 1.0);
    return out;
}

@fragment
fn fs_skybox(in: SkyboxOut) -> @location(0) vec4f {
    let d = normalize(in.dir);
    let c = textureSample(skybox_tex, skybox_sampler, vec3f(d.x, d.z, -d.y)).rgb;
    let rgb = select(
        finish_color(c, skybox.params.x, skybox.params.y > 0.5),
        linear_to_srgb3(c),
        skybox.params.z > 0.5,
    );
    return vec4f(rgb, 1.0);
}
