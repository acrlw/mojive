// Skybox shaders for the webgpu backend: a fullscreen triangle that unprojects
// each pixel through the inverse view-projection and samples the environment
// cubemap.  Direct port of forge's skybox.vert/skybox.frag with WebGPU clip
// conventions: z in [0, 1], so the near end of the ray uses ndc z = 0.  The
// far-plane write (z = w, the WebGPU far plane) and the Z-up cubemap swizzle
// carry over unchanged.
//
// Expects common.wgsl prepended (finish_color).

struct SkyboxUniforms {
    inv_view_proj: mat4x4f,     // inverse of the WebGPU-clip view-projection
    params: vec4f,              // x exposure, y tonemap on
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

@fragment
fn fs_skybox(in: SkyboxOut) -> @location(0) vec4f {
    let d = normalize(in.dir);
    let c = textureSample(skybox_tex, skybox_sampler, vec3f(d.x, d.z, -d.y)).rgb;
    return vec4f(finish_color(c, skybox.params.x, skybox.params.y > 0.5), 1.0);
}
