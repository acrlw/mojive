// Shared uniforms and helpers for the webgpu debug draw shaders, mirroring
// the uniform set of opengl's debug_*.vert programs.  Prepended to the other
// debug WGSL sources by programs.load_wgsl; do not compile standalone.
//
// All debug pipelines bind the same block at group(0) binding(0) with a
// dynamic offset: slot 0 carries alpha=1, slot 1 the GHOST occluded alpha
// (see passes/debug.py).

struct DebugUniforms {
    view_proj: mat4x4f,  // WebGPU-clip view-projection
    view: mat4x4f,
    proj: mat4x4f,
    params: vec4f,       // viewport width, viewport height, px_scale, alpha
};

@group(0) @binding(0) var<uniform> dbg: DebugUniforms;

fn dbg_viewport() -> vec2f {
    return dbg.params.xy;
}

fn dbg_px_scale() -> f32 {
    return dbg.params.z;
}

fn dbg_alpha() -> f32 {
    return dbg.params.w;
}

// NDC to pixel coordinates (debug_line.vert's screen_pos).
fn dbg_screen_pos(clip: vec4f) -> vec2f {
    return (clip.xy / clip.w * 0.5 + vec2f(0.5)) * dbg_viewport();
}

// Pixel coordinates back into clip space (debug_line.vert's place).
fn dbg_place(clip: vec4f, screen: vec2f) -> vec4f {
    return vec4f((screen / dbg_viewport() * 2.0 - vec2f(1.0)) * clip.w, clip.z, clip.w);
}
