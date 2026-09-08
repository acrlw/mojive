// Pixel-radius round points with fwidth antialiasing; port of opengl's
// debug_point.vert/debug_point.frag.  The quad is offset in view space by
// radius * px_scale * clip.w, so the radius is constant in pixels.

struct DebugPointIn {
    @location(0) p: vec3f,
    @location(1) color: vec4f,
    @location(2) radius: f32,  // pixels
};

struct DebugPointOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
};

@vertex
fn vs_debug_point(in: DebugPointIn, @builtin(vertex_index) v: u32) -> DebugPointOut {
    var C = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
    );
    var out: DebugPointOut;
    out.color = vec4f(in.color.rgb, in.color.a * dbg_alpha());
    out.uv = C[v];

    var p = (dbg.view * vec4f(in.p, 1.0)).xyz;
    let w = (dbg.proj * vec4f(p, 1.0)).w;
    p = vec3f(p.xy + out.uv * (in.radius * dbg_px_scale() * w), p.z);
    out.pos = dbg.proj * vec4f(p, 1.0);
    return out;
}

@fragment
fn fs_debug_point(in: DebugPointOut) -> @location(0) vec4f {
    let r = length(in.uv);
    if r > 1.0 {
        discard;
    }
    let aa = max(fwidth(r), 1e-4);
    return vec4f(in.color.rgb, in.color.a * (1.0 - smoothstep(1.0 - aa, 1.0, r)));
}
