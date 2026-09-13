// Drag-link compound shape (hollow start ring, connector, filled target dot)
// resolved by a screen-space SDF in the fragment stage; port of opengl's
// debug_drag_link.vert/debug_drag_link.frag.  GLSL ``noperspective`` becomes
// WGSL ``@interpolate(linear)`` and ``flat`` becomes ``@interpolate(flat)``;
// the quad spans the pixel-space bounding box of both endpoints.

struct DebugDragLinkIn {
    @location(0) a: vec3f,
    @location(1) b: vec3f,
    @location(2) core_color: vec4f,
    @location(3) edge_color: vec4f,
    @location(4) width: f32,
    @location(5) radius: f32,
    @location(6) edge: f32,
    @location(7) smoothing: f32,
};

struct DebugDragLinkOut {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(linear) pixel: vec2f,
    @location(1) @interpolate(flat) a: vec2f,
    @location(2) @interpolate(flat) b: vec2f,
    @location(3) @interpolate(flat) core_color: vec4f,
    @location(4) @interpolate(flat) edge_color: vec4f,
    @location(5) @interpolate(flat) width: f32,
    @location(6) @interpolate(flat) radius: f32,
    @location(7) @interpolate(flat) edge: f32,
    @location(8) @interpolate(flat) smoothing: f32,
};

@vertex
fn vs_debug_drag_link(in: DebugDragLinkIn, @builtin(vertex_index) v: u32) -> DebugDragLinkOut {
    let clip_a = dbg.view_proj * vec4f(in.a, 1.0);
    let clip_b = dbg.view_proj * vec4f(in.b, 1.0);
    let viewport = dbg_viewport();
    var out: DebugDragLinkOut;
    out.a = (clip_a.xy / clip_a.w * 0.5 + vec2f(0.5)) * viewport;
    out.b = (clip_b.xy / clip_b.w * 0.5 + vec2f(0.5)) * viewport;
    out.core_color = vec4f(in.core_color.rgb, in.core_color.a * dbg_alpha());
    out.edge_color = vec4f(in.edge_color.rgb, in.edge_color.a * dbg_alpha());
    out.width = in.width;
    out.radius = in.radius;
    out.edge = in.edge;
    out.smoothing = in.smoothing;

    // The hollow start ring extends half a core stroke beyond its radius.
    // Include that stroke as well as the contrast edge and AA guard in the
    // primitive quad; otherwise large UI scales clip the ring at the quad.
    let pad = in.radius + 0.5 * in.width + in.edge + in.radius * in.smoothing * 0.3125 + 2.0;
    let lo = min(out.a, out.b) - vec2f(pad);
    let hi = max(out.a, out.b) + vec2f(pad);
    var C = array<vec2f, 6>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
        vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
    );
    out.pixel = mix(lo, hi, C[v]);
    let ndc = out.pixel / viewport * 2.0 - vec2f(1.0);
    out.pos = vec4f(ndc * clip_a.w, clip_a.z, clip_a.w);
    return out;
}

fn dbg_capsule_sdf(p: vec2f, a: vec2f, b: vec2f, radius: f32) -> f32 {
    let ab = b - a;
    let d2 = dot(ab, ab);
    let t = select(0.0, clamp(dot(p - a, ab) / d2, 0.0, 1.0), d2 > 1e-6);
    return length(p - (a + t * ab)) - radius;
}

// Even sextic matches abs(t) through its third derivative at both blend limits.
fn dbg_smooth_union(a: f32, b: f32, blend: f32) -> f32 {
    if blend <= 0.0 || abs(a - b) >= blend { return min(a, b); }
    let t = (a - b) / blend;
    let t2 = t * t;
    let profile = (5.0 + t2 * (15.0 + t2 * (-5.0 + t2))) / 16.0;
    return 0.5 * (a + b - blend * profile);
}

@fragment
fn fs_debug_drag_link(in: DebugDragLinkOut) -> @location(0) vec4f {
    let half_width = 0.5 * in.width;
    let ab = in.b - in.a;
    let length_ab = length(ab);
    let direction = select(vec2f(1.0, 0.0), ab / length_ab, length_ab > 1e-4);

    // Blend only the outside of the origin. The connector cannot dent its hole.
    let link_start = in.a + direction * in.radius;
    let radial = length(in.pixel - in.a);
    let disk = radial - in.radius - half_width;
    let hole = in.radius - half_width - radial;
    let link = dbg_capsule_sdf(in.pixel, link_start, in.b, half_width);
    let dot_end = length(in.pixel - in.b) - in.radius;
    let blend = in.radius * in.smoothing;
    let origin = max(dbg_smooth_union(disk, link, blend), hole);
    let shape = dbg_smooth_union(origin, dot_end, blend);

    let aa = max(fwidth(shape), 0.65);
    let outer = 1.0 - smoothstep(-aa, aa, shape - in.edge);
    if outer <= 0.0 {
        discard;
    }
    let core = 1.0 - smoothstep(-aa, aa, shape);
    let color = mix(in.edge_color, in.core_color, core);
    return vec4f(color.rgb, color.a * outer);
}
