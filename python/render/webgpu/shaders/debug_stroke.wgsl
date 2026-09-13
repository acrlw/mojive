// Connected screen-space stroke, port of opengl's debug_stroke.vert (the
// fragment is debug_line.wgsl's fs_debug_line).  Each instance is one segment
// plus its previous vertex: the body is a quad and the outside of its start
// corner is one small arc fan — a real round join, not a circle patched over
// two independent line caps.

const STROKE_JOIN_SEGMENTS: i32 = 6;

struct DebugStrokeIn {
    @location(0) prev: vec3f,
    @location(1) a: vec3f,
    @location(2) b: vec3f,
    @location(3) color: vec4f,
    @location(4) width: f32,
};

// debug_stroke.vert's screen_pos: scaled NDC, no [0,1] remap.
fn dbg_stroke_screen(clip: vec4f) -> vec2f {
    return (clip.xy / clip.w) * dbg_viewport();
}

fn dbg_safe_dir(a: vec2f, b: vec2f, fallback: vec2f) -> vec2f {
    let delta = b - a;
    let len = length(delta);
    return select(fallback, delta / len, len > 1e-5);
}

fn dbg_rotate_2d(v: vec2f, angle: f32) -> vec2f {
    let c = cos(angle);
    let s = sin(angle);
    return vec2f(c * v.x - s * v.y, s * v.x + c * v.y);
}

@vertex
fn vs_debug_stroke(in: DebugStrokeIn, @builtin(vertex_index) v: u32) -> DebugLineOut {
    var out: DebugLineOut;
    out.color = vec4f(in.color.rgb, in.color.a * dbg_alpha());

    let c_prev = dbg.view_proj * vec4f(in.prev, 1.0);
    let c_a = dbg.view_proj * vec4f(in.a, 1.0);
    let c_b = dbg.view_proj * vec4f(in.b, 1.0);

    let s_prev = dbg_stroke_screen(c_prev);
    let s_a = dbg_stroke_screen(c_a);
    let s_b = dbg_stroke_screen(c_b);
    let current = dbg_safe_dir(s_a, s_b, vec2f(1.0, 0.0));
    let incoming = dbg_safe_dir(s_prev, s_a, current);

    var clip: vec4f;
    var offset: vec2f;
    let half_width = 0.5 * in.width;
    if v < 6u {
        var T = array<f32, 6>(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
        var S = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
        let t = T[v];
        clip = select(c_b, c_a, t < 0.5);
        let normal = vec2f(-current.y, current.x);
        offset = normal * (S[v] * half_width);
    } else {
        let local = i32(v) - 6;
        let wedge = local / 3;
        let corner = local - wedge * 3;
        let turn = atan2(
            incoming.x * current.y - incoming.y * current.x,
            dot(incoming, current),
        );
        clip = c_a;
        if corner == 0 || abs(turn) < 1e-5 {
            offset = vec2f(0.0);
        } else {
            let step = f32(wedge + corner - 1) / f32(STROKE_JOIN_SEGMENTS);
            let incoming_normal = vec2f(-incoming.y, incoming.x);
            let outside = select(1.0, -1.0, turn > 0.0);
            offset = dbg_rotate_2d(incoming_normal * outside, turn * step) * half_width;
        }
    }
    clip = vec4f(clip.xy + offset * (2.0 / dbg_viewport()) * clip.w, clip.z, clip.w);
    out.pos = clip;
    return out;
}
