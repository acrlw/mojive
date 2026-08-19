// Pixel-wide line segments and arrows, expanded from @builtin(vertex_index)
// into screen-space triangles.  Port of forge's debug_line.vert/debug_line.frag;
// the fragment is shared with debug_stroke.wgsl.  One instance is one record:
// a(3) b(3) rgba(4) width_px(1) head_px(1) start_mask_px(1).

struct DebugLineIn {
    @location(0) a: vec3f,
    @location(1) b: vec3f,
    @location(2) color: vec4f,
    @location(3) width: f32,       // full shaft width in pixels
    @location(4) head: f32,        // arrowhead length in pixels; zero draws a line
    @location(5) start_mask: f32,  // hidden center-shell radius in pixels
};

struct DebugLineOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
};

@vertex
fn vs_debug_line(in: DebugLineIn, @builtin(vertex_index) v: u32) -> DebugLineOut {
    var out: DebugLineOut;
    out.color = vec4f(in.color.rgb, in.color.a * dbg_alpha());

    let c_a = dbg.view_proj * vec4f(in.a, 1.0);
    let c_b = dbg.view_proj * vec4f(in.b, 1.0);
    let s_a = dbg_screen_pos(c_a);
    let s_b = dbg_screen_pos(c_b);
    let delta = s_b - s_a;
    let length_px = length(delta);
    if length_px < 1e-5 {
        out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let direction = delta / length_px;
    let side = vec2f(-direction.y, direction.x);
    let head = select(0.0, min(in.head, length_px * 0.42), in.head > 0.0);
    let start_t = min(max(in.start_mask, 0.0), length_px) / length_px;
    let neck_t = (length_px - head) / length_px;

    if in.head <= 0.0 {
        if v >= 6u {
            out.pos = c_b;
            return out;
        }
        var T = array<f32, 6>(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
        var S = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
        let t = mix(start_t, 1.0, T[v]);
        let clip = mix(c_a, c_b, t);
        let screen = mix(s_a, s_b, t) + side * (S[v] * 0.5 * in.width);
        out.pos = dbg_place(clip, screen);
        return out;
    }

    var I = array<i32, 15>(0, 1, 6, 1, 5, 6, 2, 3, 1, 1, 3, 5, 5, 3, 4);
    let p = I[v];
    let base_clip = mix(c_a, c_b, neck_t);
    let start_clip = mix(c_a, c_b, start_t);
    let start = mix(s_a, s_b, start_t);
    let base = s_b - direction * head;
    let shaft = 0.5 * in.width;
    let wing = in.head * (7.0 / 12.0);
    if p == 3 {
        out.pos = c_b;
    } else if p == 0 || p == 6 {
        out.pos = dbg_place(start_clip, start + side * select(shaft, -shaft, p == 0));
    } else {
        var offset = shaft;
        if p == 1 {
            offset = -shaft;
        } else if p == 2 {
            offset = -wing;
        } else if p == 4 {
            offset = wing;
        }
        out.pos = dbg_place(base_clip, base + side * offset);
    }
    return out;
}

@fragment
fn fs_debug_line(in: DebugLineOut) -> @location(0) vec4f {
    return in.color;
}
