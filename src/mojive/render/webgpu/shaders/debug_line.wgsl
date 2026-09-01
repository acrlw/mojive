// Pixel-wide line segments and arrows, expanded from @builtin(vertex_index)
// into screen-space triangles.  Port of opengl's debug_line.vert/debug_line.frag;
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

struct DebugArrowOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
    @location(1) arrow_pos: vec2f,
    @location(2) @interpolate(flat) arrow_shape: vec4f,
};

// Matches ARROW_CORNER_RADIUS_PT / (2 * AXIS_SHAFT_HALF_PT).
const DEBUG_ARROW_CORNER_RADIUS_RATIO: f32 = 0.5 / 4.4;

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
    let start_t = min(max(in.start_mask, 0.0), length_px) / length_px;
    var T = array<f32, 6>(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
    var S = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let t = mix(start_t, 1.0, T[v]);
    let clip = mix(c_a, c_b, t);
    let screen = mix(s_a, s_b, t) + side * (S[v] * 0.5 * in.width);
    out.pos = dbg_place(clip, screen);
    return out;
}

@vertex
fn vs_debug_arrow(in: DebugLineIn, @builtin(vertex_index) v: u32) -> DebugArrowOut {
    var out: DebugArrowOut;
    out.color = vec4f(in.color.rgb, in.color.a * dbg_alpha());

    let c_a = dbg.view_proj * vec4f(in.a, 1.0);
    let c_b = dbg.view_proj * vec4f(in.b, 1.0);
    let s_a = dbg_screen_pos(c_a);
    let s_b = dbg_screen_pos(c_b);
    let delta = s_b - s_a;
    let length_px = length(delta);
    if length_px < 1e-5 {
        out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
        out.arrow_pos = vec2f(0.0);
        out.arrow_shape = vec4f(0.0);
        return out;
    }

    let direction = delta / length_px;
    let side = vec2f(-direction.y, direction.x);
    let head = min(in.head, length_px * 0.42);
    let start_t = min(max(in.start_mask, 0.0), length_px) / length_px;
    let neck_t = (length_px - head) / length_px;

    var I = array<i32, 15>(0, 1, 6, 1, 5, 6, 2, 3, 1, 1, 3, 5, 5, 3, 4);
    let p = I[v];
    let base_clip = mix(c_a, c_b, neck_t);
    let start_clip = mix(c_a, c_b, start_t);
    let start = mix(s_a, s_b, start_t);
    let base = s_b - direction * head;
    let shaft = 0.5 * in.width;
    let wing = in.head * (7.0 / 12.0);
    var local: vec2f;
    if p == 3 {
        local = vec2f(head, 0.0);
        out.pos = c_b;
    } else if p == 0 || p == 6 {
        let offset = select(shaft, -shaft, p == 0);
        local = vec2f(-length(start - base), offset);
        out.pos = dbg_place(start_clip, start + side * offset);
    } else {
        var offset = shaft;
        if p == 1 {
            offset = -shaft;
        } else if p == 2 {
            offset = -wing;
        } else if p == 4 {
            offset = wing;
        }
        local = vec2f(0.0, offset);
        out.pos = dbg_place(base_clip, base + side * offset);
    }
    out.arrow_pos = local;
    out.arrow_shape = vec4f(
        head,
        shaft,
        wing,
        in.width * DEBUG_ARROW_CORNER_RADIUS_RATIO,
    );
    return out;
}

fn dbg_rounded_corner_coverage(
    point: vec2f,
    corner: vec2f,
    previous: vec2f,
    next: vec2f,
    radius: f32,
) -> f32 {
    let incoming_edge = previous - corner;
    let outgoing_edge = next - corner;
    let incoming_length = length(incoming_edge);
    let outgoing_length = length(outgoing_edge);
    let incoming = incoming_edge / max(incoming_length, 1e-6);
    let outgoing = outgoing_edge / max(outgoing_length, 1e-6);
    let half_angle = 0.5 * acos(clamp(dot(incoming, outgoing), -1.0, 1.0));
    let tan_half = max(tan(half_angle), 1e-6);
    let tangent = min(
        radius / tan_half,
        0.45 * min(incoming_length, outgoing_length),
    );
    let effective_radius = tangent * tan_half;
    let bisector = normalize(incoming + outgoing);
    let center = corner + bisector * (
        effective_radius / max(sin(half_angle), 1e-6)
    );
    let relative = point - corner;
    let circle_distance = effective_radius - distance(point, center);
    let aa = max(fwidth(circle_distance), 0.5);
    let rounded = smoothstep(-0.5 * aa, 0.5 * aa, circle_distance);
    let near_corner = dot(relative, incoming) < tangent
        && dot(relative, outgoing) < tangent;
    return select(1.0, rounded, near_corner);
}

fn dbg_arrow_coverage(point: vec2f, shape: vec4f) -> f32 {
    let head = shape.x;
    let shaft = shape.y;
    let wing = shape.z;
    let radius = shape.w;
    let lower_shaft = vec2f(0.0, -shaft);
    let lower_wing = vec2f(0.0, -wing);
    let tip = vec2f(head, 0.0);
    let upper_wing = vec2f(0.0, wing);
    let upper_shaft = vec2f(0.0, shaft);
    var coverage = dbg_rounded_corner_coverage(
        point, lower_wing, lower_shaft, tip, radius,
    );
    coverage = min(
        coverage,
        dbg_rounded_corner_coverage(point, tip, lower_wing, upper_wing, radius),
    );
    coverage = min(
        coverage,
        dbg_rounded_corner_coverage(point, upper_wing, tip, upper_shaft, radius),
    );
    return coverage;
}

@fragment
fn fs_debug_line(in: DebugLineOut) -> @location(0) vec4f {
    return in.color;
}

@fragment
fn fs_debug_arrow(in: DebugArrowOut) -> @location(0) vec4f {
    let coverage = dbg_arrow_coverage(in.arrow_pos, in.arrow_shape);
    return vec4f(in.color.rgb, in.color.a * coverage);
}
