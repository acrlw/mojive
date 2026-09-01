#version 330 core

in vec3 in_a;
in vec3 in_b;
in vec4 in_color;
in float in_width;      // Full shaft width in pixels
in float in_head;       // Arrowhead length in pixels; zero draws a line
in float in_start_mask; // Hidden center-shell radius in pixels

uniform mat4 u_view_proj;
uniform vec2 u_viewport;
uniform float u_alpha;

out vec4 v_color;
#ifdef ROUND_ARROW
out vec2 v_arrow_pos;
flat out vec4 v_arrow_shape; // head length, shaft half-width, wing half-width, corner radius
#endif

vec2 screen_pos(vec4 clip) {
    vec2 ndc = clip.xy / clip.w;
    return (ndc * 0.5 + 0.5) * u_viewport;
}

vec4 place(vec4 clip, vec2 screen) {
    clip.xy = (screen / u_viewport * 2.0 - 1.0) * clip.w;
    return clip;
}

void main() {
    v_color = vec4(in_color.rgb, in_color.a * u_alpha);
#ifdef ROUND_ARROW
    v_arrow_pos = vec2(0.0);
    v_arrow_shape = vec4(0.0);
#endif

    vec4 c_a = u_view_proj * vec4(in_a, 1.0);
    vec4 c_b = u_view_proj * vec4(in_b, 1.0);
    vec2 s_a = screen_pos(c_a);
    vec2 s_b = screen_pos(c_b);
    vec2 delta = s_b - s_a;
    float length_px = length(delta);
    if (length_px < 1e-5) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    vec2 direction = delta / length_px;
    vec2 side = vec2(-direction.y, direction.x);
    float head = in_head > 0.0 ? min(in_head, length_px * 0.42) : 0.0;
    float start_t = min(max(in_start_mask, 0.0), length_px) / length_px;
    float neck_t = (length_px - head) / length_px;

    int v = gl_VertexID;
    if (in_head <= 0.0) {
        if (v >= 6) {
            gl_Position = c_b;
            return;
        }
        const float T[6] = float[6](0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
        const float S[6] = float[6](-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
        float t = mix(start_t, 1.0, T[v]);
        vec4 clip = mix(c_a, c_b, t);
        vec2 screen = mix(s_a, s_b, t) + side * (S[v] * 0.5 * in_width);
        gl_Position = place(clip, screen);
        return;
    }

    const int I[15] = int[15](
        0, 1, 6,  1, 5, 6,
        2, 3, 1,  1, 3, 5,  5, 3, 4
    );
    int p = I[v];
    vec4 base_clip = mix(c_a, c_b, neck_t);
    vec4 start_clip = mix(c_a, c_b, start_t);
    vec2 start = mix(s_a, s_b, start_t);
    vec2 base = s_b - direction * head;
    float shaft = 0.5 * in_width;
    float wing = in_head * (7.0 / 12.0);
    vec2 local;
    if (p == 3) {
        local = vec2(head, 0.0);
        gl_Position = c_b;
    } else if (p == 0 || p == 6) {
        float offset = p == 0 ? -shaft : shaft;
        local = vec2(-length(start - base), offset);
        gl_Position = place(start_clip, start + side * offset);
    } else {
        float offset = p == 1 ? -shaft : p == 2 ? -wing : p == 4 ? wing : shaft;
        local = vec2(0.0, offset);
        gl_Position = place(base_clip, base + side * offset);
    }
#ifdef ROUND_ARROW
    v_arrow_pos = local;
    v_arrow_shape = vec4(
        head,
        shaft,
        wing,
        in_width * ARROW_CORNER_RADIUS_RATIO
    );
#endif
}
