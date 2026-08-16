#version 330 core
// Connected screen-space stroke. Each instance is one segment plus its previous
// vertex. The body is a quad; the outside of its start corner is
// one small arc fan. This is a real round join, not a circle patched over two
// independent line caps.

#ifndef STROKE_JOIN_SEGMENTS
#define STROKE_JOIN_SEGMENTS 6
#endif

in vec3 in_prev;
in vec3 in_a;
in vec3 in_b;
in vec4 in_color;
in float in_width;

uniform mat4 u_view_proj;
uniform vec2 u_viewport;
uniform float u_alpha;

out vec4 v_color;

vec2 screen_pos(vec4 clip) {
    return (clip.xy / clip.w) * u_viewport;
}

vec2 safe_dir(vec2 a, vec2 b, vec2 fallback) {
    vec2 delta = b - a;
    float len = length(delta);
    return len > 1e-5 ? delta / len : fallback;
}

vec2 rotate_2d(vec2 v, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

void main() {
    v_color = vec4(in_color.rgb, in_color.a * u_alpha);

    vec4 c_prev = u_view_proj * vec4(in_prev, 1.0);
    vec4 c_a = u_view_proj * vec4(in_a, 1.0);
    vec4 c_b = u_view_proj * vec4(in_b, 1.0);

    vec2 s_prev = screen_pos(c_prev);
    vec2 s_a = screen_pos(c_a);
    vec2 s_b = screen_pos(c_b);
    vec2 current = safe_dir(s_a, s_b, vec2(1.0, 0.0));
    vec2 incoming = safe_dir(s_prev, s_a, current);

    vec4 clip;
    vec2 offset;
    float half_width = 0.5 * in_width;
    if (gl_VertexID < 6) {
        const float T[6] = float[6](0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
        const float S[6] = float[6](-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
        float t = T[gl_VertexID];
        clip = t < 0.5 ? c_a : c_b;
        vec2 normal = vec2(-current.y, current.x);
        offset = normal * (S[gl_VertexID] * half_width);
    } else {
        int local = gl_VertexID - 6;
        int wedge = local / 3;
        int corner = local - wedge * 3;
        float turn = atan(
            incoming.x * current.y - incoming.y * current.x,
            dot(incoming, current)
        );
        clip = c_a;
        if (corner == 0 || abs(turn) < 1e-5) {
            offset = vec2(0.0);
        } else {
            float step = float(wedge + corner - 1) / float(STROKE_JOIN_SEGMENTS);
            vec2 incoming_normal = vec2(-incoming.y, incoming.x);
            float outside = turn > 0.0 ? -1.0 : 1.0;
            offset = rotate_2d(incoming_normal * outside, turn * step) * half_width;
        }
    }
    clip.xy += offset * (2.0 / u_viewport) * clip.w;
    gl_Position = clip;
}
