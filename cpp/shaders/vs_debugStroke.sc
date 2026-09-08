$input a_position, i_data0, i_data1, i_data2, i_data3
$output v_color0
#include <bgfx_shader.sh>
uniform mat4 u_debugViewProj, u_debugProj;
uniform vec4 u_debugParams, u_debugDepth;
#define u_viewport (u_debugParams.xy)
#define u_px_scale (u_debugParams.z)
#define u_alpha (u_debugParams.w)
#define in_prev (vec3(i_data0.x,i_data0.y,i_data0.z))
#define in_a (vec3(i_data0.w,i_data1.x,i_data1.y))
#define in_b (vec3(i_data1.z,i_data1.w,i_data2.x))
#define in_color (vec4(i_data2.y,i_data2.z,i_data2.w,i_data3.x))
#define in_width (i_data3.y)
// Connected screen-space stroke. Each instance is one segment plus its previous
// vertex. The body is a quad; the outside of its start corner is
// one small arc fan. This is a real round join, not a circle patched over two
// independent line caps.

#ifndef STROKE_JOIN_SEGMENTS
#define STROKE_JOIN_SEGMENTS 6
#endif




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

vec4 nativeClip(vec4 clip) { if(u_debugDepth.x<.5) clip.z=(clip.z+clip.w)*.5; return clip; }
void main() {
    v_color0 = vec4(in_color.rgb, in_color.a * u_alpha);

    vec4 c_prev = mul(u_debugViewProj, vec4(in_prev, 1.0));
    vec4 c_a = mul(u_debugViewProj, vec4(in_a, 1.0));
    vec4 c_b = mul(u_debugViewProj, vec4(in_b, 1.0));

    vec2 s_prev = screen_pos(c_prev);
    vec2 s_a = screen_pos(c_a);
    vec2 s_b = screen_pos(c_b);
    vec2 current = safe_dir(s_a, s_b, vec2(1.0, 0.0));
    vec2 incoming = safe_dir(s_prev, s_a, current);

    vec4 clip;
    vec2 offset;
    float half_width = 0.5 * in_width;
    if (int(a_position.x) < 6) {
        const float T[6] = {0.0, 1.0, 0.0, 1.0, 1.0, 0.0};
        const float S[6] = {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0};
        float t = T[int(a_position.x)];
        clip = t < 0.5 ? c_a : c_b;
        vec2 normal = vec2(-current.y, current.x);
        offset = normal * (S[int(a_position.x)] * half_width);
    } else {
        int local = int(a_position.x) - 6;
        int wedge = local / 3;
        int corner = local - wedge * 3;
        float turn = atan2(
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
    gl_Position = nativeClip(clip);
}
