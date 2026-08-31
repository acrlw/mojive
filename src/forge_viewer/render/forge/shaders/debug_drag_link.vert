#version 330 core

in vec3 in_a;
in vec3 in_b;
in vec4 in_core_color;
in vec4 in_edge_color;
in float in_width;
in float in_radius;
in float in_edge;

uniform mat4 u_view_proj;
uniform vec2 u_viewport;
uniform float u_alpha;

noperspective out vec2 v_pixel;
flat out vec2 v_a;
flat out vec2 v_b;
flat out vec4 v_core_color;
flat out vec4 v_edge_color;
flat out float v_width;
flat out float v_radius;
flat out float v_edge;

void main() {
    vec4 clip_a = u_view_proj * vec4(in_a, 1.0);
    vec4 clip_b = u_view_proj * vec4(in_b, 1.0);
    v_a = (clip_a.xy / clip_a.w * 0.5 + 0.5) * u_viewport;
    v_b = (clip_b.xy / clip_b.w * 0.5 + 0.5) * u_viewport;
    v_core_color = vec4(in_core_color.rgb, in_core_color.a * u_alpha);
    v_edge_color = vec4(in_edge_color.rgb, in_edge_color.a * u_alpha);
    v_width = in_width;
    v_radius = in_radius;
    v_edge = in_edge;

    // The hollow start ring extends half a core stroke beyond its radius.
    // Include that stroke as well as the contrast edge and AA guard in the
    // primitive quad; otherwise large UI scales clip the ring at the quad.
    float pad = in_radius + 0.5 * in_width + in_edge + 2.0;
    vec2 lo = min(v_a, v_b) - pad;
    vec2 hi = max(v_a, v_b) + pad;
    const vec2 C[6] = vec2[6](
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    );
    v_pixel = mix(lo, hi, C[gl_VertexID]);
    vec2 ndc = v_pixel / u_viewport * 2.0 - 1.0;
    gl_Position = vec4(ndc * clip_a.w, clip_a.z, clip_a.w);
}
