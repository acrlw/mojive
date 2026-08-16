#version 330 core

noperspective in vec2 v_pixel;
flat in vec2 v_a;
flat in vec2 v_b;
flat in vec4 v_core_color;
flat in vec4 v_edge_color;
flat in float v_width;
flat in float v_radius;
flat in float v_edge;

layout(location = 0) out vec4 o_color;

float capsule_sdf(vec2 p, vec2 a, vec2 b, float radius) {
    vec2 ab = b - a;
    float d2 = dot(ab, ab);
    float t = d2 > 1e-6 ? clamp(dot(p - a, ab) / d2, 0.0, 1.0) : 0.0;
    return length(p - (a + t * ab)) - radius;
}

void main() {
    float half_width = 0.5 * v_width;
    vec2 ab = v_b - v_a;
    float length_ab = length(ab);
    vec2 direction = length_ab > 1e-4 ? ab / length_ab : vec2(1.0, 0.0);

    // Center the connector cap on the ring. Its rear edge meets the ring's inner edge,
    // preserving the hollow center while the outer half merges with the ring core.
    vec2 link_start = v_a + direction * v_radius;
    float ring = abs(length(v_pixel - v_a) - v_radius) - half_width;
    float link = capsule_sdf(v_pixel, link_start, v_b, half_width);
    float target = length(v_pixel - v_b) - v_radius;
    float shape = min(ring, min(link, target));

    float aa = max(fwidth(shape), 0.65);
    float outer = 1.0 - smoothstep(-aa, aa, shape - v_edge);
    if (outer <= 0.0) {
        discard;
    }
    float core = 1.0 - smoothstep(-aa, aa, shape);
    vec4 color = mix(v_edge_color, v_core_color, core);
    o_color = vec4(color.rgb, color.a * outer);
}
