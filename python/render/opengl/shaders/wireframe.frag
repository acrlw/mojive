#version 330 core
#include "common.glsl"
#include "coverage.glsl"

in WireData { vec4 color; } wire;
in vec3 v_bary;
uniform float u_wire_width;
layout(location = 0) out vec4 o_color;

void main() {
    float distanceToEdge = min(v_bary.x, min(v_bary.y, v_bary.z));
    float width = max(fwidth(distanceToEdge), 1e-6) * u_wire_width;
    if (distanceToEdge > width) discard;
    float opacity = coverageAlpha(wire.color.a, gl_FragCoord.xy);
    o_color = vec4(gamma_encode(wire.color.rgb), opacity);
}
