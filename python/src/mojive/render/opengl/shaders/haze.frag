#version 330 core
#include "common.glsl"

in float v_alpha;

layout(location = 0) out vec4 o_color;

uniform vec3 u_color;
uniform float u_exposure;
uniform int u_tonemap;
uniform int u_classic_lighting;

void main() {
    vec3 color = u_classic_lighting != 0
        ? u_color
        : finish_color(srgb_to_linear(u_color), u_exposure, u_tonemap != 0);
    o_color = vec4(color, v_alpha);
}
