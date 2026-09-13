#version 330 core
in vec3 v_world;
uniform vec3 u_light_pos;
uniform float u_light_range;
layout(location = 0) out float o_dist;

void main() {
    float dist = length(v_world - u_light_pos);
    if (u_light_range > 0.0 && dist > u_light_range) discard;
    o_dist = dist;
}
