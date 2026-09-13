#version 330 core
#include "common.glsl"

in vec3 v_dir;

layout(location = 0) out vec4 o_color;

uniform samplerCube u_skybox;
uniform float u_exposure;
uniform int u_tonemap;
uniform int u_classic_lighting;

void main() {
    vec3 d = normalize(v_dir);

    vec3 c = texture(u_skybox, vec3(d.x, d.z, -d.y)).rgb;
    vec3 rgb = u_classic_lighting != 0
        ? linear_to_srgb(c)
        : finish_color(c, u_exposure, u_tonemap != 0);
    o_color = vec4(rgb, 1.0);
}
