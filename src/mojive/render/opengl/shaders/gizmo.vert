#version 330 core

in vec3 in_position;
in vec3 in_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_view_proj;
uniform vec4 u_color;

out vec4 v_color;
out vec3 v_normal;
out vec3 v_view_pos;
out vec3 v_local_position;

void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_color = u_color;
    v_local_position = in_position;
    v_normal = mat3(u_model) * in_normal;
    v_view_pos = (u_view * world).xyz;
    vec4 clip = u_view_proj * world;
    clip.z = -0.999 * clip.w + 0.001 * clip.z;
    gl_Position = clip;
}
