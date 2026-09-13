#version 330 core

in vec3 in_position;

uniform mat4 u_view_proj;
uniform vec3 u_eye;
uniform float u_distance;

out vec3 v_dir;

void main() {
    v_dir = in_position;
    vec3 world = u_eye + u_distance * in_position;
    gl_Position = u_view_proj * vec4(world, 1.0);
}
