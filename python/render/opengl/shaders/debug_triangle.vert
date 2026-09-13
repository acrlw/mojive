#version 330 core

in vec3 in_a;
in vec3 in_b;
in vec3 in_c;
in vec4 in_color;
uniform mat4 u_view_proj;
uniform float u_alpha;
out vec4 v_color;

void main() {
    vec3 vertex = gl_VertexID == 0 ? in_a : gl_VertexID == 1 ? in_b : in_c;
    gl_Position = u_view_proj * vec4(vertex, 1.0);
    v_color = vec4(in_color.rgb, in_color.a * u_alpha);
}
