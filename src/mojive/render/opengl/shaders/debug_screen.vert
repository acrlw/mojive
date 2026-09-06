#version 330 core

in vec3 in_a;
in vec3 in_b;
in vec3 in_c;
in vec4 in_color;
uniform vec2 u_viewport;
uniform float u_alpha;
out vec4 v_color;

void main() {
    vec3 vertex = gl_VertexID == 0 ? in_a : gl_VertexID == 1 ? in_b : in_c;
    vec2 ndc = vertex.xy / u_viewport * 2.0 - 1.0;
    gl_Position = vec4(ndc.x, -ndc.y, 0.0, 1.0);
    v_color = vec4(in_color.rgb, in_color.a * vertex.z * u_alpha);
}
