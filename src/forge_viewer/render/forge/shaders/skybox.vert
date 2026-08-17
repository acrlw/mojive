#version 330 core

uniform mat4 u_inv_view_proj;

out vec3 v_dir;

void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vec2 ndc = p * 2.0 - 1.0;

    vec4 near = u_inv_view_proj * vec4(ndc, -1.0, 1.0);
    vec4 far = u_inv_view_proj * vec4(ndc, 1.0, 1.0);
    v_dir = far.xyz / far.w - near.xyz / near.w;

    gl_Position = vec4(ndc, 1.0, 1.0);
}
