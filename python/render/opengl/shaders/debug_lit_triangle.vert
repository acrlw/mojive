#version 330 core
in vec3 in_a, in_b, in_c;
in vec3 in_na, in_nb, in_nc;
in vec4 in_color;
uniform mat4 u_view, u_view_proj;
uniform float u_alpha;
uniform bool u_foreground;
out vec4 v_color;
out vec3 v_normal, v_view_pos;
void main() {
    vec3 p = gl_VertexID == 0 ? in_a : gl_VertexID == 1 ? in_b : in_c;
    vec3 n = gl_VertexID == 0 ? in_na : gl_VertexID == 1 ? in_nb : in_nc;
    v_color = vec4(in_color.rgb, in_color.a * u_alpha);
    v_normal = mat3(u_view) * n;
    v_view_pos = (u_view * vec4(p, 1.0)).xyz;
    vec4 clip = u_view_proj * vec4(p, 1.0);
    if (u_foreground) clip.z = -0.99 * clip.w + 0.01 * clip.z;
    gl_Position = clip;
}
