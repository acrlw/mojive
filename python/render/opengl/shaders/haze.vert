#version 330 core

in vec3 in_haze;

uniform mat4 u_view_proj;
uniform vec3 u_eye;
uniform vec3 u_basis_x;
uniform vec3 u_basis_y;
uniform vec3 u_normal;
uniform vec4 u_geometry;  // skybox distance, elevation, radius, transition height

out float v_alpha;

void main() {
    float layer = in_haze.z;
    float height = layer < 0.5 ? 0.0 : (layer < 1.5 ? u_geometry.w : 1.0);
    float radial = 1.0 - u_geometry.z * (1.0 - height);
    vec3 world = u_eye
        + u_geometry.x * radial * (in_haze.x * u_basis_x + in_haze.y * u_basis_y)
        + u_geometry.y * (height - 1.0) * u_normal;
    v_alpha = (layer > 0.5 && layer < 1.5) ? 1.0 : 0.0;
    gl_Position = u_view_proj * vec4(world, 1.0);
}
