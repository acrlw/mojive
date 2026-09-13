#version 330 core
#ifndef SECTOR_SEGMENTS
#define SECTOR_SEGMENTS 32
#endif

in vec3 in_center;
in vec3 in_rot_end;  // Center plus rotation vector
in vec3 in_ref_end;  // Center plus reference direction
in vec4 in_color;
in float in_radius;  // Pixel radius; zero uses the reference-vector length

uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_view_proj;
uniform float u_px_scale;
uniform float u_alpha;

out vec4 v_color;

void main() {
    v_color = vec4(in_color.rgb, in_color.a * u_alpha);

    vec3 c = in_center;
    vec3 rotvec = in_rot_end - c;
    vec3 ref = in_ref_end - c;
    float angle = length(rotvec);
    float ref_len = length(ref);
    if (ref_len < 1e-9) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
    vec3 axis = angle > 1e-9 ? rotvec / angle : vec3(0.0, 0.0, 1.0);
    vec3 dir = ref / ref_len;

    float radius = ref_len;
    if (in_radius > 0.0) {
        float w = (u_proj * (u_view * vec4(c, 1.0))).w;
        radius = in_radius * u_px_scale * w;
    }

    int tri = gl_VertexID / 3;
    int corner = gl_VertexID % 3;
    vec3 p = c;
    if (corner > 0) {
        float t = angle * float(tri + corner - 1) / float(SECTOR_SEGMENTS);
        vec3 v = dir * cos(t) + cross(axis, dir) * sin(t) + axis * dot(axis, dir) * (1.0 - cos(t));
        p = c + v * radius;
    }
    gl_Position = u_view_proj * vec4(p, 1.0);
}
