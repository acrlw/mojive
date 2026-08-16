#version 330 core

in vec4 v_color;
in vec3 v_normal;
in vec3 v_view_pos;
in vec3 v_local_position;

uniform float u_mask_radius;

layout(location = 0) out vec4 o_color;

void main() {
    if (u_mask_radius > 0.0 && length(v_local_position) < u_mask_radius) {
        discard;
    }
    vec3 n = normalize(v_normal);
    vec3 to_eye = normalize(-v_view_pos);
    float facing = abs(dot(n, to_eye));
    float shade = 0.72 + 0.28 * facing;
    o_color = vec4(v_color.rgb * shade, v_color.a);
}
