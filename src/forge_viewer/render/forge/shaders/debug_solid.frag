#version 330 core
//
//

in vec4 v_color;
in vec3 v_normal;
in vec3 v_view_pos;

layout(location = 0) out vec4 o_color;

void main() {
    vec3 n = normalize(v_normal);
    vec3 to_eye = normalize(-v_view_pos);
    float facing = abs(dot(n, to_eye));
    o_color = vec4(v_color.rgb * (0.55 + 0.45 * facing), v_color.a);
}
