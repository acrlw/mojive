#version 330 core
in vec3 in_position;
in vec3 in_normal;

in vec4 in_model0;
in vec4 in_model1;
in vec4 in_model2;
in vec4 in_model3;
in vec4 in_color;

uniform mat4 u_view;
uniform mat4 u_view_proj;
uniform float u_alpha;

out vec4 v_color;
out vec3 v_normal;
out vec3 v_view_pos;

void main() {
    mat4 model = mat4(in_model0, in_model1, in_model2, in_model3);
    vec4 world = model * vec4(in_position, 1.0);
    v_color = vec4(in_color.rgb, in_color.a * u_alpha);
    mat3 normal_matrix = mat3(cross(model[1].xyz, model[2].xyz),
                              cross(model[2].xyz, model[0].xyz),
                              cross(model[0].xyz, model[1].xyz));
    v_normal = mat3(u_view) * normal_matrix * in_normal;
    v_view_pos = (u_view * world).xyz;
    gl_Position = u_view_proj * world;
}
