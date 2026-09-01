#version 330 core

in vec3 in_position;
in vec4 in_model0;
in vec4 in_model1;
in vec4 in_model2;
in vec4 in_model3;
in uint in_object_id;

uniform mat4 u_view_proj;

flat out uint v_id;

void main() {
    mat4 m = mat4(in_model0, in_model1, in_model2, in_model3);
    v_id = in_object_id;
    gl_Position = u_view_proj * m * vec4(in_position, 1.0);
}
