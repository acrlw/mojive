#version 330 core

in vec3 in_position;
in vec4 in_model0;
in vec4 in_model1;
in vec4 in_model2;
in vec4 in_model3;
in int in_segment_id;
in int in_segment_type;

uniform mat4 u_view_proj;

flat out ivec2 v_segmentation;

void main() {
    mat4 model = mat4(in_model0, in_model1, in_model2, in_model3);
    v_segmentation = ivec2(in_segment_id, in_segment_type);
    gl_Position = u_view_proj * model * vec4(in_position, 1.0);
}
