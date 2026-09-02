#version 330 core

in vec3 in_position;
in vec4 in_model0;
in vec4 in_model1;
in vec4 in_model2;
in vec4 in_model3;
in int in_segment_id;
in int in_segment_type;

uniform mat4 u_view_proj;
uniform mat4 u_view;

flat out ivec2 v_segmentation;
out float v_view_depth;

void main() {
    mat4 model = mat4(in_model0, in_model1, in_model2, in_model3);
    vec4 world = model * vec4(in_position, 1.0);
    v_segmentation = ivec2(in_segment_id, in_segment_type);
    v_view_depth = -(u_view * world).z;
    gl_Position = u_view_proj * world;
}
