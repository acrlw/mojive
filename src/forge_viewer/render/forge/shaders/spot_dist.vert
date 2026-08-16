#version 330 core

//

in vec3 in_position;
in vec4 in_model0;
in vec4 in_model1;
in vec4 in_model2;
in vec4 in_model3;

uniform mat4 u_view_proj;
uniform vec3 u_light_pos;

out vec3 v_world;

void main() {
    mat4 m = mat4(in_model0, in_model1, in_model2, in_model3);
    vec4 world = m * vec4(in_position, 1.0);
    v_world = world.xyz;
    gl_Position = u_view_proj * world;
}
