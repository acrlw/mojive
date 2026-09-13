#version 330 core

uniform sampler2D u_atlas;

in vec2 v_uv;
in vec4 v_color;

layout(location = 0) out vec4 o_color;

void main() {
    float coverage = texture(u_atlas, v_uv).r;
    o_color = vec4(v_color.rgb, v_color.a * coverage);
}
