#version 330 core
in vec2 v_uv;
uniform sampler2D u_visual;
uniform sampler2D u_collision;
layout(location = 0) out vec4 o_color;
void main() {
    vec4 value = texture(u_visual, v_uv) + texture(u_collision, v_uv);
    o_color = vec4(value.rgb / max(value.a, 1e-5), min(value.a, 1.0));
}
