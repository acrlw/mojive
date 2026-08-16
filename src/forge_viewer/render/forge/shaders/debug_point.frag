#version 330 core
//

in vec4 v_color;
in vec2 v_uv;

layout(location = 0) out vec4 o_color;

void main() {
    float r = length(v_uv);
    if (r > 1.0) {
        discard;
    }
    float aa = max(fwidth(r), 1e-4);
    o_color = vec4(v_color.rgb, v_color.a * (1.0 - smoothstep(1.0 - aa, 1.0, r)));
}
