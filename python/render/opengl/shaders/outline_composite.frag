#version 330 core

uniform sampler2D u_alpha;
uniform vec4 u_color;

layout(location = 0) out vec4 o_color;

void main() {
    float alpha = texelFetch(u_alpha, ivec2(gl_FragCoord.xy), 0).r;
    if (alpha <= 0.0) discard;
    o_color = vec4(u_color.rgb, u_color.a * alpha);
}
