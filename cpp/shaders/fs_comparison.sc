#include <bgfx_shader.sh>
SAMPLER2D(s_image, 0);
SAMPLER2D(s_comparison, 1);
uniform vec4 u_comparison;
void main() {
    vec2 uv = gl_FragCoord.xy / u_comparison.yz;
    vec4 value = texture2D(s_image, uv) + texture2D(s_comparison, uv);
    gl_FragColor = vec4(value.rgb / max(value.a, 1e-5), min(value.a, 1.0));
}
