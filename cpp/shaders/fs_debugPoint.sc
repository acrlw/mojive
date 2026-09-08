$input v_color0, v_texcoord0
#include <bgfx_shader.sh>



void main() {
    float r = length(v_texcoord0);
    if (r > 1.0) {
        discard;
    }
    float aa = max(fwidth(r), 1e-4);
    gl_FragColor = vec4(v_color0.rgb, v_color0.a * (1.0 - smoothstep(1.0 - aa, 1.0, r)));
}
