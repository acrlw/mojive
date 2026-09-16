$input v_color0, v_meta0, v_meta1, v_normal, v_depth, v_texcoord0, v_world
#include <bgfx_shader.sh>
#include "coverage.sh"
void main()
{
    coverageAlpha(v_meta1.z, gl_FragCoord.xy);
    gl_FragColor = vec4(v_meta0.zw, v_meta1.xy) / 65535.0;
}
