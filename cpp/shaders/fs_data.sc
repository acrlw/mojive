$input v_color0, v_meta0, v_meta1, v_normal, v_depth, v_texcoord0, v_world
#include <bgfx_shader.sh>
#include "dataEncoding.sh"
void main()
{
    gl_FragData[0] = packWords(v_meta0.xy);
    // Four normalized 16-bit channels preserve both signed 32-bit IDs exactly.
    gl_FragData[1] = vec4(v_meta0.zw, v_meta1.xy) / 65535.0;
    gl_FragData[2] = vec4(v_depth, 0.0, 0.0, 1.0);
}
