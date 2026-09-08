$input v_color0, v_meta0, v_meta1, v_normal, v_depth, v_texcoord0, v_world
#include <bgfx_shader.sh>
#include "dataEncoding.sh"
void main()
{
    gl_FragColor = packWords(v_meta0.xy);
}
