$input v_color0, v_texcoord0
#include <bgfx_shader.sh>
SAMPLER2D(s_image,0);
void main() { gl_FragColor = v_color0 * texture2D(s_image,v_texcoord0); }
