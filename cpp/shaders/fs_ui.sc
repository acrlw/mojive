$input v_color0, v_texcoord0
#include <bgfx_shader.sh>
SAMPLER2D(s_image,0);
uniform vec4 u_uiTextureInfo;
void main() {
    vec2 uv = v_texcoord0;
    if (u_uiTextureInfo.x > .5) uv.y = 1.0 - uv.y;
    gl_FragColor = v_color0 * texture2D(s_image, uv);
}
