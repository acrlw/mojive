$input v_color0, v_meta0, v_meta1, v_normal, v_depth, v_texcoord0, v_world
#include <bgfx_shader.sh>
vec4 packWords(vec2 words)
{
    words = floor(words + vec2(0.5, 0.5));
    return vec4(mod(words.x,256.0), floor(words.x/256.0), mod(words.y,256.0), floor(words.y/256.0))/255.0;
}
void main()
{
    gl_FragData[0] = packWords(v_meta0.xy);
    gl_FragData[1] = packWords(v_meta0.zw);
    gl_FragData[2] = packWords(v_meta1.xy);
    gl_FragData[3] = vec4(v_depth,0.0,0.0,1.0);
}
