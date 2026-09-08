$input v_color0, v_meta0, v_meta1, v_normal, v_depth
#include <bgfx_shader.sh>
void main()
{
    float shade = 0.3 + 0.7 * abs(dot(normalize(v_normal), normalize(vec3(0.4, -0.5, 0.8))));
    gl_FragColor = vec4(v_color0.rgb * shade, v_color0.a);
}
