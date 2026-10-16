$input a_position, a_texcoord1, i_data0, i_data1, i_data2, i_data3
$output v_color0, v_bary
#include <bgfx_shader.sh>
void main()
{
    vec4 position = vec4(a_position, 1.0);
    vec4 world = vec4(dot(i_data0, position), dot(i_data1, position), dot(i_data2, position), 1.0);
    gl_Position = mul(u_viewProj, world);
    v_bary = a_texcoord1;
    v_color0 = i_data3;
}
