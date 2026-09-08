$input a_position, a_normal, a_texcoord0, i_data0, i_data1, i_data2, i_data3, i_data4
$output v_color0, v_meta0, v_meta1, v_normal, v_depth, v_texcoord0, v_world
#include <bgfx_shader.sh>
void main()
{
    vec4 position = vec4(a_position, 1.0);
    vec4 world = vec4(dot(i_data0,position), dot(i_data1,position), dot(i_data2,position), 1.0);
    gl_Position = mul(u_viewProj, world);
    v_world = world.xyz;
    v_depth = -mul(u_view, world).z;
    v_color0 = i_data3;
    v_texcoord0 = a_texcoord0 * i_data4.xy + i_data4.zw;
    v_meta0 = i_data3;
    v_meta1 = i_data4;
    v_normal = vec3(dot(cross(i_data1.xyz,i_data2.xyz),a_normal), dot(cross(i_data2.xyz,i_data0.xyz),a_normal), dot(cross(i_data0.xyz,i_data1.xyz),a_normal));
}
