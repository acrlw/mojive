$input a_position, i_data0, i_data1, i_data2
$output v_world
#include <bgfx_shader.sh>
void main() {
    vec4 p=vec4(a_position,1);
    vec4 world=vec4(dot(i_data0,p),dot(i_data1,p),dot(i_data2,p),1);
    v_world=world.xyz;
    gl_Position=mul(u_viewProj,world);
}
