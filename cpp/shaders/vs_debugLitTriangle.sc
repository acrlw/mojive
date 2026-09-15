$input a_position, i_data0, i_data1, i_data2, i_data3, i_data4, i_data5
$output v_color0, v_normal, v_view_pos
#include <bgfx_shader.sh>
uniform mat4 u_debugViewProj;
uniform vec4 u_debugDepth;
void main() {
    vec3 a = i_data0.xyz;
    vec3 b = vec3(i_data0.w, i_data1.xy);
    vec3 c = vec3(i_data1.zw, i_data2.x);
    vec3 na = i_data2.yzw;
    vec3 nb = i_data3.xyz;
    vec3 nc = vec3(i_data3.w, i_data4.xy);
    int index = int(a_position.x);
    vec3 p = index == 0 ? a : index == 1 ? b : c;
    vec3 n = index == 0 ? na : index == 1 ? nb : nc;
    v_color0 = vec4(i_data4.zw, i_data5.xy);
    v_normal = mul(u_view, vec4(n, 0.0)).xyz;
    v_view_pos = mul(u_view, vec4(p, 1.0)).xyz;
    vec4 clip = mul(u_debugViewProj, vec4(p, 1.0));
    if (u_debugDepth.y > .5) clip.z = -.99 * clip.w + .01 * clip.z;
    if (u_debugDepth.x < .5) clip.z = (clip.z + clip.w) * .5;
    gl_Position = clip;
}
