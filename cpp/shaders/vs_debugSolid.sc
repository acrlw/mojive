$input a_position, a_normal, i_data0, i_data1, i_data2, i_data3, i_data4
$output v_color0, v_normal, v_view_pos
#include <bgfx_shader.sh>
uniform mat4 u_debugViewProj, u_debugProj;
uniform vec4 u_debugParams, u_debugDepth;
#define u_viewport (u_debugParams.xy)
#define u_px_scale (u_debugParams.z)
#define u_alpha (u_debugParams.w)
#define in_model0 (vec4(i_data0.x,i_data0.y,i_data0.z,i_data0.w))
#define in_model1 (vec4(i_data1.x,i_data1.y,i_data1.z,i_data1.w))
#define in_model2 (vec4(i_data2.x,i_data2.y,i_data2.z,i_data2.w))
#define in_model3 (vec4(i_data3.x,i_data3.y,i_data3.z,i_data3.w))
#define in_color (vec4(i_data4.x,i_data4.y,i_data4.z,i_data4.w))




vec4 nativeClip(vec4 clip) { if(u_debugDepth.x<.5) clip.z=(clip.z+clip.w)*.5; return clip; }
void main() {
    mat4 model = mtxFromCols(in_model0, in_model1, in_model2, in_model3);
    vec4 world = mul(model, vec4(a_position, 1.0));
    v_color0 = vec4(in_color.rgb, in_color.a * u_alpha);
    v_normal = mul(model, vec4(a_normal,0)).xyz;
    v_view_pos = (mul(u_view, world)).xyz;
    gl_Position = nativeClip(mul(u_debugViewProj, world));
}
