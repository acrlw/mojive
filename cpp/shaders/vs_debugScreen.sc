$input a_position, i_data0, i_data1, i_data2, i_data3
$output v_color0
#include <bgfx_shader.sh>
uniform mat4 u_debugViewProj, u_debugProj;
uniform vec4 u_debugParams, u_debugDepth;
#define u_viewport (u_debugParams.xy)
#define u_px_scale (u_debugParams.z)
#define u_alpha (u_debugParams.w)
#define in_a (vec3(i_data0.x,i_data0.y,i_data0.z))
#define in_b (vec3(i_data0.w,i_data1.x,i_data1.y))
#define in_c (vec3(i_data1.z,i_data1.w,i_data2.x))
#define in_color (vec4(i_data2.y,i_data2.z,i_data2.w,i_data3.x))


vec4 nativeClip(vec4 clip) { if(u_debugDepth.x<.5) clip.z=(clip.z+clip.w)*.5; return clip; }
void main() {
    vec3 vertex = int(a_position.x) == 0 ? in_a : int(a_position.x) == 1 ? in_b : in_c;
    vec2 ndc = vertex.xy / u_viewport * 2.0 - 1.0;
    gl_Position = nativeClip(vec4(ndc.x, -ndc.y, 0.0, 1.0));
    v_color0 = vec4(in_color.rgb, in_color.a * vertex.z * u_alpha);
}
