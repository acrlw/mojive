$input a_position, i_data0, i_data1, i_data2, i_data3, i_data4
$output v_texcoord0, v_color0
#include <bgfx_shader.sh>
uniform mat4 u_debugViewProj, u_debugProj;
uniform vec4 u_debugParams, u_debugDepth;
#define u_viewport (u_debugParams.xy)
#define u_px_scale (u_debugParams.z)
#define u_alpha (u_debugParams.w)
#define in_anchor (vec3(i_data0.x,i_data0.y,i_data0.z))
#define in_offset (vec2(i_data0.w,i_data1.x))
#define in_rect (vec4(i_data1.y,i_data1.z,i_data1.w,i_data2.x))
#define in_uv_rect (vec4(i_data2.y,i_data2.z,i_data2.w,i_data3.x))
#define in_color (vec4(i_data3.y,i_data3.z,i_data3.w,i_data4.x))




vec4 nativeClip(vec4 clip) { if(u_debugDepth.x<.5) clip.z=(clip.z+clip.w)*.5; return clip; }
void main() {
    const vec2 C[6] = {
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    };
    vec2 corner = C[int(a_position.x)];
    vec2 pixel = in_offset + mix(in_rect.xy, in_rect.zw, corner);
    vec4 clip = mul(u_debugViewProj, vec4(in_anchor, 1.0));
    clip.xy += vec2(2.0 * pixel.x / u_viewport.x, -2.0 * pixel.y / u_viewport.y) * clip.w;
    gl_Position = nativeClip(clip);
    v_texcoord0 = mix(in_uv_rect.xy, in_uv_rect.zw, corner);
    v_color0 = vec4(in_color.rgb, in_color.a * u_alpha);
}
