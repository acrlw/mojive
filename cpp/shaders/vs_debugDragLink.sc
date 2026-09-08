$input a_position, i_data0, i_data1, i_data2, i_data3, i_data4
$output v_pixel, v_a, v_b, v_core_color, v_edge_color, v_width, v_radius, v_edge, v_smoothing
#include <bgfx_shader.sh>
uniform mat4 u_debugViewProj, u_debugProj;
uniform vec4 u_debugParams, u_debugDepth;
#define u_viewport (u_debugParams.xy)
#define u_px_scale (u_debugParams.z)
#define u_alpha (u_debugParams.w)
#define in_a (vec3(i_data0.x,i_data0.y,i_data0.z))
#define in_b (vec3(i_data0.w,i_data1.x,i_data1.y))
#define in_core_color (vec4(i_data1.z,i_data1.w,i_data2.x,i_data2.y))
#define in_edge_color (vec4(i_data2.z,i_data2.w,i_data3.x,i_data3.y))
#define in_width (i_data3.z)
#define in_radius (i_data3.w)
#define in_edge (i_data4.x)
#define in_smoothing (i_data4.y)




vec4 nativeClip(vec4 clip) { if(u_debugDepth.x<.5) clip.z=(clip.z+clip.w)*.5; return clip; }
void main() {
    vec4 clip_a = mul(u_debugViewProj, vec4(in_a, 1.0));
    vec4 clip_b = mul(u_debugViewProj, vec4(in_b, 1.0));
    v_a = (clip_a.xy / clip_a.w * 0.5 + 0.5) * u_viewport;
    v_b = (clip_b.xy / clip_b.w * 0.5 + 0.5) * u_viewport;
    v_core_color = vec4(in_core_color.rgb, in_core_color.a * u_alpha);
    v_edge_color = vec4(in_edge_color.rgb, in_edge_color.a * u_alpha);
    v_width = in_width;
    v_radius = in_radius;
    v_edge = in_edge;
    v_smoothing = in_smoothing;

    // The hollow start ring extends half a core stroke beyond its radius.
    // Include that stroke as well as the contrast edge and AA guard in the
    // primitive quad; otherwise large UI scales clip the ring at the quad.
    float pad = in_radius + 0.5 * in_width + in_edge + in_radius * in_smoothing * 0.3125 + 2.0;
    vec2 lo = min(v_a, v_b) - pad;
    vec2 hi = max(v_a, v_b) + pad;
    const vec2 C[6] = {
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    };
    v_pixel = mix(lo, hi, C[int(a_position.x)]);
    vec2 ndc = v_pixel / u_viewport * 2.0 - 1.0;
    gl_Position = nativeClip(vec4(ndc * clip_a.w, clip_a.z, clip_a.w));
}
