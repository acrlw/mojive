$input a_position, i_data0, i_data1
$output v_color0, v_texcoord0
#include <bgfx_shader.sh>
uniform mat4 u_debugViewProj, u_debugProj;
uniform vec4 u_debugParams, u_debugDepth;
#define u_viewport (u_debugParams.xy)
#define u_px_scale (u_debugParams.z)
#define u_alpha (u_debugParams.w)
#define in_p (vec3(i_data0.x,i_data0.y,i_data0.z))
#define in_color (vec4(i_data0.w,i_data1.x,i_data1.y,i_data1.z))
#define in_radius (i_data1.w)



vec4 nativeClip(vec4 clip) { if(u_debugDepth.x<.5) clip.z=(clip.z+clip.w)*.5; return clip; }
void main() {
    v_color0 = vec4(in_color.rgb, in_color.a * u_alpha);

    const vec2 C[6] = {
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
    };
    v_texcoord0 = C[int(a_position.x)];

    vec3 p = (mul(u_view, vec4(in_p, 1.0))).xyz;
    float w = (mul(u_debugProj, vec4(p, 1.0))).w;
    p.xy += v_texcoord0 * (in_radius * u_px_scale * w);
    gl_Position = nativeClip(mul(u_debugProj, vec4(p, 1.0)));
}
