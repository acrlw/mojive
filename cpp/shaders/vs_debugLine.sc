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
#define in_color (vec4(i_data1.z,i_data1.w,i_data2.x,i_data2.y))
#define in_width (i_data2.z)
#define in_head (i_data2.w)
#define in_start_mask (i_data3.x)


vec2 screen_pos(vec4 clip) {
    vec2 ndc = clip.xy / clip.w;
    return (ndc * 0.5 + 0.5) * u_viewport;
}

vec4 place(vec4 clip, vec2 screen) {
    clip.xy = (screen / u_viewport * 2.0 - 1.0) * clip.w;
    return clip;
}

void main() {
    vec4 a=mul(u_debugViewProj,vec4(in_a,1));
    vec4 b=mul(u_debugViewProj,vec4(in_b,1));
    vec2 sa=screen_pos(a),sb=screen_pos(b),delta=sb-sa;
    float lengthPx=length(delta);
    vec2 side=vec2(-delta.y,delta.x)/max(lengthPx,1e-5);
    const float t[6]={0,1,0,1,1,0};
    const float signSide[6]={-1,-1,1,-1,1,1};
    int vertex=int(a_position.x);
    vec4 clip=place(mix(a,b,t[vertex]),mix(sa,sb,t[vertex])+side*(signSide[vertex]*.5*in_width));
    if(lengthPx<1e-5)clip=vec4(0,0,2,1);
    if(u_debugDepth.x<.5)clip.z=(clip.z+clip.w)*.5;
    gl_Position=clip;
    v_color0=vec4(in_color.rgb,in_color.a*u_alpha);
}
