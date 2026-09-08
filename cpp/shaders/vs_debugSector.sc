$input a_position, i_data0, i_data1, i_data2, i_data3
$output v_color0
#include <bgfx_shader.sh>
uniform mat4 u_debugViewProj, u_debugProj;
uniform vec4 u_debugParams, u_debugDepth;
#define u_viewport (u_debugParams.xy)
#define u_px_scale (u_debugParams.z)
#define u_alpha (u_debugParams.w)
#define in_center (vec3(i_data0.x,i_data0.y,i_data0.z))
#define in_rot_end (vec3(i_data0.w,i_data1.x,i_data1.y))
#define in_ref_end (vec3(i_data1.z,i_data1.w,i_data2.x))
#define in_color (vec4(i_data2.y,i_data2.z,i_data2.w,i_data3.x))
#define in_radius (i_data3.y)
#ifndef SECTOR_SEGMENTS
#define SECTOR_SEGMENTS 32
#endif




vec4 nativeClip(vec4 clip) { if(u_debugDepth.x<.5) clip.z=(clip.z+clip.w)*.5; return clip; }
void main() {
    v_color0 = vec4(in_color.rgb, in_color.a * u_alpha);

    vec3 c = in_center;
    vec3 rotvec = in_rot_end - c;
    vec3 ref = in_ref_end - c;
    float angle = length(rotvec);
    float ref_len = length(ref);
    if (ref_len < 1e-9) {
        gl_Position = nativeClip(vec4(0.0, 0.0, 2.0, 1.0));
        return;
    }
    vec3 axis = angle > 1e-9 ? rotvec / angle : vec3(0.0, 0.0, 1.0);
    vec3 dir = ref / ref_len;

    float radius = ref_len;
    if (in_radius > 0.0) {
        float w = mul(u_debugProj, mul(u_view, vec4(c,1.0))).w;
        radius = in_radius * u_px_scale * w;
    }

    int tri = int(a_position.x) / 3;
    int corner = int(a_position.x) % 3;
    vec3 p = c;
    if (corner > 0) {
        float t = angle * float(tri + corner - 1) / float(SECTOR_SEGMENTS);
        vec3 v = dir * cos(t) + cross(axis, dir) * sin(t) + axis * dot(axis, dir) * (1.0 - cos(t));
        p = c + v * radius;
    }
    gl_Position = nativeClip(mul(u_debugViewProj, vec4(p, 1.0)));
}
