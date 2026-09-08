$input a_position, a_normal
$output v_world, v_normal, v_litCube
#include <bgfx_shader.sh>
uniform vec4 u_gizmoParams;
void main() {
    vec4 world=mul(u_model[0],vec4(a_position,1));
    v_world=a_position;
    v_normal=mul(u_model[0],vec4(a_normal,0)).xyz;
    v_litCube=mul(u_view,world);
    vec4 clip=mul(u_viewProj,world);
    clip.z=.001*clip.z-(u_gizmoParams.y>.5?.999:0.0)*clip.w;
    gl_Position=clip;
}
