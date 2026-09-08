$input a_position
$output v_world
#include <bgfx_shader.sh>
uniform vec4 u_skyEyeDistance;
void main() {
    v_world=a_position;
    gl_Position=mul(u_viewProj,vec4(u_skyEyeDistance.xyz+u_skyEyeDistance.w*a_position,1));
}
