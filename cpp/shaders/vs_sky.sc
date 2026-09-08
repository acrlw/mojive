$input a_position
$output v_world
#include <bgfx_shader.sh>
uniform mat4 u_skyInverse;
void main() {
    vec4 nearPoint=mul(u_skyInverse,vec4(a_position.xy,-1,1));
    vec4 farPoint=mul(u_skyInverse,vec4(a_position.xy,1,1));
    v_world=farPoint.xyz/farPoint.w-nearPoint.xyz/nearPoint.w;
    gl_Position=vec4(a_position.xy,1,1);
}
