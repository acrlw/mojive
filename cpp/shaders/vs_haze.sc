$input a_position
$output v_color0
#include <bgfx_shader.sh>
uniform vec4 u_skyEyeDistance, u_hazeGeometry, u_hazeBasisX, u_hazeBasisY, u_hazeNormal;
void main() {
    float layer=a_position.z;
    float height=layer<.5?0.0:layer<1.5?u_hazeGeometry.w:1.0;
    float radial=1.0-u_hazeGeometry.z*(1.0-height);
    vec3 world=u_skyEyeDistance.xyz+u_hazeGeometry.x*radial*(a_position.x*u_hazeBasisX.xyz+a_position.y*u_hazeBasisY.xyz)+u_hazeGeometry.y*(height-1.0)*u_hazeNormal.xyz;
    v_color0=vec4(0,0,0,layer>.5&&layer<1.5?1.0:0.0);
    gl_Position=mul(u_viewProj,vec4(world,1));
}
