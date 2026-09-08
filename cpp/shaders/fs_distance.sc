$input v_world
#include <bgfx_shader.sh>
uniform vec4 u_shadowLightPosition;
void main() {
    float d=length(v_world-u_shadowLightPosition.xyz);
    if(u_shadowLightPosition.w>0 && d>u_shadowLightPosition.w) discard;
    gl_FragColor=vec4(d,0,0,1);
}
