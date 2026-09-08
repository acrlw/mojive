$input v_world
#include <bgfx_shader.sh>
uniform vec4 u_options;
SAMPLERCUBE(s_sky, 0);
#include "color.sh"
void main() {
    vec3 d=normalize(v_world);
    vec3 c=textureCube(s_sky,vec3(d.x,d.z,-d.y)).rgb;
    gl_FragColor=vec4(u_options.y>.5?linearToSrgb(c):finishColor(c),1);
}
