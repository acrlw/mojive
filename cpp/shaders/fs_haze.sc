$input v_color0
#include <bgfx_shader.sh>
uniform vec4 u_options,u_hazeColor;
#include "color.sh"
void main() {
    vec3 c=u_options.y>.5?u_hazeColor.rgb:finishColor(srgbToLinear(u_hazeColor.rgb));
    gl_FragColor=vec4(c,v_color0.a);
}
