$input v_color0, v_normal, v_view_pos
#include <bgfx_shader.sh>



void main() {
    vec3 n = normalize(v_normal);
    vec3 to_eye = normalize(-v_view_pos);
    float facing = abs(dot(n, to_eye));
    gl_FragColor = vec4(v_color0.rgb * (0.55 + 0.45 * facing), v_color0.a);
}
