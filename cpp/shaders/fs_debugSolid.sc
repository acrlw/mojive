$input v_color0, v_normal, v_view_pos
#include <bgfx_shader.sh>



void main() {
    vec3 n = normalize(v_normal);
    vec3 to_eye = normalize(-v_view_pos);
    // Convex solid overlays must not paint their back surface over the front.
    float facing = dot(n, to_eye);
    if (facing <= 0.0) discard;
    gl_FragColor = vec4(v_color0.rgb * (0.55 + 0.45 * facing), v_color0.a);
}
