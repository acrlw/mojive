$input v_color0, v_bary
#include <bgfx_shader.sh>
#include "coverage.sh"
void main()
{
    float distanceToEdge = min(v_bary.x, min(v_bary.y, v_bary.z));
    float width = max(fwidth(distanceToEdge), 1e-6) * 1.2;
    if (distanceToEdge > width) discard;
    float opacity = coverageAlpha(v_color0.a, gl_FragCoord.xy);
    gl_FragColor = vec4(pow(max(v_color0.rgb, vec3_splat(0)), vec3_splat(1.0 / 2.2)), opacity);
}
