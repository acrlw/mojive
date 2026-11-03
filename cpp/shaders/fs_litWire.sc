$input v_color0, v_bary
#include <bgfx_shader.sh>
#include "coverage.sh"
#include "comparison.sh"
void main()
{
    coverageAlpha(v_color0.a, gl_FragCoord.xy);
    if (u_comparison.x > 1.5) { gl_FragColor = vec4(0.0); return; }
    float distanceToEdge = min(v_bary.x, min(v_bary.y, v_bary.z));
    float width = max(fwidth(distanceToEdge), 1e-6) * 1.2;
    if (distanceToEdge > width) discard;
    float opacity = coverageAlpha(v_color0.a, gl_FragCoord.xy);
    gl_FragColor = comparisonColor(vec4(pow(max(v_color0.rgb, vec3_splat(0)), vec3_splat(1.0 / 2.2)), opacity));
}
