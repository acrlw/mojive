#version 330 core
#ifndef OUTLINE_RADIUS
#define OUTLINE_RADIUS 3
#endif

uniform sampler2D u_mask;
uniform ivec2 u_size;
uniform vec4 u_color;
uniform float u_xray_alpha;

layout(location = 0) out vec4 o_color;

const int AA_RADIUS = OUTLINE_RADIUS + 1;
const float CORE_R2 = float(OUTLINE_RADIUS * OUTLINE_RADIUS);
const float OUTER_R2 = float(AA_RADIUS * AA_RADIUS);

float coverage_at(ivec2 c) {
    if (c.x < 0 || c.y < 0 || c.x >= u_size.x || c.y >= u_size.y) return 0.0;
    return texelFetch(u_mask, c, 0).r;
}

bool selected_at(ivec2 c) {
    return coverage_at(c) >= 0.5;
}

float outline_alpha(ivec2 p) {
    float expanded = 0.0;
    for (int dy = -AA_RADIUS; dy <= AA_RADIUS; ++dy) {
        for (int dx = -AA_RADIUS; dx <= AA_RADIUS; ++dx) {
            float r2 = float(dx * dx + dy * dy);
            if (r2 > OUTER_R2) continue;
            float kernel = 1.0 - smoothstep(CORE_R2, OUTER_R2, r2);
            expanded = max(expanded, coverage_at(p + ivec2(dx, dy)) * kernel);
        }
    }
    return max(expanded - coverage_at(p), 0.0);
}

bool runs_to_border(ivec2 p, ivec2 step, int to_border) {
    if (to_border >= OUTLINE_RADIUS) return false;
    for (int i = 1; i <= OUTLINE_RADIUS; ++i) {
        if (i > to_border) break;
        if (!selected_at(p + step * i)) return false;
    }
    return true;
}

bool clipped_by_border(ivec2 p) {
    return runs_to_border(p, ivec2(-1, 0), p.x)
        || runs_to_border(p, ivec2(1, 0), u_size.x - 1 - p.x)
        || runs_to_border(p, ivec2(0, -1), p.y)
        || runs_to_border(p, ivec2(0, 1), u_size.y - 1 - p.y);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float center = coverage_at(p);
    float alpha = max(outline_alpha(p), center * u_xray_alpha);
    if (center > 0.0 && clipped_by_border(p)) alpha = max(alpha, center);
    if (alpha <= 0.0) discard;

    o_color = vec4(u_color.rgb, u_color.a * alpha);
}
