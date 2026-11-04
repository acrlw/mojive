// Copied from python/render/opengl/shaders/outline.frag; GLSL 330 body is GLSL ES 300 compatible.
export const outlineFragment = `
precision highp float;
precision highp int;

#ifndef OUTLINE_RADIUS
#define OUTLINE_RADIUS 3
#endif

uniform sampler2D u_mask;
uniform ivec2 u_size;
uniform float u_xray_alpha;

#ifdef OUTLINE_ALPHA
layout(location = 0) out float o_alpha;
#else
uniform vec4 u_color;
layout(location = 0) out vec4 o_color;
#endif

const int AA_RADIUS = OUTLINE_RADIUS + 1;
const float CORE_R2 = float(OUTLINE_RADIUS * OUTLINE_RADIUS);
const float OUTER_R2 = float(AA_RADIUS * AA_RADIUS);

float coverage_at(ivec2 c) {
    if (c.x < 0 || c.y < 0 || c.x >= u_size.x || c.y >= u_size.y) return 0.0;
    return texelFetch(u_mask, c, 0).r;
}

bool selected_at(ivec2 c) {
    return coverage_at(c) > 0.0;
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
    // At a slanted intersection, the connected path can run diagonally to the
    // border. Restricting it to one column leaves a notch in the closing band.
    ivec2 tangent = ivec2(-step.y, step.x);
    for (int offset = -AA_RADIUS; offset <= AA_RADIUS; ++offset) {
        ivec2 delta = step * to_border + tangent * offset;
        if (dot(vec2(delta), vec2(delta)) > OUTER_R2) continue;
        int count = max(to_border, abs(offset));
        bool connected = true;
        for (int i = 1; i <= AA_RADIUS; ++i) {
            if (i > count) break;
            ivec2 sample_offset = ivec2(floor(vec2(delta) * float(i) / float(count) + vec2(0.5)));
            if (!selected_at(p + sample_offset)) { connected = false; break; }
        }
        if (connected) return true;
    }
    return false;
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
    float alpha = outline_alpha(p);
    // These are complementary subpixel areas, not overlapping layers.
    if (center > 0.0 && clipped_by_border(p)) alpha += center;
    alpha = max(alpha, center * u_xray_alpha);
#ifdef OUTLINE_ALPHA
    o_alpha = alpha;
#else
    if (alpha <= 0.0) discard;
    o_color = vec4(u_color.rgb, u_color.a * alpha);
#endif
}
`;
