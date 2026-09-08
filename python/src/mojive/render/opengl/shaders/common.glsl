#ifndef MOJIVE_COMMON_GLSL
#define MOJIVE_COMMON_GLSL

const float MOJIVE_GAMMA = 2.2;
const float MOJIVE_KNEE = 0.8;

vec3 srgb_to_linear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((max(c, vec3(0.0)) + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

vec3 linear_to_srgb(vec3 c) {
    c = max(c, vec3(0.0));
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

vec3 gamma_encode(vec3 c) { return pow(max(c, vec3(0.0)), vec3(1.0 / MOJIVE_GAMMA)); }

#define MOJIVE_AMBIENT_GAIN 1.0
vec3 ambient_linear(vec3 a) {
    return srgb_to_linear(clamp(MOJIVE_AMBIENT_GAIN * a, 0.0, 1.0));
}

float softroll(float excess, float headroom) {
    return headroom * excess / (excess + headroom);
}

vec3 tonemap(vec3 c) {
    float peak = max(c.r, max(c.g, c.b));
    if (peak <= MOJIVE_KNEE) return c;
    float headroom = 1.0 - MOJIVE_KNEE;
    float mapped = MOJIVE_KNEE + softroll(peak - MOJIVE_KNEE, headroom);
    return c * (mapped / peak);
}

vec3 finish_color(vec3 c, float exposure, bool tonemap_on) {
    c *= exposure;
    c = tonemap_on ? tonemap(c) : clamp(c, 0.0, 1.0);
    return gamma_encode(clamp(c, 0.0, 1.0));
}

float depth_view(float view_depth, float near, float far) {
    return clamp((view_depth - near) / max(far - near, 1e-6), 0.0, 1.0);
}

#endif
