// Diagnostic coverage uses alpha = -(1 + opacity); normal alpha stays unchanged.
// Ordered coverage writes depth and batches like opaque geometry, with no sorting.
float coverageAlpha(float alpha, vec2 pixel) {
    if (alpha >= 0.0) return alpha;
    vec2 lo = mod(floor(pixel), 2.0);
    vec2 hi = mod(floor(pixel / 2.0), 2.0);
    float low = 2.0 * mod(lo.x + lo.y, 2.0) + lo.y;
    float high = 2.0 * mod(hi.x + hi.y, 2.0) + hi.y;
    if (-alpha - 1.0 <= (4.0 * low + high + 0.5) / 16.0) discard;
    return 1.0;
}
