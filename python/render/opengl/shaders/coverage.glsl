// Diagnostic alpha is -(1 + opacity), or -(3 + opacity) for collision coverage.
// Ordered coverage writes depth and batches like opaque geometry, with no sorting.
float coverageAlpha(float alpha, vec2 pixel) {
    if (alpha >= 0.0) return alpha;
    vec2 lo = mod(floor(pixel), 2.0);
    vec2 hi = mod(floor(pixel / 2.0), 2.0);
    float low = 2.0 * mod(lo.x + lo.y, 2.0) + lo.y;
    float high = 2.0 * mod(hi.x + hi.y, 2.0) + hi.y;
    float threshold = (4.0 * low + high + 0.5) / 16.0;
    // Classify inside the unused encoding gap so interpolation cannot change the role at -3.
    bool collision = alpha < -2.5;
    float opacity = -alpha - (collision ? 3.0 : 1.0);
    // Complementary patterns keep an inner collision surface visible through a visual shell.
    if (opacity <= (collision ? 1.0 - threshold : threshold)) discard;
    return 1.0;
}
