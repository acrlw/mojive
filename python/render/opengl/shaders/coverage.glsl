// Diagnostic alpha retains its role encoding without spatial pixel rejection.
float coverageAlpha(float alpha, vec2 pixel) {
    if (alpha >= 0.0) return alpha;
    float opacity = -alpha - (alpha < -2.5 ? 3.0 : 1.0);
    if (opacity <= 0.0) discard;
    return opacity;
}
