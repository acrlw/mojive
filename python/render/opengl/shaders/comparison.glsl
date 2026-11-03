// Premultiply the nearest diagnostic surface for antialiasing and composition.
uniform vec4 u_comparison;
vec4 comparisonColor(vec4 color) {
    if (u_comparison.x < 0.5) return color;
    float weight = clamp(color.a, 0.0, 1.0);
    return vec4(color.rgb * weight, weight);
}
