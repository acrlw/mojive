#version 330 core

in vec4 v_color;
#ifdef ROUND_ARROW
in vec2 v_arrow_pos;
flat in vec4 v_arrow_shape;
#endif

layout(location = 0) out vec4 o_color;

#ifdef ROUND_ARROW
float rounded_corner_coverage(
    vec2 point,
    vec2 corner,
    vec2 previous,
    vec2 next,
    float radius
) {
    vec2 incoming_edge = previous - corner;
    vec2 outgoing_edge = next - corner;
    float incoming_length = length(incoming_edge);
    float outgoing_length = length(outgoing_edge);
    vec2 incoming = incoming_edge / max(incoming_length, 1e-6);
    vec2 outgoing = outgoing_edge / max(outgoing_length, 1e-6);
    float half_angle = 0.5 * acos(clamp(dot(incoming, outgoing), -1.0, 1.0));
    float tan_half = max(tan(half_angle), 1e-6);
    float tangent = min(
        radius / tan_half,
        0.45 * min(incoming_length, outgoing_length)
    );
    float effective_radius = tangent * tan_half;
    vec2 bisector = normalize(incoming + outgoing);
    vec2 center = corner + bisector * (
        effective_radius / max(sin(half_angle), 1e-6)
    );
    vec2 relative = point - corner;
    float circle_distance = effective_radius - distance(point, center);
    float aa = max(fwidth(circle_distance), 0.5);
    float rounded = smoothstep(-0.5 * aa, 0.5 * aa, circle_distance);
    bool near_corner = dot(relative, incoming) < tangent
        && dot(relative, outgoing) < tangent;
    return near_corner ? rounded : 1.0;
}

float arrow_coverage(vec2 point, vec4 shape) {
    float head = shape.x;
    float shaft = shape.y;
    float wing = shape.z;
    float radius = shape.w;
    vec2 lower_shaft = vec2(0.0, -shaft);
    vec2 lower_wing = vec2(0.0, -wing);
    vec2 tip = vec2(head, 0.0);
    vec2 upper_wing = vec2(0.0, wing);
    vec2 upper_shaft = vec2(0.0, shaft);
    float coverage = rounded_corner_coverage(
        point, lower_wing, lower_shaft, tip, radius
    );
    coverage = min(
        coverage,
        rounded_corner_coverage(point, tip, lower_wing, upper_wing, radius)
    );
    coverage = min(
        coverage,
        rounded_corner_coverage(point, upper_wing, tip, upper_shaft, radius)
    );
    return coverage;
}
#endif

void main() {
    float coverage = 1.0;
#ifdef ROUND_ARROW
    if (v_arrow_shape.x > 0.0) {
        coverage = arrow_coverage(v_arrow_pos, v_arrow_shape);
    }
#endif
    o_color = vec4(v_color.rgb, v_color.a * coverage);
}
