#version 330 core

uniform float u_near;
uniform float u_far;
uniform bool u_orthographic;

flat in ivec2 v_segmentation;

layout(location = 0) out float o_metric_depth;
layout(location = 1) out ivec2 o_segmentation;

void main() {
    float z = gl_FragCoord.z;
    if (u_orthographic) {
        o_metric_depth = u_near + z * (u_far - u_near);
    } else {
        o_metric_depth = (u_near * u_far) /
            max(u_far - z * (u_far - u_near), 1e-12);
    }
    o_segmentation = v_segmentation;
}
