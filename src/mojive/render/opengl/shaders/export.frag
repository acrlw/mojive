#version 330 core

flat in ivec2 v_segmentation;
in float v_view_depth;

layout(location = 0) out float o_metric_depth;
layout(location = 1) out ivec2 o_segmentation;

void main() {
    o_metric_depth = v_view_depth;
    o_segmentation = v_segmentation;
}
