#version 330 core

in vec3 in_position;
in vec4 in_model0;
in vec4 in_model1;
in vec4 in_model2;
in vec4 in_model3;
in vec4 in_color;
in uint in_object_id;
uniform mat4 u_view_proj;
uniform vec4 u_clip_plane;
uniform uint u_selected_id;
uniform vec3 u_highlight_color;
uniform vec2 u_highlight;

out WireData { vec4 color; } wire;

void main() {
    vec4 world = mat4(in_model0, in_model1, in_model2, in_model3) * vec4(in_position, 1.0);
    wire.color = in_color;
    if (u_selected_id != 0u && in_object_id == u_selected_id)
        wire.color.rgb = mix(wire.color.rgb, u_highlight_color, u_highlight.x);
    gl_ClipDistance[0] = dot(u_clip_plane, world);
    gl_Position = u_view_proj * world;
}
