#version 330 core

in vec3 in_anchor;
in vec2 in_offset;
in vec4 in_rect;
in vec4 in_uv_rect;
in vec4 in_color;

uniform mat4 u_view_proj;
uniform vec2 u_viewport;
uniform float u_alpha;

out vec2 v_uv;
out vec4 v_color;

void main() {
    const vec2 C[6] = vec2[6](
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    );
    vec2 corner = C[gl_VertexID];
    vec2 pixel = in_offset + mix(in_rect.xy, in_rect.zw, corner);
    vec4 clip = u_view_proj * vec4(in_anchor, 1.0);
    clip.xy += vec2(2.0 * pixel.x / u_viewport.x, -2.0 * pixel.y / u_viewport.y) * clip.w;
    gl_Position = clip;
    v_uv = mix(in_uv_rect.xy, in_uv_rect.zw, corner);
    v_color = vec4(in_color.rgb, in_color.a * u_alpha);
}
