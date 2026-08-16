#version 330 core
in vec3 in_p;
in vec4 in_color;
in float in_radius;  // Pixels

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_px_scale;
uniform float u_alpha;

out vec4 v_color;
out vec2 v_uv;

void main() {
    v_color = vec4(in_color.rgb, in_color.a * u_alpha);

    const vec2 C[6] = vec2[6](
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
    );
    v_uv = C[gl_VertexID];

    vec3 p = (u_view * vec4(in_p, 1.0)).xyz;
    float w = (u_proj * vec4(p, 1.0)).w;
    p.xy += v_uv * (in_radius * u_px_scale * w);
    gl_Position = u_proj * vec4(p, 1.0);
}
