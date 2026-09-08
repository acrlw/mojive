#version 330 core
in vec2 v_uv;
out vec4 o_color;

uniform sampler2D u_color;
#ifdef ID_MULTISAMPLE
uniform usampler2DMS u_ids;
#else
uniform usampler2D u_ids;
#endif
uniform ivec2 u_size;
uniform int u_mode;      // 0 color, 1 segment, 2 object ID
uniform int u_selected;

uint fetch_id(ivec2 c) {
#ifdef ID_MULTISAMPLE
    return texelFetch(u_ids, c, 0).r;
#else
    return texelFetch(u_ids, c, 0).r;
#endif
}

vec3 id_color(uint id) {
    if (id == 0u) return vec3(0.0);
    uint h = id * 2654435761u;
    return vec3(float((h >> 16) & 255u), float((h >> 8) & 255u), float(h & 255u)) / 255.0;
}

void main() {
    ivec2 c = ivec2(v_uv * vec2(u_size));
    if (u_mode == 0) {
        o_color = texture(u_color, v_uv);
        return;
    }
    uint id = fetch_id(c);
    if (u_mode == 1) {
        o_color = vec4(id == uint(u_selected) && id != 0u ? vec3(1.0) : id_color(id), 1.0);
    } else {
        o_color = vec4(id_color(id), 1.0);
    }
}
