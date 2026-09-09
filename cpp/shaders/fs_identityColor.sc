#include <bgfx_shader.sh>
SAMPLER2D(s_identity, 0);
uniform vec4 u_identitySize;
void main() {
    vec4 bytes = floor(texture2DLod(s_identity, gl_FragCoord.xy / u_identitySize.xy, 0) * 255.0 + 0.5);
    uint id = uint(bytes.x) | (uint(bytes.y) << 8u) | (uint(bytes.z) << 16u) | (uint(bytes.w) << 24u);
    uint selected = uint(u_identitySize.z) | (uint(u_identitySize.w) << 16u);
    uint h = id * 2654435761u;
    vec3 color = vec3(float((h >> 16u) & 255u), float((h >> 8u) & 255u), float(h & 255u)) / 255.0;
    gl_FragColor = vec4(id == 0u ? vec3(0.0) : id == selected ? vec3(1.0) : color, 1.0);
}
