#include <bgfx_compute.sh>

SAMPLER2D(s_readbackImage, 0);
BUFFER_RW(b_readback, uint, 1);
uniform vec4 u_readbackRegion;
// Source height, vertical flip, encoding: RGB / RGBA-u32 / float32 / int32 pair.
uniform vec4 u_readbackLayout;

vec4 pixel(uint x, uint y) {
    ivec2 p = ivec2(u_readbackRegion.xy) + ivec2(min(x, uint(u_readbackRegion.z) - 1u), y);
    if (u_readbackLayout.y > .5) p.y = int(u_readbackLayout.x) - 1 - p.y;
    return texelFetch(s_readbackImage, p, 0);
}

uint rgbWord(uint x, uint y) {
    uvec3 rgb = uvec3(floor(pixel(x, y).rgb * 255.0 + .5));
    return rgb.x | (rgb.y << 8u) | (rgb.z << 16u);
}

NUM_THREADS(64, 1, 1)
void main() {
    uint format = uint(u_readbackLayout.z);
    uint x = gl_GlobalInvocationID.x * (format == 0u ? 4u : 1u);
    uint y = gl_GlobalInvocationID.y;
    uint width = uint(u_readbackRegion.z);
    if (x >= width || y >= uint(u_readbackRegion.w)) return;
    if (format != 0u) {
        vec4 value = pixel(x, y);
        uint offset = y * width + x;
        if (format == 2u) {
            b_readback[offset] = floatBitsToUint(value.r);
        } else if (format == 3u) {
            uvec4 words = uvec4(floor(value * 65535.0 + .5));
            b_readback[offset * 2u] = words.x | (words.y << 16u);
            b_readback[offset * 2u + 1u] = words.z | (words.w << 16u);
        } else {
            uvec4 bytes = uvec4(floor(value * 255.0 + .5));
            b_readback[offset] = bytes.x | (bytes.y << 8u) | (bytes.z << 16u) | (bytes.w << 24u);
        }
        return;
    }
    uint a = rgbWord(x, y);
    uint b = rgbWord(x + 1u, y);
    uint c = rgbWord(x + 2u, y);
    uint d = rgbWord(x + 3u, y);
    uint offset = (y * ((width + 3u) / 4u) + x / 4u) * 3u;
    b_readback[offset] = a | (b << 24u);
    b_readback[offset + 1u] = (b >> 8u) | (c << 16u);
    b_readback[offset + 2u] = (c >> 16u) | (d << 8u);
}
