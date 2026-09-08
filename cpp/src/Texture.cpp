#include <algorithm>
#include <cstring>
#include <limits>
#include <mojive/Texture.hpp>
#include <stdexcept>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include <stb_image_resize2.h>

namespace mojive {
TextureSource prepareTexture(Extent size, std::span<const uint8_t> pixels, uint32_t components,
                             bool cube, bool srgb) {
    const size_t faces = cube ? 6 : 1;
    if (!size.width || !size.height || size.width > 16384 || size.height > 16384 ||
        components < 1 || components > 4 || (cube && size.width != size.height) ||
        pixels.size() != uint64_t(size.width) * size.height * components * faces)
        throw std::invalid_argument("Invalid texture dimensions or pixel count");
    TextureSource result;
    result.size = size;
    result.mipmaps = true;
    result.cube = cube;
    result.srgb = srgb;
    size_t faceBytes = 0;
    for (auto w = size.width, h = size.height;; w = std::max(1u, w / 2), h = std::max(1u, h / 2)) {
        faceBytes += size_t(w) * h * 4;
        if (w == 1 && h == 1)
            break;
    }
    if (faceBytes > std::numeric_limits<uint32_t>::max() / faces)
        throw std::invalid_argument("Texture payload exceeds the upload limit");
    auto rgba = std::make_shared<std::vector<std::byte>>(faceBytes * faces);
    result.rgba = rgba;
    for (size_t face = 0; face < faces; ++face) {
        auto *level = rgba->data() + face * faceBytes;
        const auto *source = pixels.data() + face * size.width * size.height * components;
        const size_t count = size_t(size.width) * size.height;
        if (components == 4)
            std::memcpy(level, source, count * 4);
        else
            for (size_t i = 0; i < count; ++i) {
                auto *pixel = level + 4 * i;
                pixel[0] = pixel[1] = pixel[2] = std::byte{0};
                pixel[3] = std::byte{255};
                std::memcpy(pixel, source + components * i, components);
            }
        auto w = size.width, h = size.height;
        while (w > 1 || h > 1) {
            const auto nextW = std::max(1u, w / 2), nextH = std::max(1u, h / 2);
            auto *next = level + size_t(w) * h * 4;
            // Match the shared mip contract: RGB is filtered in linear light and
            // alpha independently. NO_AW avoids stb's default alpha-weighted RGB.
            if (!stbir_resize(level, w, h, 0, next, nextW, nextH, 0, STBIR_RGBA_NO_AW,
                              srgb ? STBIR_TYPE_UINT8_SRGB : STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP,
                              STBIR_FILTER_BOX))
                throw std::runtime_error("Cannot generate texture mipmaps");
            level = next;
            w = nextW;
            h = nextH;
        }
    }
    return result;
}
} // namespace mojive
