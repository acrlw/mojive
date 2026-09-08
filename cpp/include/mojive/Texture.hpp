#pragma once

#include <mojive/Render.hpp>

namespace mojive {
// Pack complete face-major RGBA8 mip chains without changing color-space or alpha semantics.
TextureSource prepareTexture(Extent size, std::span<const uint8_t> pixels, uint32_t components,
                             bool cube, bool srgb);
} // namespace mojive
