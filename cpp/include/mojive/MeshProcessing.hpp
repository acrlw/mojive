#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace mojive {
struct SimplifiedIndices {
    std::vector<uint32_t> indices;
    float relativeError = 0;
};
// Preserve original vertices and attribute seams. The requested ratio is a
// target; topology and the relative error bound may require more triangles.
SimplifiedIndices simplifyMesh(std::span<const std::array<float, 3>> positions,
                               std::span<const std::array<float, 3>> normals,
                               std::span<const std::array<float, 2>> texcoords,
                               std::span<const uint32_t> indices, float ratio, float maxError);
} // namespace mojive
