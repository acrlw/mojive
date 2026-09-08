#include <meshoptimizer.h>
#include <mojive/MeshProcessing.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mojive {
SimplifiedIndices simplifyMesh(std::span<const std::array<float, 3>> positions,
                               std::span<const std::array<float, 3>> normals,
                               std::span<const std::array<float, 2>> texcoords,
                               std::span<const uint32_t> indices, float ratio, float maxError) {
    if (normals.size() != positions.size() || texcoords.size() != positions.size() ||
        indices.size() % 3 || !std::isfinite(ratio) || ratio <= 0 || ratio > 1 ||
        !std::isfinite(maxError) || maxError < 0)
        throw std::invalid_argument("Invalid mesh dimensions, ratio, or error bound");
    for (auto index : indices)
        if (index >= positions.size())
            throw std::invalid_argument("Mesh index exceeds vertex count");
    std::vector<std::array<float, 5>> attributes(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        for (auto value : positions[i])
            if (!std::isfinite(value))
                throw std::invalid_argument("Mesh positions must be finite");
        attributes[i] = {normals[i][0], normals[i][1], normals[i][2], texcoords[i][0],
                         texcoords[i][1]};
        for (auto value : attributes[i])
            if (!std::isfinite(value))
                throw std::invalid_argument("Mesh attributes must be finite");
    }
    SimplifiedIndices result{std::vector<uint32_t>(indices.begin(), indices.end()), 0};
    if (indices.empty() || ratio == 1)
        return result;
    // UVs and normals participate in error estimation; disconnected components
    // are retained and experimental seam-collapsing options remain disabled.
    constexpr float weights[] = {.5f, .5f, .5f, 1, 1};
    const size_t target = std::max(size_t(3), size_t(indices.size() / 3 * ratio) * 3);
    size_t count = meshopt_simplifyWithAttributes(
        result.indices.data(), indices.data(), indices.size(), positions[0].data(),
        positions.size(), sizeof(positions[0]), attributes[0].data(), sizeof(attributes[0]),
        weights, 5, nullptr, target, maxError, 0, &result.relativeError);
    result.indices.resize(count);
    return result;
}
} // namespace mojive
