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
std::vector<MeshLod> prepareMeshLods(const Mesh &mesh, std::stop_token stop) {
    std::vector<MeshLod> result;
    if (mesh.indices.size() < 192 || stop.stop_requested())
        return result;
    std::vector<std::array<float, 5>> attributes(mesh.vertices.size());
    for (size_t i = 0; i < attributes.size(); ++i) {
        const auto &n = mesh.vertices[i].normal;
        auto uv = mesh.texcoords.empty() ? std::array<float, 2>{} : mesh.texcoords[i];
        attributes[i] = {n[0], n[1], n[2], uv[0], uv[1]};
    }
    const auto *positions = mesh.vertices[0].position.data();
    const float scale = meshopt_simplifyScale(positions, mesh.vertices.size(), sizeof(Vertex));
    constexpr float weights[] = {.5f, .5f, .5f, 1, 1};
    size_t previous = mesh.indices.size();
    float previousError = 0;
    for (size_t target = mesh.indices.size() / 4; target >= 3; target /= 4) {
        if (stop.stop_requested())
            return {};
        std::vector<uint32_t> indices(mesh.indices.size());
        float error = 0;
        size_t count = meshopt_simplifyWithAttributes(
            indices.data(), mesh.indices.data(), mesh.indices.size(), positions,
            mesh.vertices.size(), sizeof(Vertex), attributes[0].data(), sizeof(attributes[0]),
            weights, 5, nullptr, target / 3 * 3, 1.f,
            meshopt_SimplifyPermissive | meshopt_SimplifyPrune, &error);
        // meshoptimizer calls are indivisible; abandon canceled results before
        // compaction and before starting another simplification level.
        if (stop.stop_requested())
            return {};
        if (count < 3 || count >= previous * .8)
            continue;
        auto level = std::make_shared<Mesh>();
        level->indices.reserve(count);
        std::vector<uint32_t> remap(mesh.vertices.size(), UINT32_MAX);
        for (size_t i = 0; i < count; ++i) {
            const auto index = indices[i];
            if (remap[index] == UINT32_MAX) {
                remap[index] = level->vertices.size();
                level->vertices.push_back(mesh.vertices[index]);
                if (!mesh.texcoords.empty())
                    level->texcoords.push_back(mesh.texcoords[index]);
            }
            level->indices.push_back(remap[index]);
        }
        // Independent simplifications need not have monotonically increasing error.
        previousError = std::max(previousError, error * scale);
        result.push_back({std::move(level), previousError});
        previous = count;
    }
    return result;
}
} // namespace mojive
