#pragma once

#include <array>
#include <cstdint>
#include <mojive/Render.hpp>
#include <span>
#include <stop_token>
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
struct MeshLod {
    std::shared_ptr<const Mesh> mesh;
    float error = 0;
};
// Shared rigid-mesh levels, with error in object-space units. Unlike explicit
// authoring simplification, distant display levels may collapse attribute seams.
std::vector<MeshLod> prepareMeshLods(const Mesh &mesh, std::stop_token stop = {});
} // namespace mojive
