#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <mojive/Render.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace mojive {
struct MeshBounds {
    glm::vec3 center{}, extent{};
};
inline MeshBounds meshBounds(std::span<const Vertex> vertices) {
    if (vertices.empty())
        return {};
    const auto &p = vertices.front().position;
    glm::vec3 lo(p[0], p[1], p[2]), hi = lo;
    for (const auto &vertex : vertices) {
        const auto &v = vertex.position;
        glm::vec3 point(v[0], v[1], v[2]);
        lo = glm::min(lo, point);
        hi = glm::max(hi, point);
    }
    return {(lo + hi) * .5f, (hi - lo) * .5f};
}
inline MeshBounds transformBounds(const MeshBounds &bounds, const float *matrix) {
    MeshBounds result;
    // Instance attributes contain three row-major affine rows, including translation.
    // Absolute row coefficients conservatively enclose negative scale and shear.
    for (int axis = 0; axis < 3; ++axis) {
        glm::vec3 row(matrix[axis * 4], matrix[axis * 4 + 1], matrix[axis * 4 + 2]);
        result.center[axis] = glm::dot(row, bounds.center) + matrix[axis * 4 + 3];
        result.extent[axis] = glm::dot(glm::abs(row), bounds.extent);
    }
    return result;
}
class ClipFrustum {
    friend class ShadowReceiverVolume;
    std::array<glm::vec4, 6> mPlanes;

  public:
    explicit ClipFrustum(const glm::mat4 &matrix) {
        // Canonical cameras use OpenGL clip depth before backend upload conversion.
        const auto rows = glm::transpose(matrix);
        for (int axis = 0; axis < 3; ++axis) {
            mPlanes[axis * 2] = rows[3] + rows[axis];
            mPlanes[axis * 2 + 1] = rows[3] - rows[axis];
        }
    }
    explicit ClipFrustum(const CameraView &camera)
        : ClipFrustum(glm::transpose(glm::make_mat4(camera.projection.data())) *
                      glm::transpose(glm::make_mat4(camera.view.data()))) {}
    bool intersects(const MeshBounds &bounds) const {
        for (const auto &plane : mPlanes) {
            const glm::vec3 normal(plane);
            const float center = glm::dot(normal, bounds.center);
            const float radius = glm::dot(glm::abs(normal), bounds.extent);
            const float tolerance = 1e-5f * (1 + std::abs(center) + std::abs(plane.w) + radius);
            if (center + plane.w + radius < -tolerance)
                return false;
        }
        return true;
    }
};

// A directional caster matters only if its extrusion along the light ray can
// reach a receiver. Plane-wise AABB support is conservative: it may retain extra
// casters, but never drops an offscreen caster whose shadow reaches the view.
class ShadowReceiverVolume {
    std::array<glm::vec4, 12> mPlanes{};
    std::array<float, 12> mSlopes{};
    size_t mCount = 6;

  public:
    ShadowReceiverVolume(const CameraView &camera, glm::vec3 direction,
                         const MeshBounds *receivers) {
        const ClipFrustum frustum(camera);
        std::copy(frustum.mPlanes.begin(), frustum.mPlanes.end(), mPlanes.begin());
        if (receivers) {
            // Clip away empty space below/above the scene. Extruding the entire
            // camera frustum would otherwise retain shadows on imaginary ground.
            for (int axis = 0; axis < 3; ++axis) {
                glm::vec4 lower(0), upper(0);
                lower[axis] = 1;
                upper[axis] = -1;
                lower.w = receivers->extent[axis] - receivers->center[axis];
                upper.w = receivers->extent[axis] + receivers->center[axis];
                mPlanes[mCount++] = lower;
                mPlanes[mCount++] = upper;
            }
        }
        for (size_t i = 0; i < mCount; ++i)
            mSlopes[i] = glm::dot(glm::vec3(mPlanes[i]), direction);
    }
    bool intersects(const MeshBounds &caster) const {
        float enter = 0, leave = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < mCount; ++i) {
            const auto &plane = mPlanes[i];
            const glm::vec3 normal(plane);
            const float center = glm::dot(normal, caster.center);
            const float radius = glm::dot(glm::abs(normal), caster.extent);
            const float tolerance = 1e-5f * (1 + std::abs(center) + std::abs(plane.w) + radius);
            const float support = center + plane.w + radius + tolerance;
            if (mSlopes[i] > 0)
                enter = std::max(enter, -support / mSlopes[i]);
            else if (mSlopes[i] < 0)
                leave = std::min(leave, -support / mSlopes[i]);
            else if (support < 0)
                return false;
            if (enter > leave)
                return false;
        }
        return true;
    }
};
} // namespace mojive
