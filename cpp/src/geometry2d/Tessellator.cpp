#include <mojive/Geometry2D.hpp>

#include "TessAllocator.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace mojive::geometry2d {
namespace {
struct Work {
    TessAllocator &allocator;
    TESSalloc callbacks;
    std::span<const std::array<TESSreal, 2>> points;
    std::span<const uint32_t> offsets;
    int winding;
    TESStesselator *tess = nullptr;
};

int run(Work &work) {
    // Only trivial locals here: allocator exhaustion jumps across C frames.
    if (setjmp(work.allocator.failure))
        return 0;
    work.tess = tessNewTess(&work.callbacks);
    if (!work.tess)
        return 0;
    for (size_t i = 1; i < work.offsets.size(); ++i) {
        auto count = work.offsets[i] - work.offsets[i - 1];
        if (count >= 3)
            tessAddContour(work.tess, 2, work.points.data() + work.offsets[i - 1],
                           sizeof(work.points[0]), int(count));
    }
    const TESSreal normal[]{0, 0, 1};
    return tessTesselate(work.tess, work.winding, TESS_CONNECTED_POLYGONS, 3, 2, normal);
}

void validate(std::span<const Point> points, std::span<const uint32_t> offsets,
              const TessellationOptions &options) {
    if (!options.maxVertices || !options.maxIndices || !options.maxScratchBytes)
        throw std::invalid_argument("Tessellation budgets must be positive");
    if (options.fillRule != FillRule::NonZero && options.fillRule != FillRule::EvenOdd)
        throw std::invalid_argument("Unknown fill rule");
    if (points.size() > options.maxVertices || points.size() > size_t(INT32_MAX))
        throw std::length_error("Tessellation input vertex budget exceeded");
    if (offsets.empty() || offsets.front() != 0 || offsets.back() != points.size() ||
        !std::is_sorted(offsets.begin(), offsets.end()))
        throw std::invalid_argument("Contour offsets must cover the input points in order");
    for (auto point : points)
        if (!std::isfinite(point[0]) || !std::isfinite(point[1]))
            throw std::invalid_argument("Contour coordinates must be finite");
}
} // namespace

Mesh tessellate(std::span<const Point> points, std::span<const uint32_t> contourOffsets,
                const TessellationOptions &options) {
    validate(points, contourOffsets, options);
    Mesh result;
    if (points.empty())
        return result;
    bool hasContour = false;
    for (size_t i = 1; i < contourOffsets.size(); ++i)
        hasContour |= contourOffsets[i] - contourOffsets[i - 1] >= 3;
    if (!hasContour)
        return result;
    Point lo = points.front(), hi = lo;
    for (auto point : points)
        for (size_t axis = 0; axis < 2; ++axis) {
            lo[axis] = std::min(lo[axis], point[axis]);
            hi[axis] = std::max(hi[axis], point[axis]);
        }
    Point center{lo[0] / 2 + hi[0] / 2, lo[1] / 2 + hi[1] / 2};
    double extent = std::max(hi[0] - lo[0], hi[1] - lo[1]);
    if (!std::isfinite(extent))
        throw std::overflow_error("Contour extent exceeds finite coordinates");
    if (extent == 0)
        return result;

    // Normalize before converting to the dependency's float coordinates. Large
    // translations neither exceed its input range nor discard small local detail.
    std::vector<std::array<TESSreal, 2>> normalized;
    normalized.reserve(points.size());
    for (auto p : points)
        normalized.push_back(
            {TESSreal((p[0] - center[0]) / extent), TESSreal((p[1] - center[1]) / extent)});
    TessAllocator allocator(options.maxScratchBytes);
    Work work{allocator, allocator.callbacks(), normalized, contourOffsets,
              options.fillRule == FillRule::EvenOdd ? TESS_WINDING_ODD : TESS_WINDING_NONZERO};
    if (!run(work))
        throw std::runtime_error("Tessellation failed or scratch memory budget exhausted");
    std::unique_ptr<TESStesselator, decltype(&tessDeleteTess)> tess(work.tess, tessDeleteTess);
    auto vertexCount = size_t(tessGetVertexCount(tess.get()));
    auto triangleCount = size_t(tessGetElementCount(tess.get()));
    if (vertexCount > options.maxVertices || triangleCount > options.maxIndices / 3)
        throw std::length_error("Tessellation output budget exceeded");
    const auto *vertices = tessGetVertices(tess.get());
    result.positions.reserve(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i)
        result.positions.push_back(
            {center[0] + vertices[i * 2] * extent, center[1] + vertices[i * 2 + 1] * extent});
    const auto *elements = tessGetElements(tess.get());
    result.indices.reserve(triangleCount * 3);
    for (size_t i = 0; i < triangleCount; ++i) {
        auto *triangle = elements + i * 6;
        for (size_t edge = 0; edge < 3; ++edge) {
            auto a = triangle[edge], b = triangle[(edge + 1) % 3];
            if (a < 0 || size_t(a) >= vertexCount || b < 0 || size_t(b) >= vertexCount)
                throw std::runtime_error("Tessellator returned an invalid vertex index");
            result.indices.push_back(uint32_t(a));
            if (triangle[edge + 3] == TESS_UNDEF)
                result.boundaryEdges.push_back({uint32_t(a), uint32_t(b)});
        }
    }
    result.scratchPeakBytes = allocator.peakBytes();
    return result;
}
} // namespace mojive::geometry2d
