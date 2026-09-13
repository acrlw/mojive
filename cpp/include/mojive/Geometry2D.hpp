#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mojive::geometry2d {

using Point = std::array<double, 2>;
using Affine = std::array<double, 6>;
inline constexpr Affine Identity{1, 0, 0, 0, 1, 0};

enum class PathVerb : uint8_t { Move, Line, Quadratic, Cubic, Arc, Close };

struct PathCommand {
    PathVerb verb;
    std::array<double, 7> values{};
};

struct Contour {
    std::vector<Point> points;
    bool closed = false;
};

struct FlattenOptions {
    // Error is measured after local-to-physical-pixel projection; output stays local.
    Affine projection = Identity;
    double tolerance = 0.25;
    size_t maxPoints = 1'000'000;
};

enum class FillRule : uint8_t { NonZero, EvenOdd };

enum class StrokeCap : uint8_t { Butt, Round, Square };
enum class StrokeJoin : uint8_t { Miter, Round, Bevel };
enum class StrokeSpace : uint8_t { Local, Screen };

struct StrokeStyle {
    double width = 1;
    StrokeCap cap = StrokeCap::Butt;
    StrokeJoin join = StrokeJoin::Miter;
    double miterLimit = 4;
    StrokeSpace space = StrokeSpace::Local;
};

struct Mesh {
    std::vector<Point> positions;
    std::vector<uint32_t> indices;
    // Directed edges of the filled region only; internal triangulation edges are excluded.
    std::vector<std::array<uint32_t, 2>> boundaryEdges;
    size_t scratchPeakBytes = 0;
};

struct TessellationOptions {
    FillRule fillRule = FillRule::NonZero;
    size_t maxVertices = 1'000'000;
    size_t maxIndices = 3'000'000;
    size_t maxScratchBytes = 64 * 1024 * 1024;
};

// CPU-only adaptive path compilation. Invalid input and exhausted quality/point
// budgets fail explicitly; the caller never receives a silently truncated path.
std::vector<Contour> flattenPath(std::span<const PathCommand> commands,
                                 const FlattenOptions &options = {});

// Open contours are implicitly closed. Supports holes, self-intersections and
// either winding rule; publishes no partial mesh on invalid input or exhaustion.
Mesh tessellate(std::span<const Point> points, std::span<const uint32_t> contourOffsets,
                const TessellationOptions &options = {});

// Local widths produce local coordinates; screen widths produce physical-pixel
// coordinates after projection. Overlapping stroke parts are unioned before
// triangulation, so one translucent stroke never blends against itself.
Mesh compileStroke(std::span<const PathCommand> commands, const StrokeStyle &style = {},
                   const FlattenOptions &quality = {}, const TessellationOptions &budget = {});

} // namespace mojive::geometry2d
