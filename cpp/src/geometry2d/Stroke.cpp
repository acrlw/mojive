#include <mojive/Geometry2D.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace mojive::geometry2d {
namespace {
Point add(Point a, Point b) {
    return {a[0] + b[0], a[1] + b[1]};
}
Point subtract(Point a, Point b) {
    return {a[0] - b[0], a[1] - b[1]};
}
Point multiply(Point p, double value) {
    return {p[0] * value, p[1] * value};
}
double dot(Point a, Point b) {
    return a[0] * b[0] + a[1] * b[1];
}
double cross(Point a, Point b) {
    return a[0] * b[1] - a[1] * b[0];
}
Point normal(Point direction) {
    return {-direction[1], direction[0]};
}

Point direction(Point a, Point b) {
    auto delta = subtract(b, a);
    double length = std::hypot(delta[0], delta[1]);
    if (!std::isfinite(length) || length == 0)
        throw std::overflow_error("Stroke segment exceeds finite extent");
    return {delta[0] / length, delta[1] / length};
}

void validate(const StrokeStyle &style) {
    if (!std::isfinite(style.width) || style.width < 0 || !std::isfinite(style.miterLimit) ||
        style.miterLimit < 1)
        throw std::invalid_argument("Invalid stroke width or miter limit");
    if (style.cap > StrokeCap::Square || style.join > StrokeJoin::Bevel ||
        style.space > StrokeSpace::Screen)
        throw std::invalid_argument("Invalid stroke cap, join or space");
}

class StrokeOutline {
    const StrokeStyle &mStyle;
    const TessellationOptions &mBudget;
    double mRadius;
    std::vector<Point> mCircle;
    std::vector<Point> mPoints;
    std::vector<uint32_t> mOffsets{0};

    void reservePolygon(size_t count) const {
        if (count > mBudget.maxVertices - mPoints.size())
            throw std::length_error("Stroke outline point budget exceeded");
    }

    void append(Point p) {
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]))
            throw std::overflow_error("Stroke outline exceeds finite coordinates");
        mPoints.push_back(p);
    }

    void polygon(std::span<const Point> vertices) {
        reservePolygon(vertices.size());
        // Consistent positive winding makes every primitive contribute to the
        // union, including reversed subpaths and concave/self-crossing strokes.
        double area = 0;
        for (size_t i = 1; i + 1 < vertices.size(); ++i)
            area +=
                cross(subtract(vertices[i], vertices[0]), subtract(vertices[i + 1], vertices[0]));
        if (!std::isfinite(area))
            throw std::overflow_error("Stroke polygon exceeds finite area");
        if (area == 0)
            return;
        if (area < 0) {
            for (auto i = vertices.rbegin(); i != vertices.rend(); ++i)
                append(*i);
        } else {
            for (auto p : vertices)
                append(p);
        }
        mOffsets.push_back(uint32_t(mPoints.size()));
    }

    void circle(Point center) {
        reservePolygon(mCircle.size());
        for (auto p : mCircle)
            append(add(center, p));
        mOffsets.push_back(uint32_t(mPoints.size()));
    }

    void join(Point center, Point incoming, Point outgoing) {
        if (mStyle.join == StrokeJoin::Round) {
            circle(center);
            return;
        }
        double turn = cross(incoming, outgoing);
        if (turn == 0)
            return;
        double side = turn > 0 ? -1 : 1;
        auto n0 = multiply(normal(incoming), side), n1 = multiply(normal(outgoing), side);
        auto a = add(center, multiply(n0, mRadius)), b = add(center, multiply(n1, mRadius));
        auto sum = add(n0, n1);
        double denominator = 1 + dot(incoming, outgoing);
        double sumLength = std::hypot(sum[0], sum[1]);
        if (mStyle.join == StrokeJoin::Miter && denominator > 0 &&
            sumLength <= mStyle.miterLimit * denominator) {
            auto tip = add(center, multiply(sum, mRadius / denominator));
            polygon(std::array{center, a, tip, b});
        } else {
            polygon(std::array{center, a, b});
        }
    }

    void pointCap(Point p) {
        if (mStyle.cap == StrokeCap::Round)
            circle(p);
        else if (mStyle.cap == StrokeCap::Square)
            polygon(std::array{add(p, {-mRadius, -mRadius}), add(p, {mRadius, -mRadius}),
                               add(p, {mRadius, mRadius}), add(p, {-mRadius, mRadius})});
    }

  public:
    StrokeOutline(const StrokeStyle &style, const FlattenOptions &quality,
                  const TessellationOptions &budget)
        : mStyle(style), mBudget(budget), mRadius(style.width / 2) {
        if (budget.maxVertices == 0 || budget.maxVertices > INT32_MAX || budget.maxIndices == 0 ||
            budget.maxScratchBytes == 0)
            throw std::invalid_argument("Invalid stroke tessellation budget");
        if (style.cap != StrokeCap::Round && style.join != StrokeJoin::Round)
            return;
        const auto &m = quality.projection;
        double radius = mRadius * std::hypot(std::hypot(m[0], m[1]), std::hypot(m[3], m[4]));
        if (!std::isfinite(radius))
            throw std::overflow_error("Stroke radius exceeds finite projection");
        double step = std::numbers::pi / 2;
        if (radius > quality.tolerance)
            step = std::min(step, 2 * std::acos(std::max(-1.0, 1 - quality.tolerance / radius)));
        double count = std::ceil(2 * std::numbers::pi / step);
        if (!std::isfinite(count) || count > budget.maxVertices)
            throw std::length_error("Stroke round cap exceeds point budget");
        // Multiples of four preserve exact extrema for bounds and mirrored caps.
        auto segments = size_t(count);
        segments = (segments + 3) / 4 * 4;
        if (segments > budget.maxVertices)
            throw std::length_error("Stroke round cap exceeds point budget");
        mCircle.reserve(segments);
        for (size_t i = 0; i < segments; ++i) {
            double angle = 2 * std::numbers::pi * double(i) / double(segments);
            mCircle.push_back({mRadius * std::cos(angle), mRadius * std::sin(angle)});
        }
    }

    void contour(const Contour &contour) {
        const auto &points = contour.points;
        if (points.empty())
            return;
        if (points.size() == 1) {
            pointCap(points.front());
            return;
        }
        size_t count = points.size(), segments = count - (contour.closed ? 0 : 1);
        for (size_t i = 0; i < segments; ++i) {
            auto a = points[i], b = points[(i + 1) % count];
            auto tangent = direction(a, b), offset = multiply(normal(tangent), mRadius);
            if (!contour.closed && mStyle.cap == StrokeCap::Square) {
                if (i == 0)
                    a = subtract(a, multiply(tangent, mRadius));
                if (i + 1 == segments)
                    b = add(b, multiply(tangent, mRadius));
            }
            polygon(std::array{add(a, offset), subtract(a, offset), subtract(b, offset),
                               add(b, offset)});
            if (i + 1 < segments || contour.closed)
                join(points[(i + 1) % count], tangent,
                     direction(points[(i + 1) % count], points[(i + 2) % count]));
        }
        if (!contour.closed && mStyle.cap == StrokeCap::Round) {
            circle(points.front());
            circle(points.back());
        }
    }

    Mesh finish() {
        auto options = mBudget;
        options.fillRule = FillRule::NonZero;
        return tessellate(mPoints, mOffsets, options);
    }
};
} // namespace

Mesh compileStroke(std::span<const PathCommand> commands, const StrokeStyle &style,
                   const FlattenOptions &quality, const TessellationOptions &budget) {
    validate(style);
    auto flattenQuality = quality;
    flattenQuality.tolerance /= 2;
    flattenQuality.maxPoints = std::min(quality.maxPoints, budget.maxVertices);
    auto contours = flattenPath(commands, flattenQuality);
    auto outlineQuality = flattenQuality;
    if (style.space == StrokeSpace::Screen) {
        const auto &m = quality.projection;
        for (auto &contour : contours) {
            for (auto &p : contour.points)
                p = {m[0] * p[0] + m[1] * p[1] + m[2], m[3] * p[0] + m[4] * p[1] + m[5]};
            // Singular projections can collapse consecutive local vertices.
            auto end = std::unique(contour.points.begin(), contour.points.end());
            contour.points.erase(end, contour.points.end());
            if (contour.closed && contour.points.size() > 1 &&
                contour.points.front() == contour.points.back())
                contour.points.pop_back();
        }
        outlineQuality.projection = Identity;
    }
    StrokeOutline outline(style, outlineQuality, budget);
    if (style.width != 0)
        for (const auto &contour : contours)
            outline.contour(contour);
    return outline.finish();
}
} // namespace mojive::geometry2d
