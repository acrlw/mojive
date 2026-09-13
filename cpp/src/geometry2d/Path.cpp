#include <mojive/Geometry2D.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace mojive::geometry2d {
namespace {
Point midpoint(Point a, Point b) {
    return {a[0] / 2 + b[0] / 2, a[1] / 2 + b[1] / 2};
}

class Flattener {
    const FlattenOptions &mOptions;
    std::vector<Contour> mContours;
    size_t mPoints = 0;
    bool mActive = false;

    Point project(Point p) const {
        const auto &m = mOptions.projection;
        Point result{m[0] * p[0] + m[1] * p[1] + m[2], m[3] * p[0] + m[4] * p[1] + m[5]};
        if (!std::isfinite(result[0]) || !std::isfinite(result[1]))
            throw std::overflow_error("Path projection exceeds finite coordinates");
        return result;
    }

    void append(Point point) {
        project(point);
        auto &points = mContours.back().points;
        if (!points.empty() && points.back() == point)
            return;
        if (mPoints == mOptions.maxPoints)
            throw std::length_error("Path point budget exceeded");
        ++mPoints;
        points.push_back(point);
    }

    double distance(Point p, Point a, Point b) const {
        const auto &m = mOptions.projection;
        double px = p[0] - a[0], py = p[1] - a[1];
        double bx = b[0] - a[0], by = b[1] - a[1];
        // Curvature quality depends on displacement, never viewport translation.
        p = {m[0] * px + m[1] * py, m[3] * px + m[4] * py};
        double dx = m[0] * bx + m[1] * by, dy = m[3] * bx + m[4] * by;
        double length = std::hypot(dx, dy);
        if (!std::isfinite(length) || !std::isfinite(p[0]) || !std::isfinite(p[1]))
            throw std::overflow_error("Path segment exceeds finite extent");
        if (length == 0)
            return std::hypot(p[0], p[1]);
        dx /= length;
        dy /= length;
        double along = std::clamp(p[0] * dx + p[1] * dy, 0.0, length);
        return std::hypot(p[0] - along * dx, p[1] - along * dy);
    }

    void cubic(Point p0, Point p1, Point p2, Point p3, unsigned depth = 0) {
        if (std::max(distance(p1, p0, p3), distance(p2, p0, p3)) <= mOptions.tolerance) {
            append(p3);
            return;
        }
        if (depth == 32)
            throw std::length_error("Path curve exceeds subdivision budget");
        auto a = midpoint(p0, p1), b = midpoint(p1, p2), c = midpoint(p2, p3);
        auto d = midpoint(a, b), e = midpoint(b, c), middle = midpoint(d, e);
        cubic(p0, a, d, middle, depth + 1);
        cubic(middle, e, c, p3, depth + 1);
    }

    void arc(const std::array<double, 7> &v) {
        if (v[2] < 0 || v[3] < 0)
            throw std::invalid_argument("Arc radii must be nonnegative");
        double cosine = std::cos(v[4]), sine = std::sin(v[4]);
        Point u{cosine * v[2], sine * v[2]}, w{-sine * v[3], cosine * v[3]};
        const auto &m = mOptions.projection;
        // The Frobenius norm bounds all projected ellipse radii, including shear.
        double radius =
            std::hypot(std::hypot(m[0] * u[0] + m[1] * u[1], m[3] * u[0] + m[4] * u[1]),
                       std::hypot(m[0] * w[0] + m[1] * w[1], m[3] * w[0] + m[4] * w[1]));
        if (!std::isfinite(radius))
            throw std::overflow_error("Arc projection exceeds finite extent");
        double step = std::numbers::pi / 2;
        if (radius > mOptions.tolerance)
            step = std::min(step, 2 * std::acos(std::max(-1.0, 1 - mOptions.tolerance / radius)));
        double count = std::max(1.0, std::ceil(std::abs(v[6]) / step));
        if (!std::isfinite(count) || count > mOptions.maxPoints - mPoints)
            throw std::length_error("Arc exceeds path point budget");
        auto segments = size_t(count);
        for (size_t i = 0; i <= segments; ++i) {
            double angle = v[5] + v[6] * (double(i) / segments);
            append({v[0] + u[0] * std::cos(angle) + w[0] * std::sin(angle),
                    v[1] + u[1] * std::cos(angle) + w[1] * std::sin(angle)});
        }
    }

  public:
    explicit Flattener(const FlattenOptions &options) : mOptions(options) {
        if (!std::isfinite(options.tolerance) || options.tolerance <= 0 || !options.maxPoints)
            throw std::invalid_argument("Path tolerance and point budget must be positive");
        for (double value : options.projection)
            if (!std::isfinite(value))
                throw std::invalid_argument("Path projection must be finite");
    }

    std::vector<Contour> run(std::span<const PathCommand> commands) {
        for (const auto &command : commands) {
            const auto &v = command.values;
            for (double value : v)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Path coordinates must be finite");
            if (command.verb == PathVerb::Move) {
                mContours.push_back({});
                mActive = true;
                append({v[0], v[1]});
                continue;
            }
            if (!mActive)
                throw std::invalid_argument("A subpath must start with move");
            auto current = mContours.back().points.back();
            switch (command.verb) {
            case PathVerb::Line:
                append({v[0], v[1]});
                break;
            case PathVerb::Quadratic:
                cubic(current,
                      {current[0] / 3 + v[0] * (2.0 / 3), current[1] / 3 + v[1] * (2.0 / 3)},
                      {v[2] / 3 + v[0] * (2.0 / 3), v[3] / 3 + v[1] * (2.0 / 3)}, {v[2], v[3]});
                break;
            case PathVerb::Cubic:
                cubic(current, {v[0], v[1]}, {v[2], v[3]}, {v[4], v[5]});
                break;
            case PathVerb::Arc:
                arc(v);
                break;
            case PathVerb::Close: {
                auto &contour = mContours.back();
                contour.closed = true;
                if (contour.points.size() > 1 && contour.points.front() == contour.points.back())
                    contour.points.pop_back();
                mActive = false;
                break;
            }
            default:
                throw std::invalid_argument("Unknown path verb");
            }
        }
        return std::move(mContours);
    }
};
} // namespace

std::vector<Contour> flattenPath(std::span<const PathCommand> commands,
                                 const FlattenOptions &options) {
    return Flattener(options).run(commands);
}
} // namespace mojive::geometry2d
