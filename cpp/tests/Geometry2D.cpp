#include <cmath>
#include <iostream>
#include <limits>
#include <mojive/Geometry2D.hpp>
#include <stdexcept>

using namespace mojive::geometry2d;

void require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
template <class Function> void rejects(Function function) {
    bool failed = false;
    try {
        function();
    } catch (const std::exception &) {
        failed = true;
    }
    require(failed, "Invalid input or exhausted budget was accepted");
}

int main() {
    try {
        const std::array line{PathCommand{PathVerb::Move, {0, 0}},
                              PathCommand{PathVerb::Line, {10, 0}},
                              PathCommand{PathVerb::Line, {10, 10}}, PathCommand{PathVerb::Close}};
        auto contours = flattenPath(line);
        require(contours.size() == 1 && contours[0].closed, "Closed contour lost");
        require(contours[0].points.size() == 3, "Polyline topology changed");
        const std::array curve{PathCommand{PathVerb::Move, {0, 0}},
                               PathCommand{PathVerb::Cubic, {0, 50, 50, 50, 50, 0}}};
        auto normal = flattenPath(curve);
        auto enlarged = flattenPath(curve, {{20, 0, 0, 0, 20, 0}});
        require(enlarged[0].points.size() > normal[0].points.size(),
                "Projection quality not refined");
        require(normal[0].points.back() == Point{50, 0}, "Curve endpoint changed");
        const std::array cusp{PathCommand{PathVerb::Move, {0, 0}},
                              PathCommand{PathVerb::Cubic, {30, 0, -30, 0, 0, 0}}};
        require(flattenPath(cusp)[0].points.size() > 2, "Collinear cusp collapsed");
        rejects([&] { flattenPath(curve, {Identity, 0.25, 2}); });
        rejects([&] { flattenPath(curve, {Identity, 0.0}); });
        const std::array invalid{PathCommand{PathVerb::Line, {1, 1}}};
        rejects([&] { flattenPath(invalid); });
        auto nan = line;
        nan[0].values[0] = std::numeric_limits<double>::quiet_NaN();
        rejects([&] { flattenPath(nan); });
        for (auto cap : {StrokeCap::Butt, StrokeCap::Round, StrokeCap::Square}) {
            for (auto join : {StrokeJoin::Miter, StrokeJoin::Round, StrokeJoin::Bevel}) {
                StrokeStyle style{4, cap, join};
                auto mesh = compileStroke(curve, style);
                require(!mesh.indices.empty(), "Stroke unexpectedly empty");
                rejects([&] { compileStroke(curve, style, {}, {FillRule::NonZero, 8}); });
                rejects([&] { compileStroke(curve, style, {}, {FillRule::NonZero, 10000, 3}); });
                for (size_t budget = 512; budget < 20000; budget += 2048)
                    rejects([&] {
                        compileStroke(curve, style, {}, {FillRule::NonZero, 10000, 30000, budget});
                    });
                require(!compileStroke(curve, style).indices.empty(),
                        "Stroke compiler did not recover after budget failure");
            }
        }
        std::cout << "Geometry2D CPU path contracts passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
