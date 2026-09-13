#include "GeometryArrays.hpp"
#include <mojive/Geometry2D.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nb = nanobind;
using namespace mojive::geometry2d;

namespace {
using PackedPath = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>;

std::vector<PathCommand> copyPath(PackedPath packed) {
    if (packed.shape(1) != 8)
        throw std::invalid_argument("Path commands require float64 [N, 8]");
    std::vector<PathCommand> commands;
    commands.reserve(packed.shape(0));
    for (size_t i = 0; i < packed.shape(0); ++i) {
        auto *row = packed.data() + i * 8;
        if (!std::isfinite(row[0]) || row[0] < 0 || row[0] > 5 || std::floor(row[0]) != row[0])
            throw std::invalid_argument("Unknown path verb");
        PathCommand command{PathVerb(int(row[0]))};
        std::copy_n(row + 1, 7, command.values.begin());
        commands.push_back(command);
    }
    return commands;
}

nb::tuple meshArrays(Mesh mesh) {
    return nb::make_tuple(ownedArray<double>(std::move(mesh.positions), 2),
                          ownedArray<uint32_t>(std::move(mesh.indices)),
                          ownedArray<uint32_t>(std::move(mesh.boundaryEdges), 2),
                          mesh.scratchPeakBytes);
}
} // namespace

void bindGeometry2D(nb::module_ &module) {
    module.attr("has_geometry2d") = true;
    module.def(
        "flatten_path",
        [](PackedPath packed, Affine projection, double tolerance, size_t maxPoints) {
            auto commands = copyPath(packed);
            std::vector<Contour> contours;
            {
                nb::gil_scoped_release release;
                contours = flattenPath(commands, {projection, tolerance, maxPoints});
            }
            nb::list result;
            for (auto &contour : contours) {
                result.append(nb::make_tuple(ownedArray<double>(std::move(contour.points), 2),
                                             contour.closed));
            }
            return result;
        },
        nb::arg("commands").noconvert(), nb::arg("projection") = Identity,
        nb::arg("tolerance") = 0.25, nb::arg("max_points") = 1'000'000);
    module.def(
        "tessellate_contours",
        [](nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> points,
           nb::ndarray<nb::numpy, const uint32_t, nb::ndim<1>, nb::c_contig> offsets, bool evenodd,
           size_t maxVertices, size_t maxIndices, size_t maxScratchBytes) {
            if (points.shape(1) != 2)
                throw std::invalid_argument("Contour points require float64 [N, 2]");
            if (points.shape(0) > maxVertices || offsets.size() > maxVertices + 1)
                throw std::length_error("Contour input budget exceeded");
            std::vector<Point> ownedPoints(points.shape(0));
            for (size_t i = 0; i < ownedPoints.size(); ++i)
                ownedPoints[i] = {points.data()[i * 2], points.data()[i * 2 + 1]};
            std::vector<uint32_t> ownedOffsets(offsets.data(), offsets.data() + offsets.size());
            Mesh mesh;
            {
                nb::gil_scoped_release release;
                mesh = tessellate(ownedPoints, ownedOffsets,
                                  {evenodd ? FillRule::EvenOdd : FillRule::NonZero, maxVertices,
                                   maxIndices, maxScratchBytes});
            }
            return meshArrays(std::move(mesh));
        },
        nb::arg("points").noconvert(), nb::arg("offsets").noconvert(), nb::arg("evenodd") = false,
        nb::arg("max_vertices") = 1'000'000, nb::arg("max_indices") = 3'000'000,
        nb::arg("max_scratch_bytes") = 64 * 1024 * 1024);
    module.def(
        "compile_stroke",
        [](PackedPath packed, double width, int cap, int join, double miterLimit, bool screen,
           Affine projection, double tolerance, size_t maxVertices, size_t maxIndices,
           size_t maxScratchBytes) {
            if (cap < 0 || cap > 2 || join < 0 || join > 2)
                throw std::invalid_argument("Invalid stroke cap or join");
            auto commands = copyPath(packed);
            Mesh mesh;
            {
                nb::gil_scoped_release release;
                mesh = compileStroke(commands,
                                     {width, StrokeCap(cap), StrokeJoin(join), miterLimit,
                                      screen ? StrokeSpace::Screen : StrokeSpace::Local},
                                     {projection, tolerance, maxVertices},
                                     {FillRule::NonZero, maxVertices, maxIndices, maxScratchBytes});
            }
            return meshArrays(std::move(mesh));
        },
        nb::arg("commands").noconvert(), nb::arg("width") = 1, nb::arg("cap") = 0,
        nb::arg("join") = 0, nb::arg("miter_limit") = 4, nb::arg("screen") = false,
        nb::arg("projection") = Identity, nb::arg("tolerance") = 0.25,
        nb::arg("max_vertices") = 1'000'000, nb::arg("max_indices") = 3'000'000,
        nb::arg("max_scratch_bytes") = 64 * 1024 * 1024);
}
