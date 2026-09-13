#include <mojive/Geometry2D.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace mojive::geometry2d;

int main() {
    try {
        const std::array<Point, 8> points{Point{0, 0}, {10, 0}, {10, 10}, {0, 10},
                                          {3, 3},      {7, 3},  {7, 7},   {3, 7}};
        const std::array<uint32_t, 3> offsets{0, 4, 8};
        auto mesh = tessellate(points, offsets, {FillRule::EvenOdd});
        double area = 0;
        for (size_t i = 0; i < mesh.indices.size(); i += 3) {
            auto a = mesh.positions[mesh.indices[i]], b = mesh.positions[mesh.indices[i + 1]],
                 c = mesh.positions[mesh.indices[i + 2]];
            area += ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2;
        }
        if (std::abs(area - 84) > 1e-5 || mesh.boundaryEdges.size() != 8)
            throw std::runtime_error("Hole fill topology changed");
        // Sweep through allocation failures, including upstream constructor and
        // output buffers. Sanitizer builds also verify reclamation of abandoned blocks.
        for (size_t budget = 1; budget < mesh.scratchPeakBytes; budget += 173) {
            bool rejected = false;
            try {
                tessellate(points, offsets, {FillRule::EvenOdd, 100, 100, budget});
            } catch (const std::runtime_error &) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("Scratch budget was not enforced");
        }
        auto recovered = tessellate(points, offsets, {FillRule::EvenOdd});
        if (recovered.indices != mesh.indices)
            throw std::runtime_error("Failed compilation changed later output");
        std::cout << "Geometry2D topology and allocation failure contracts passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
