#include "../src/bgfx/Visibility.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>

namespace {
void require(bool condition) {
    if (!condition)
        throw std::runtime_error("Conservative visibility contract failed");
}
} // namespace
int main() {
    using namespace mojive;
    const ClipFrustum clip(glm::mat4(1));
    require(clip.intersects({{0, 0, 0}, {.1f, .1f, .1f}}));
    require(clip.intersects({{1.1f, 0, 0}, {.1f, .1f, .1f}}));
    require(!clip.intersects({{1.2f, 0, 0}, {.1f, .1f, .1f}}));
    // The frustum can pass through a box even with every corner outside it.
    require(clip.intersects({{0, 0, 0}, {10, 10, 10}}));
    const MeshBounds box{{.5f, -.3f, .1f}, {1, 2, 3}};
    const Matrix shear{-2, 3, 1, 7, 1, -4, 2, -3, 0, 1, .5f, 9, 0, 0, 0, 1};
    const auto transformed = transformBounds(box, shear.data());
    for (int bits = 0; bits < 8; ++bits) {
        const glm::vec3 sign(bits & 1 ? 1 : -1, bits & 2 ? 1 : -1, bits & 4 ? 1 : -1);
        const auto point = transformBounds({box.center + box.extent * sign, {}}, shear.data());
        require(glm::all(glm::lessThanEqual(glm::abs(point.center - transformed.center),
                                            transformed.extent + glm::vec3(1e-5f))));
    }
    const ClipFrustum perspective(glm::perspectiveRH_NO(glm::radians(60.f), 1.f, .1f, 100.f));
    require(perspective.intersects({{0, 0, -2}, {.1f, .1f, .1f}}));
    require(!perspective.intersects({{0, 0, 2}, {.1f, .1f, .1f}}));
    require(!perspective.intersects({{30, 0, -2}, {.1f, .1f, .1f}}));
}
