#include "../src/bgfx/Lod.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <random>
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
    const CameraView unitCamera;
    const MeshBounds receiverBounds{{0, 0, 0}, {10, 10, .1f}};
    const ShadowReceiverVolume down(unitCamera, {0, 0, -1}, &receiverBounds);
    require(down.intersects({{0, 0, 20}, {.1f, .1f, .1f}}));
    require(!down.intersects({{2, 0, 20}, {.1f, .1f, .1f}}));
    require(!down.intersects({{0, 0, -2}, {.1f, .1f, .1f}}));
    require(down.intersects({{1.1f, 0, 20}, {.1f, .1f, .1f}}));
    const ShadowReceiverVolume slant(unitCamera, {1, 0, -1}, &receiverBounds);
    const ShadowReceiverVolume unbounded(unitCamera, {1, 0, -1}, nullptr);
    require(slant.intersects({{-20, 0, 20}, {.1f, .1f, .1f}}));
    require(!slant.intersects({{-2, 0, .2f}, {.01f, .01f, .01f}}));
    require(unbounded.intersects({{-2, 0, .2f}, {.01f, .01f, .01f}}));
    const ShadowReceiverVolume parallel(unitCamera, {1, 0, 0}, &receiverBounds);
    require(parallel.intersects({{-20, 0, 0}, {}}));
    require(!parallel.intersects({{-20, 0, 1}, {}}));
    // Every upstream point on a ray reaching an actual visible receiver must
    // survive, including tilted lights and perspective frustum boundaries.
    std::mt19937 random(42);
    std::uniform_real_distribution<float> sample(-1, 1);
    CameraView testCamera;
    auto projection = glm::transpose(glm::perspectiveRH_NO(glm::radians(60.f), 1.f, .1f, 100.f));
    std::copy_n(&projection[0][0], 16, testCamera.projection.begin());
    const MeshBounds perspectiveReceivers{{0, 0, -10}, {5, 5, 1}};
    for (int i = 0; i < 10000; ++i) {
        const glm::vec3 direction(sample(random), sample(random), sample(random));
        const glm::vec3 receiver(sample(random) * 4, sample(random) * 4, -10 + sample(random));
        const float distance = 100 * std::abs(sample(random));
        const ShadowReceiverVolume volume(testCamera, direction, &perspectiveReceivers);
        require(volume.intersects({receiver - distance * direction, {.001f, .001f, .001f}}));
    }
    const Matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const LodProjection ortho(glm::mat4(1), {200, 100});
    require(std::abs(ortho.pixelsPerUnit(box, identity.data()) - 100) < 1e-4f);
    const LodProjection zoom(glm::mat4(1), {400, 200});
    require(std::abs(zoom.pixelsPerUnit(box, identity.data()) - 200) < 1e-4f);
    auto scaled = identity;
    scaled[0] = -3;
    require(std::abs(ortho.pixelsPerUnit(box, scaled.data()) - 300) < 1e-4f);
    const LodProjection projected(glm::perspectiveRH_NO(glm::radians(60.f), 1.f, .1f, 100.f),
                                  {200, 200});
    require(!std::isfinite(projected.pixelsPerUnit({{0, 0, 0}, {1, 1, 1}}, identity.data())));
    const float near = projected.pixelsPerUnit({{0, 0, -2}, {.1f, .1f, .1f}}, identity.data());
    const float far = projected.pixelsPerUnit({{0, 0, -20}, {.1f, .1f, .1f}}, identity.data());
    require(near > far * 9);
    const std::array<float, 3> errors{.01f, .1f, 1.f};
    require(chooseLod(errors, 7.9f, 0) == 2);
    require(chooseLod(errors, 9.f, 1) == 1);
    require(chooseLod(errors, 9.f, 2) == 2);
    require(chooseLod(errors, 10.1f, 2) == 1);
    require(chooseLod(errors, 101.f, 3) == 0);
    require(chooseLod(errors, std::numeric_limits<float>::infinity(), 3) == 0);
}
