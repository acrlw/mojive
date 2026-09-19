#include "../src/bgfx/Lod.hpp"
#include <chrono>
#include <stdexcept>

namespace {
void require(bool condition) {
    if (!condition)
        throw std::runtime_error("Optional mesh LOD lifecycle failed");
}
template <typename Predicate> void waitFor(Predicate done) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!done()) {
        require(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
} // namespace

int main() {
    using namespace mojive;
    auto mesh = std::make_shared<Mesh>();
    constexpr uint32_t side = 64;
    for (uint32_t y = 0; y <= side; ++y)
        for (uint32_t x = 0; x <= side; ++x)
            mesh->vertices.push_back({{float(x), float(y), 0}, {0, 0, 1}});
    for (uint32_t y = 0; y < side; ++y)
        for (uint32_t x = 0; x < side; ++x) {
            const auto a = y * (side + 1) + x;
            mesh->indices.insert(mesh->indices.end(),
                                 {a, a + 1, a + side + 1, a + 1, a + side + 2, a + side + 1});
        }
    std::stop_source stopped;
    stopped.request_stop();
    require(prepareMeshLods(*mesh, stopped.get_token()).empty());
    LodWorker worker;
    require(worker.idle());
    for (int repeat = 0; repeat < 3; ++repeat) {
        auto job = worker.submit(mesh);
        waitFor([&] { return job->ready.load(std::memory_order_acquire); });
        require(!job->failure && !job->levels.empty());
        waitFor([&] { return worker.idle(); });
        std::vector<std::weak_ptr<LodJob>> canceled;
        for (int i = 0; i < 32; ++i) {
            auto pending = worker.submit(mesh);
            pending->stop.request_stop();
            canceled.push_back(pending);
        }
        waitFor([&] { return worker.idle(); });
        for (const auto &pending : canceled)
            require(pending.expired());
    }
}
