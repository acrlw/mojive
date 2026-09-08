#include <atomic>
#include <future>
#include <iostream>
#include <mojive/renderRuntime.hpp>
#include <stdexcept>
#include <thread>

using namespace mojive;
static void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
struct Probe {
    std::thread::id owner;
    std::atomic<bool> destroyedOnOwner{false};
    std::promise<void> entered, release;
    std::shared_future<void> gate = release.get_future().share();
};
class TestRenderer final : public Renderer {
    Probe &mProbe;
    Capabilities mCaps;
    uint64_t mTargets = 0;
    void owner() const {
        check(std::this_thread::get_id() == mProbe.owner, "Owner thread changed");
    }

  public:
    explicit TestRenderer(Probe &probe) : mProbe(probe) {
        mProbe.owner = std::this_thread::get_id();
        mCaps.backend = "test";
    }
    ~TestRenderer() override {
        mProbe.destroyedOnOwner = std::this_thread::get_id() == mProbe.owner;
    }
    const Capabilities &capabilities() const override {
        owner();
        return mCaps;
    }
    void setScene(const SceneSource &) override {
        owner();
        throw std::invalid_argument("Invalid scene fixture");
    }
    FrameStats advance() override {
        owner();
        mProbe.entered.set_value();
        mProbe.gate.wait();
        return {17};
    }
    Target createTarget(Extent, uint32_t) override {
        owner();
        return {++mTargets};
    }
    void update(const SceneFrame &) override {
        owner();
    }
    void updateMesh(uint32_t, std::span<const Vertex>) override {
        owner();
    }
    Target createSurface(NativeWindow) override {
        owner();
        return {};
    }
    void resize(Target, Extent) override {
        owner();
    }
    void destroy(Target) override {
        owner();
    }
    FrameToken render(Target, const CameraView &) override {
        owner();
        return {};
    }
    ReadbackTicket readback(FrameToken, Product, Region) override {
        owner();
        return {};
    }
    ReadbackResult poll(ReadbackTicket) override {
        owner();
        return {};
    }
    Texture targetTexture(Target) const override {
        owner();
        return {};
    }
    Texture uploadTexture(Extent, std::span<const std::byte>) override {
        owner();
        return {};
    }
    void destroy(Texture) override {
        owner();
    }
    FrameToken renderUi(const UiFrame &, Target) override {
        owner();
        return {};
    }
};
int main() {
    try {
        bool caught = false;
        try {
            RenderRuntime failed([]() -> std::unique_ptr<Renderer> {
                throw std::invalid_argument("Initialization failed");
            });
        } catch (const std::invalid_argument &) {
            caught = true;
        }
        check(caught, "Initialization did not propagate its exception");
        Probe probe;
        RenderRuntime runtime([&] { return std::make_unique<TestRenderer>(probe); });
        check(probe.owner != std::this_thread::get_id(),
              "Backend initialized on the caller thread");
        caught = false;
        try {
            runtime.setScene({});
        } catch (const std::invalid_argument &) {
            caught = true;
        }
        check(caught, "Dispatch lost the backend exception");
        std::vector<std::future<Target>> targets;
        for (int i = 0; i < 80; ++i)
            targets.push_back(
                std::async(std::launch::async, [&] { return runtime.createTarget({8, 8}); }));
        uint64_t total = 0;
        for (auto &target : targets)
            total += target.get().id;
        check(total == 80 * 81 / 2, "Concurrent submission lost targets");
        auto active = std::async(std::launch::async, [&] { return runtime.advance(); });
        probe.entered.get_future().get();
        std::promise<void> closing;
        auto closer = std::async(std::launch::async, [&] {
            closing.set_value();
            runtime.close();
        });
        closing.get_future().get();
        probe.release.set_value();
        check(active.get().drawCalls == 17, "Close canceled an accepted operation");
        closer.get();
        runtime.close();
        check(probe.destroyedOnOwner && runtime.closed(), "Close did not join backend destruction");
        caught = false;
        try {
            runtime.createTarget({8, 8});
        } catch (const std::runtime_error &) {
            caught = true;
        }
        check(caught, "Closed runtime accepted work");
        auto records = runtime.log().read().records;
        check(records.size() == 2 && records[0].threadId == records[1].threadId,
              "Native lifecycle diagnostics lost their owner");
        std::cout << "Native owner, concurrent dispatch, exceptions, and joined shutdown passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
