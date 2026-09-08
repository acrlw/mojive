#include <condition_variable>
#include <deque>
#include <future>
#include <mojive/readback.hpp>
#include <mojive/renderRuntime.hpp>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace mojive {
class RenderRuntime::Impl {
  public:
    Log log;
    Capabilities caps;
    mutable std::mutex mutex;
    std::mutex closeMutex;
    std::condition_variable ready, space;
    std::deque<std::function<void(Renderer &)>> jobs;
    std::thread worker;
    bool stopping = false;
    explicit Impl(Factory factory, const LogOptions &options) : log(options) {
        if (!factory)
            throw std::invalid_argument("Missing renderer factory");
        std::promise<Capabilities> initialized;
        auto future = initialized.get_future();
        worker = std::thread(
            [this, factory = std::move(factory), initialized = std::move(initialized)]() mutable {
                std::unique_ptr<Renderer> renderer;
                try {
                    renderer = factory();
                    if (!renderer)
                        throw std::runtime_error("Renderer factory returned no backend");
                    initialized.set_value(renderer->capabilities());
                } catch (...) {
                    initialized.set_exception(std::current_exception());
                    return;
                }
                diagnostic("Render owner started");
                for (;;) {
                    std::function<void(Renderer &)> job;
                    {
                        std::unique_lock lock(mutex);
                        ready.wait(lock, [this] { return stopping || !jobs.empty(); });
                        if (jobs.empty())
                            break;
                        job = std::move(jobs.front());
                        jobs.pop_front();
                    }
                    space.notify_one();
                    job(*renderer);
                }
                renderer.reset();
                diagnostic("Render owner stopped");
            });
        try {
            caps = future.get();
        } catch (...) {
            worker.join();
            throw;
        }
    }
    void diagnostic(const char *message) noexcept {
        try {
            log.publish(LogLevel::Info, "render.runtime", message);
        } catch (...) {
            // Diagnostics must not strand callers waiting for a native result.
        }
    }
    template <class F> auto invoke(F &&function) {
        using Result = std::invoke_result_t<F, Renderer &>;
        auto task =
            std::make_shared<std::packaged_task<Result(Renderer &)>>(std::forward<F>(function));
        auto result = task->get_future();
        {
            std::unique_lock lock(mutex);
            space.wait(lock, [this] { return stopping || jobs.size() < 64; });
            if (stopping)
                throw std::runtime_error("Render runtime is closed");
            jobs.emplace_back([task](Renderer &renderer) { (*task)(renderer); });
        }
        ready.notify_one();
        return result.get();
    }
    void close() {
        std::lock_guard closer(closeMutex);
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        space.notify_all();
        if (worker.joinable())
            worker.join();
        log.close();
    }
};
RenderRuntime::RenderRuntime(Factory factory, const LogOptions &options)
    : mImpl(std::make_unique<Impl>(std::move(factory), options)) {}
RenderRuntime::~RenderRuntime() {
    close();
}
void RenderRuntime::close() {
    mImpl->close();
}
bool RenderRuntime::closed() const {
    std::lock_guard lock(mImpl->mutex);
    return mImpl->stopping;
}
Log &RenderRuntime::log() {
    return mImpl->log;
}
const Capabilities &RenderRuntime::capabilities() const {
    return mImpl->caps;
}
void RenderRuntime::setScene(const SceneSource &scene) {
    return mImpl->invoke([&](Renderer &r) { return r.setScene(scene); });
}
void RenderRuntime::update(const SceneFrame &frame) {
    return mImpl->invoke([&](Renderer &r) { return r.update(frame); });
}
void RenderRuntime::updateMesh(uint32_t mesh, std::span<const Vertex> vertices) {
    return mImpl->invoke([&](Renderer &r) { return r.updateMesh(mesh, vertices); });
}
Target RenderRuntime::createTarget(Extent size, uint32_t samples) {
    return mImpl->invoke([&](Renderer &r) { return r.createTarget(size, samples); });
}
Target RenderRuntime::createSurface(NativeWindow) {
    throw std::invalid_argument("Offscreen runtime cannot own platform windows");
}
void RenderRuntime::resize(Target target, Extent size) {
    return mImpl->invoke([&](Renderer &r) { return r.resize(target, size); });
}
void RenderRuntime::destroy(Target target) {
    return mImpl->invoke([&](Renderer &r) { return r.destroy(target); });
}
FrameToken RenderRuntime::render(Target target, const CameraView &camera) {
    validateCamera(camera);
    return mImpl->invoke([&](Renderer &r) { return r.render(target, camera); });
}
ReadbackTicket RenderRuntime::readback(FrameToken frame, Product product, Region region) {
    return mImpl->invoke([&](Renderer &r) { return r.readback(frame, product, region); });
}
ReadbackResult RenderRuntime::poll(ReadbackTicket ticket) {
    return mImpl->invoke([&](Renderer &r) { return r.poll(ticket); });
}
FrameStats RenderRuntime::advance() {
    return mImpl->invoke([&](Renderer &r) { return r.advance(); });
}
Texture RenderRuntime::targetTexture(Target target) const {
    return mImpl->invoke([&](Renderer &r) { return r.targetTexture(target); });
}
Texture RenderRuntime::uploadTexture(Extent size, std::span<const std::byte> bytes) {
    return mImpl->invoke([&](Renderer &r) { return r.uploadTexture(size, bytes); });
}
void RenderRuntime::destroy(Texture texture) {
    return mImpl->invoke([&](Renderer &r) { return r.destroy(texture); });
}
FrameToken RenderRuntime::renderUi(const UiFrame &frame, Target output) {
    return mImpl->invoke([&](Renderer &r) { return r.renderUi(frame, output); });
}
ReadbackResult RenderRuntime::wait(ReadbackTicket ticket) {
    return mImpl->invoke([&](Renderer &r) { return waitForReadback(r, ticket); });
}
} // namespace mojive
