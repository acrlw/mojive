#include <condition_variable>
#include <deque>
#include <future>
#include <mojive/readback.hpp>
#include <mojive/renderRuntime.hpp>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <pthread.h>
#endif

namespace mojive {
namespace {
void servicePlatformQueue() {
#ifdef __APPLE__
    // Metal swapchain creation/destruction dispatches AppKit work to the main
    // run loop. Keep servicing it while the render owner waits for the GPU.
    if (pthread_main_np())
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0, true);
#endif
}
template <class T> T receive(std::future<T> &future) {
#ifdef __APPLE__
    if (pthread_main_np())
        while (future.wait_for(std::chrono::microseconds(100)) != std::future_status::ready)
            servicePlatformQueue();
#endif
    return future.get();
}
} // namespace
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
    std::promise<void> finished;
    std::future<void> completion = finished.get_future();
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
                    finished.set_value();
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
                finished.set_value();
            });
        try {
            caps = receive(future);
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
            while (!stopping && jobs.size() >= 64) {
                space.wait_for(lock, std::chrono::milliseconds(1));
                lock.unlock();
                servicePlatformQueue();
                lock.lock();
            }
            if (stopping)
                throw std::runtime_error("Render runtime is closed");
            jobs.emplace_back([task](Renderer &renderer) { (*task)(renderer); });
        }
        ready.notify_one();
        return receive(result);
    }
    void close() {
        std::unique_lock closer(closeMutex, std::defer_lock);
        while (!closer.try_lock()) {
            servicePlatformQueue();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        space.notify_all();
        if (worker.joinable()) {
            receive(completion);
            worker.join();
        }
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
void RenderRuntime::configure(Scene scene, const SceneStyle &style) {
    return mImpl->invoke([&](Renderer &r) { return r.configure(scene, style); });
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
Target RenderRuntime::createSurface(NativeWindow window) {
    return mImpl->invoke([&](Renderer &r) { return r.createSurface(window); });
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
Scene RenderRuntime::createScene(const SceneSource &source) {
    return mImpl->invoke([&](Renderer &r) { return r.createScene(source); });
}
void RenderRuntime::setScene(Scene scene, const SceneSource &source) {
    return mImpl->invoke([&](Renderer &r) { return r.setScene(scene, source); });
}
void RenderRuntime::update(Scene scene, const SceneFrame &frame) {
    return mImpl->invoke([&](Renderer &r) { return r.update(scene, frame); });
}
void RenderRuntime::updateMesh(Scene scene, uint32_t mesh, std::span<const Vertex> vertices) {
    return mImpl->invoke([&](Renderer &r) { return r.updateMesh(scene, mesh, vertices); });
}
Target RenderRuntime::createTarget(Scene scene, Extent size, uint32_t samples) {
    return mImpl->invoke([&](Renderer &r) { return r.createTarget(scene, size, samples); });
}
void RenderRuntime::destroy(Scene scene) {
    return mImpl->invoke([&](Renderer &r) { return r.destroy(scene); });
}
} // namespace mojive
