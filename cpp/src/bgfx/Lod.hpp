#pragma once

#include "Visibility.hpp"
#include <mojive/MeshProcessing.hpp>

#include <atomic>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>

namespace mojive {
struct LodJob {
    std::shared_ptr<const Mesh> source;
    std::vector<MeshLod> levels;
    std::exception_ptr failure;
    std::atomic<bool> ready = false;
    std::stop_source stop;
};
// Only immutable CPU geometry crosses this boundary. The render owner installs
// GPU resources; unloading a mesh cancels queued work through weak ownership.
class LodWorker {
    std::mutex mMutex;
    std::deque<std::weak_ptr<LodJob>> mQueue;
    std::weak_ptr<LodJob> mActive;
    bool mStopping = false;
    std::atomic<bool> mRunning = false;
    std::thread mThread;

    void run() {
        for (;;) {
            std::shared_ptr<LodJob> job;
            {
                std::lock_guard lock(mMutex);
                if (mStopping || mQueue.empty()) {
                    mActive.reset();
                    mRunning = false;
                    return;
                }
                job = mQueue.front().lock();
                mQueue.pop_front();
                mActive = job;
            }
            if (!job)
                continue;
            try {
                job->levels = prepareMeshLods(*job->source, job->stop.get_token());
            } catch (...) {
                job->failure = std::current_exception();
            }
            job->ready.store(true, std::memory_order_release);
        }
    }

  public:
    bool idle() const {
        return !mRunning.load(std::memory_order_acquire);
    }
    ~LodWorker() {
        {
            std::lock_guard lock(mMutex);
            mStopping = true;
            if (auto job = mActive.lock())
                job->stop.request_stop();
        }
        if (mThread.joinable())
            mThread.join();
    }
    std::shared_ptr<LodJob> submit(std::shared_ptr<const Mesh> source) {
        auto job = std::make_shared<LodJob>();
        job->source = std::move(source);
        {
            std::lock_guard lock(mMutex);
            std::erase_if(mQueue, [](const auto &item) { return item.expired(); });
            mQueue.push_back(job);
            if (!mRunning) {
                // Idle workers exit instead of retaining a sleeping OS thread.
                // The exiting thread will not acquire mMutex again.
                if (mThread.joinable())
                    mThread.join();
                mThread = std::thread([this] { run(); });
                mRunning = true;
            }
        }
        return job;
    }
};

class LodProjection {
    glm::mat4 mRows;
    glm::vec2 mHalfPixels;

  public:
    LodProjection(const glm::mat4 &matrix, Extent size)
        : mRows(glm::transpose(matrix)), mHalfPixels(size.width * .5f, size.height * .5f) {}
    LodProjection(const CameraView &camera, Extent size)
        : LodProjection(glm::transpose(glm::make_mat4(camera.projection.data())) *
                            glm::transpose(glm::make_mat4(camera.view.data())),
                        size) {}
    float pixelsPerUnit(const MeshBounds &bounds, const float *transform) const {
        const glm::vec3 w(mRows[3]);
        const float minW =
            glm::dot(w, bounds.center) + mRows[3].w - glm::dot(glm::abs(w), bounds.extent);
        if (minW <= 1e-6f)
            return std::numeric_limits<float>::infinity();
        float pixels = 0;
        // Bound the perspective divide over the entire original bounds. This also
        // covers orthographic/asymmetric projections without extracting a FOV.
        for (int axis = 0; axis < 2; ++axis) {
            const glm::vec3 row(mRows[axis]);
            const float clip = std::abs(glm::dot(row, bounds.center) + mRows[axis].w) +
                               glm::dot(glm::abs(row), bounds.extent);
            pixels = std::max(pixels, mHalfPixels[axis] *
                                          (glm::length(row) + clip / minW * glm::length(w)) / minW);
        }
        float maxRow = 0, maxColumn = 0;
        for (int i = 0; i < 3; ++i) {
            float row = 0, column = 0;
            for (int j = 0; j < 3; ++j) {
                row += std::abs(transform[i * 4 + j]);
                column += std::abs(transform[j * 4 + i]);
            }
            maxRow = std::max(maxRow, row);
            maxColumn = std::max(maxColumn, column);
        }
        return pixels * std::sqrt(maxRow * maxColumn);
    }
};
inline uint8_t chooseLod(std::span<const float> errors, float pixelsPerUnit, uint8_t previous) {
    if (!std::isfinite(pixelsPerUnit))
        return 0;
    size_t level = std::min<size_t>(previous, errors.size());
    while (level && errors[level - 1] * pixelsPerUnit > 1.f)
        --level;
    while (level < errors.size() && errors[level] * pixelsPerUnit <= .8f)
        ++level;
    return uint8_t(level);
}
} // namespace mojive
