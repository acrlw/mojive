#pragma once
#include <bgfx/bgfx.h>
#include <mojive/Render.hpp>
#include <unordered_map>

namespace mojive {
// bgfx reuses view IDs and returns delayed timestamps. Resolve each sample using
// the ownership recorded for its GPU frame, never the view's current target.
class PassTiming {
    struct Owner {
        Target target;
        uint64_t submission = 0;
        RenderPass pass = RenderPass::Color;
    };
    struct Frame {
        uint32_t number = UINT32_MAX;
        std::array<Owner, 256> owners{};
    };
    std::array<Owner, 256> mPending{};
    std::array<Frame, 32> mHistory{};
    std::unordered_map<uint64_t, PassTimings> mTargets;

  public:
    void record(uint16_t view, Target target, uint64_t submission, RenderPass pass) {
        mTargets.try_emplace(target.id);
        mPending.at(view) = {target, submission, pass};
        bgfx::setViewName(view, renderPassNames[size_t(pass)]);
    }
    void collect(uint32_t frameNumber, const bgfx::Stats &stats) {
        auto &frame = mHistory[frameNumber % mHistory.size()];
        frame.number = frameNumber;
        frame.owners = mPending;
        mPending.fill({});
        for (uint16_t i = 0; i < stats.numViews; ++i) {
            const auto &view = stats.viewStats[i];
            if (view.view >= frame.owners.size())
                continue;
            // getStats() exposes the submit buffer, which retains the previous
            // rendered frame's CPU statistics after bgfx swaps its frame pair.
            const auto &cpuFrame = mHistory[(frameNumber - 1) % mHistory.size()];
            const auto &cpu = cpuFrame.owners[view.view];
            auto cpuTarget = mTargets.find(cpu.target.id);
            if (cpuFrame.number == frameNumber - 1 && cpu.submission &&
                cpuTarget != mTargets.end() && stats.cpuTimerFreq > 0 &&
                view.cpuTimeEnd >= view.cpuTimeBegin) {
                auto &out = cpuTarget->second;
                if (out.cpuSubmission != cpu.submission) {
                    out.cpuSubmission = cpu.submission;
                    out.cpuMs.fill(0);
                    out.cpuMask = 0;
                }
                out.cpuMs[size_t(cpu.pass)] +=
                    1000.0 * double(view.cpuTimeEnd - view.cpuTimeBegin) / stats.cpuTimerFreq;
                out.cpuMask |= 1u << size_t(cpu.pass);
            }
            const auto &history = mHistory[view.gpuFrameNum % mHistory.size()];
            if (history.number != view.gpuFrameNum || stats.gpuTimerFreq <= 0 ||
                view.gpuTimeEnd <= view.gpuTimeBegin)
                continue;
            const auto &gpu = history.owners[view.view];
            auto it = mTargets.find(gpu.target.id);
            if (!gpu.submission || it == mTargets.end() ||
                gpu.submission < it->second.gpuSubmission)
                continue;
            auto &out = it->second;
            if (out.gpuSubmission != gpu.submission) {
                out.gpuSubmission = gpu.submission;
                out.gpuMs.fill(0);
                out.gpuMask = 0;
            }
            // A view may report the same delayed sample on successive frames.
            // Store per-view samples before summing to avoid counting it twice.
            auto &sample = mSamples[view.view];
            if (sample.target == gpu.target && sample.submission == gpu.submission)
                out.gpuMs[size_t(sample.pass)] -= sample.ms;
            sample = {gpu.target, gpu.submission, gpu.pass,
                      1000.0 * double(view.gpuTimeEnd - view.gpuTimeBegin) / stats.gpuTimerFreq};
            out.gpuMs[size_t(gpu.pass)] += sample.ms;
            out.gpuMask |= 1u << size_t(gpu.pass);
        }
    }
    PassTimings get(Target target) const {
        const auto it = mTargets.find(target.id);
        return it == mTargets.end() ? PassTimings{} : it->second;
    }
    void erase(Target target) {
        mTargets.erase(target.id);
    }

  private:
    struct Sample {
        Target target;
        uint64_t submission = 0;
        RenderPass pass = RenderPass::Color;
        double ms = 0;
    };
    std::array<Sample, 256> mSamples{};
};
} // namespace mojive
