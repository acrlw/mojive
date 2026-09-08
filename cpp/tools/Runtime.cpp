#include "RendererFactory.hpp"
#include "SceneStream.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mojive/Readback.hpp>
#include <mujoco/mujoco.h>
#include <mutex>
#include <thread>

using namespace mojive;
using Clock = std::chrono::steady_clock;
static double milliseconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double percentile(std::vector<double> values, double p) {
    if (values.empty())
        return 0;
    std::sort(values.begin(), values.end());
    return values[size_t(p * (values.size() - 1))];
}
struct Snapshot {
    std::vector<Matrix> transforms;
    uint64_t sequence = 0;
    Clock::time_point produced = Clock::now();
};
// Three owned, preallocated buffers. The consumer never observes the producer's
// writable storage or mjData, and the lock only swaps completed snapshots.
struct Mailbox {
    Snapshot latest;
    std::mutex mutex;
    explicit Mailbox(size_t count) {
        latest.transforms.resize(count);
    }
    void publish(Snapshot &producer) {
        std::lock_guard lock(mutex);
        std::swap(latest, producer);
    }
    void consume(Snapshot &consumer) {
        std::lock_guard lock(mutex);
        if (latest.sequence > consumer.sequence)
            std::swap(latest, consumer);
    }
};
struct Physics {
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model{nullptr, mj_deleteModel};
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data{nullptr, mj_deleteData};
    std::vector<int> geoms;
    std::vector<Matrix> local;
    uint64_t sequence = 0;
    std::vector<double> stepTimes;
    Physics(const char *path, const probe::Trajectory &scene) {
        char error[1024]{};
        model.reset(mj_loadXML(path, nullptr, error, sizeof(error)));
        if (!model)
            throw std::runtime_error(error);
        data.reset(mj_makeData(model.get()));
        if (!data)
            throw std::runtime_error("Cannot allocate physics state");
        mj_forward(model.get(), data.get());
        local = scene.frames[0];
        for (size_t i = 0; i < local.size(); ++i) {
            auto seg = scene.source.instances[i].segmentation;
            if (seg[1] != mjOBJ_GEOM || seg[0] < 0 || seg[0] >= model->ngeom)
                throw std::runtime_error("Live fixture requires MuJoCo geom identities");
            geoms.push_back(seg[0]);
            auto *r = data->geom_xmat + 9 * seg[0], *p = data->geom_xpos + 3 * seg[0];
            auto initial = local[i];
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 4; ++col) {
                    double value = 0;
                    for (int k = 0; k < 3; ++k)
                        value += r[k * 3 + row] * (initial[k * 4 + col] - (col == 3 ? p[k] : 0));
                    local[i][row * 4 + col] = float(value);
                }
        }
        stepTimes.reserve(200000);
    }
    void snapshot(Snapshot &out) {
        for (size_t i = 0; i < local.size(); ++i) {
            const auto *r = data->geom_xmat + 9 * geoms[i], *p = data->geom_xpos + 3 * geoms[i];
            auto &m = out.transforms[i];
            m = identity();
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 4; ++col) {
                    double value = col == 3 ? p[row] : 0;
                    for (int k = 0; k < 3; ++k)
                        value += r[row * 3 + k] * local[i][k * 4 + col];
                    if (!std::isfinite(value))
                        throw std::runtime_error("Non-finite live transform");
                    m[row * 4 + col] = float(value);
                }
        }
        out.sequence = sequence;
        out.produced = Clock::now();
    }
    void step(Snapshot &out) {
        auto begin = Clock::now();
        mj_step(model.get(), data.get());
        ++sequence;
        snapshot(out);
        stepTimes.push_back(milliseconds(begin, Clock::now()));
    }
    void reset() {
        mj_resetData(model.get(), data.get());
        mj_forward(model.get(), data.get());
        sequence = 0;
        stepTimes.clear();
    }
};
struct Pending {
    ReadbackTicket ticket;
    FrameToken frame;
    Clock::time_point request, produced;
};
int main(int argc, char **argv) {
    try {
        if (argc != 8)
            throw std::invalid_argument("Usage: runtime shaders trajectory model output backend "
                                        "serial|parallel pick|record|multi");
        auto scene = probe::load(argv[2]);
        Physics physics(argv[3], scene);
        Snapshot current{scene.frames[0]}, producer{scene.frames[0]};
        float maxError = 0;
        // Check every recorded pose, including split capsules, against the Python
        // export before timing the independently owned native simulation.
        for (const auto &expected : scene.frames) {
            physics.snapshot(current);
            for (size_t i = 0; i < expected.size(); ++i)
                for (size_t j = 0; j < 16; ++j)
                    maxError =
                        std::max(maxError, std::abs(expected[i][j] - current.transforms[i][j]));
            physics.step(producer);
        }
        if (maxError > 0.0001f)
            throw std::runtime_error("Native/Python fixture pose mismatch: " +
                                     std::to_string(maxError));
        physics.reset();
        physics.snapshot(current);
        std::string backend = argv[5], execution = argv[6], mode = argv[7];
        if (execution != "serial" && execution != "parallel")
            throw std::invalid_argument("Unknown execution mode");
        if (mode != "pick" && mode != "record" && mode != "multi")
            throw std::invalid_argument("Unknown output mode");
        auto renderer = probe::makeRenderer({{}, argv[1]}, backend);
        renderer->setScene(scene.source);
        auto target = renderer->createTarget({1920, 1080}, 4);
        Target peer;
        if (mode == "multi")
            peer = renderer->createTarget({640, 480}, 4);
        auto peerCamera = scene.camera;
        peerCamera.view = lookAt({-18, -22, 18}, {0, 0, .5f}, {0, 0, 1});
        peerCamera.projection[0] *= (16.0f / 9) / (4.0f / 3);
        peerCamera.revision = 2;
        for (int i = 0; i < 90; ++i) {
            renderer->update({1, 0, current.transforms});
            renderer->render(target, scene.camera);
            renderer->advance();
        }
        auto initial = renderer->render(target, scene.camera);
        waitForReadback(*renderer, renderer->readback(initial, Product::Color));
        Mailbox mailbox(scene.source.instances.size());
        std::exception_ptr failure;
        const auto start = Clock::now();
        const auto stepDuration = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(physics.model->opt.timestep));
        auto nextStep = start + stepDuration;
        std::jthread worker;
        if (execution == "parallel")
            worker = std::jthread([&](std::stop_token stop) {
                try {
                    auto due = start + stepDuration;
                    while (!stop.stop_requested()) {
                        std::this_thread::sleep_until(due);
                        if (stop.stop_requested())
                            break;
                        physics.step(producer);
                        mailbox.publish(producer);
                        due += stepDuration;
                        // Saturated physics advances sequentially without dropping steps.
                    }
                } catch (...) {
                    failure = std::current_exception();
                }
            });
        std::vector<double> cpu, age, deadline, pickLatency, recordLatency, completedAge;
        std::deque<Pending> pending;
        uint64_t frames = 0, picks = 0, records = 0, skipped = 0, lastSequence = 0, changed = 0;
        auto consume = [&] {
            for (auto i = pending.begin(); i != pending.end();) {
                auto result = renderer->poll(i->ticket);
                if (result.state == ReadbackState::Pending) {
                    ++i;
                    continue;
                }
                if (result.state != ReadbackState::Ready || result.frame != i->frame)
                    throw std::runtime_error("Live output lost snapshot provenance");
                auto now = Clock::now();
                (result.image.product == Product::ObjectId ? pickLatency : recordLatency)
                    .push_back(milliseconds(i->request, now));
                completedAge.push_back(milliseconds(i->produced, now));
                i = pending.erase(i);
            }
        };
        auto enqueue = [&](FrameToken frame, Product product, Region region = {}) {
            const auto requested = Clock::now();
            pending.push_back(
                {renderer->readback(frame, product, region), frame, requested, current.produced});
        };
        constexpr double seconds = 10, fps = 120;
        while (milliseconds(start, Clock::now()) < seconds * 1000) {
            auto begin = Clock::now();
            auto scheduled = start + std::chrono::duration_cast<Clock::duration>(
                                         std::chrono::duration<double>(frames / fps));
            deadline.push_back(std::max(0.0, milliseconds(scheduled, begin)));
            if (execution == "parallel")
                mailbox.consume(current);
            else {
                // Bound catch-up so an overloaded simulation cannot starve input forever.
                for (int i = 0; i < 4 && Clock::now() >= nextStep; ++i) {
                    physics.step(current);
                    nextStep += stepDuration;
                }
            }
            if (current.sequence < lastSequence)
                throw std::runtime_error("Snapshot sequence regressed");
            if (current.sequence > lastSequence)
                ++changed;
            lastSequence = current.sequence;
            age.push_back(milliseconds(current.produced, Clock::now()));
            consume();
            const bool requestPick = mode == "pick" || frames % 2 == 0;
            const bool requestRecord = mode != "pick" && frames % 4 == 0;
            const bool requestPeer = peer.id && frames % 4 == 0;
            const size_t needed =
                size_t(requestPick) + 3 * size_t(requestRecord) + size_t(requestPeer);
            auto queueDeadline = Clock::now() + std::chrono::seconds(10);
            while (pending.size() + needed > 8) {
                renderer->advance();
                consume();
                if (Clock::now() > queueDeadline)
                    throw std::runtime_error("Live output backpressure timed out");
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            renderer->update({1, current.sequence, current.transforms});
            auto token = renderer->render(target, scene.camera);
            if (requestPick) {
                enqueue(token, Product::ObjectId, {960, 540, 1, 1});
                ++picks;
            }
            if (requestRecord) {
                enqueue(token, Product::Color);
                enqueue(token, Product::MetricDepth);
                enqueue(token, Product::Segmentation);
                ++records;
            }
            if (peer.id) {
                auto peerToken = renderer->render(peer, peerCamera);
                if (requestPeer)
                    enqueue(peerToken, Product::Color);
            }
            renderer->advance();
            consume();
            cpu.push_back(milliseconds(begin, Clock::now()));
            ++frames;
            std::this_thread::sleep_until(start + std::chrono::duration_cast<Clock::duration>(
                                                      std::chrono::duration<double>(frames / fps)));
        }
        auto end = Clock::now();
        if (worker.joinable()) {
            worker.request_stop();
            worker.join();
        }
        if (failure)
            std::rethrow_exception(failure);
        auto drainDeadline = Clock::now() + std::chrono::seconds(10);
        while (!pending.empty()) {
            renderer->advance();
            consume();
            if (Clock::now() > drainDeadline)
                throw std::runtime_error("Live readback drain timed out");
        }
        if (physics.sequence < 100 || changed < 30)
            throw std::runtime_error("Live simulation did not progress");
        std::filesystem::path output = argv[4];
        std::filesystem::create_directories(output.parent_path());
        std::ofstream report(output);
        auto elapsed = milliseconds(start, end) / 1000;
        report << std::fixed << std::setprecision(4) << "{\n\"backend\":\"" << backend
               << "\",\n\"execution\":\"" << execution << "\",\n\"mode\":\"" << mode
               << "\",\n\"mujoco\":\"" << mj_versionString() << "\",\n\"seconds\":" << elapsed
               << ",\n\"frames\":" << frames << ",\n\"fps\":" << frames / elapsed
               << ",\n\"physics_steps\":" << physics.sequence
               << ",\n\"physics_steps_per_second\":" << physics.sequence / elapsed
               << ",\n\"physics_cpu_p95_ms\":" << percentile(physics.stepTimes, .95)
               << ",\n\"frame_cpu_p95_ms\":" << percentile(cpu, .95)
               << ",\n\"frame_cpu_p99_ms\":" << percentile(cpu, .99)
               << ",\n\"snapshot_age_p95_ms\":" << percentile(age, .95)
               << ",\n\"readback_state_age_p95_ms\":" << percentile(completedAge, .95)
               << ",\n\"frame_lateness_p95_ms\":" << percentile(deadline, .95)
               << ",\n\"pick_latency_p95_ms\":" << percentile(pickLatency, .95)
               << ",\n\"record_latency_p95_ms\":" << percentile(recordLatency, .95)
               << ",\n\"pick_requests\":" << picks << ",\n\"record_groups\":" << records
               << ",\n\"skipped_requests\":" << skipped
               << ",\n\"pose_reference_max_error\":" << maxError << "\n}\n";
        std::cout << backend << " " << execution << " " << mode << ": " << frames / elapsed
                  << " FPS, " << physics.sequence / elapsed << " steps/s, CPU P95 "
                  << percentile(cpu, .95) << " ms\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
