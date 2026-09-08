#include "scene_stream.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mojive/backends/bgfx.hpp>
#include <mojive/readback.hpp>
#include <numeric>
#include <thread>

using namespace mojive;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double percentile(std::vector<double> values, double p) {
    if (values.empty())
        return 0;
    std::sort(values.begin(), values.end());
    double at = p * (values.size() - 1);
    size_t low = at;
    return values[low] + (values[std::min(low + 1, values.size() - 1)] - values[low]) * (at - low);
}
struct Pending {
    ReadbackTicket ticket;
    Clock::time_point start;
};
int main(int argc, char **argv) {
    try {
        if (argc < 8)
            throw std::invalid_argument(
                "Usage: benchmark shaders trajectory output width height seconds mode [fps-limit]");
        auto scene = probe::load(argv[2]);
        Extent size{uint32_t(std::stoul(argv[4])), uint32_t(std::stoul(argv[5]))};
        double seconds = std::stod(argv[6]), fps_limit = argc > 8 ? std::stod(argv[8]) : 0;
        std::string mode = argv[7];
        if (mode != "none" && mode != "pick" && mode != "color" && mode != "depth" &&
            mode != "segmentation")
            throw std::invalid_argument("Unknown readback mode");
        if (seconds <= 0 || seconds > 600 || fps_limit < 0)
            throw std::invalid_argument("Invalid benchmark duration/rate");
        Product product = mode == "color"          ? Product::Color
                          : mode == "depth"        ? Product::MetricDepth
                          : mode == "segmentation" ? Product::Segmentation
                                                   : Product::ObjectId;
        BgfxOptions options;
        options.shader_directory = argv[1];
        auto renderer = make_bgfx_renderer(options);
        renderer->set_scene(scene.source);
        auto target = renderer->create_target(size, 4);
        auto camera = scene.camera;
        camera.projection[0] *= (16.0f / 9) / (float(size.width) / size.height);
        std::deque<Pending> pending;
        std::vector<double> cpu, gpu, latency;
        uint64_t frames = 0, completed = 0, upload = 0, draws = 0;
        auto consume = [&](bool record) {
            while (!pending.empty()) {
                auto result = renderer->poll(pending.front().ticket);
                if (result.state == ReadbackState::Pending)
                    break;
                if (result.state != ReadbackState::Ready)
                    throw std::runtime_error("Unexpected canceled benchmark output");
                if (record) {
                    latency.push_back(ms(pending.front().start, Clock::now()));
                    ++completed;
                }
                pending.pop_front();
            }
        };
        // Warm the same readback path used in the measured run.
        for (size_t n = 0; n < 90; ++n) {
            renderer->update({1, n, scene.frames[n % scene.frames.size()]});
            auto token = renderer->render(target, camera);
            if (mode != "none")
                pending.push_back(
                    {renderer->readback(
                         token, product,
                         mode == "pick" ? Region{size.width / 2, size.height / 2, 1, 1} : Region{}),
                     Clock::now()});
            renderer->advance();
            consume(false);
            while (pending.size() >= 3) {
                renderer->advance();
                consume(false);
            }
        }
        while (!pending.empty()) {
            renderer->advance();
            consume(false);
        }
        auto start = Clock::now();
        while (ms(start, Clock::now()) < seconds * 1000) {
            auto begin = Clock::now();
            renderer->update({1, frames, scene.frames[frames % scene.frames.size()]});
            auto token = renderer->render(target, camera);
            if (mode != "none")
                pending.push_back(
                    {renderer->readback(
                         token, product,
                         mode == "pick" ? Region{size.width / 2, size.height / 2, 1, 1} : Region{}),
                     Clock::now()});
            auto stats = renderer->advance();
            consume(true);
            while (pending.size() >= 3) {
                renderer->advance();
                consume(true);
            }
            cpu.push_back(ms(begin, Clock::now()));
            if (stats.gpu_ms >= 0)
                gpu.push_back(stats.gpu_ms);
            upload += stats.upload_bytes;
            draws += stats.draw_calls;
            ++frames;
            if (fps_limit > 0)
                std::this_thread::sleep_until(
                    start + std::chrono::duration_cast<Clock::duration>(
                                std::chrono::duration<double>(frames / fps_limit)));
        }
        while (!pending.empty()) {
            renderer->advance();
            consume(true);
        }
        // A final readback drains queued GPU work even in the no-readback case.
        renderer->update({1, frames, scene.frames[frames % scene.frames.size()]});
        auto token = renderer->render(target, camera);
        auto ticket = renderer->readback(token, Product::Color);
        ReadbackResult snapshot;
        snapshot = wait_for_readback(*renderer, ticket);
        auto elapsed = ms(start, Clock::now()) / 1000;
        std::filesystem::path output = argv[3];
        std::filesystem::create_directories(output.parent_path());
        std::ofstream image(output.string() + ".ppm", std::ios::binary);
        image << "P6\n" << size.width << " " << size.height << "\n255\n";
        image.write(reinterpret_cast<const char *>(snapshot.image.pixels.data()),
                    snapshot.image.pixels.size());
        std::ofstream samples(output.string() + ".csv");
        samples << "sample,cpu_ms,gpu_ms,readback_ms\n";
        for (size_t i = 0; i < std::max(cpu.size(), latency.size()); ++i) {
            samples << i << ",";
            if (i < cpu.size())
                samples << cpu[i];
            samples << ",";
            if (i < gpu.size())
                samples << gpu[i];
            samples << ",";
            if (i < latency.size())
                samples << latency[i];
            samples << "\n";
        }
        std::ofstream report(output);
        report << std::fixed << std::setprecision(4) << "{\n\"backend\":\""
               << renderer->capabilities().backend << "\",\n\"mode\":\"" << mode
               << "\",\n\"width\":" << size.width << ",\n\"height\":" << size.height
               << ",\n\"instances\":" << scene.source.instances.size()
               << ",\n\"samples\":4,\n\"frames\":" << frames << ",\n\"seconds\":" << elapsed
               << ",\n\"fps_limit\":" << fps_limit << ",\n\"fps\":" << frames / elapsed
               << ",\n\"frame_cpu_p50_ms\":" << percentile(cpu, .5)
               << ",\n\"frame_cpu_p95_ms\":" << percentile(cpu, .95)
               << ",\n\"frame_cpu_p99_ms\":" << percentile(cpu, .99)
               << ",\n\"gpu_p50_ms\":" << percentile(gpu, .5)
               << ",\n\"gpu_p95_ms\":" << percentile(gpu, .95)
               << ",\n\"readback_p50_ms\":" << percentile(latency, .5)
               << ",\n\"readback_p95_ms\":" << percentile(latency, .95)
               << ",\n\"readbacks\":" << completed
               << ",\n\"upload_bytes_per_frame\":" << double(upload) / frames
               << ",\n\"draw_calls_per_frame\":" << double(draws) / frames << "\n}\n";
        std::cout << mode << " " << size.width << "x" << size.height << ": " << frames / elapsed
                  << " FPS, CPU P95 " << percentile(cpu, .95) << " ms, readback P95 "
                  << percentile(latency, .95) << " ms" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
