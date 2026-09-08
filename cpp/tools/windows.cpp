#include "platformWindow.hpp"
#include "rendererFactory.hpp"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mojive/readback.hpp>
#include <thread>
#ifdef __APPLE__
#include <mach/mach.h>
#endif
using namespace mojive;
static uint64_t footprint() {
#ifdef __APPLE__
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) ==
        KERN_SUCCESS)
        return info.phys_footprint;
#endif
    return 0;
}
static FrameToken present(Renderer &r, Target target, Extent size, Texture tex) {
    float w = size.width, h = size.height;
    std::array<UiVertex, 4> vertices{{{0, 0, 0, 0, 0xffffffff},
                                      {w, 0, 1, 0, 0xffffffff},
                                      {w, h, 1, 1, 0xffffffff},
                                      {0, h, 0, 1, 0xffffffff}}};
    std::array<uint32_t, 6> indices{0, 1, 2, 0, 2, 3};
    std::array<UiCommand, 1> commands{{{0, 6, 0, {0, 0, w, h}, tex}}};
    return r.renderUi({size, vertices, indices, commands}, target);
}
int main(int argc, char **argv) {
    try {
        if (argc != 4 && argc != 5)
            throw std::invalid_argument("Usage: windows shaders output backend [cycles]");
        if (!glfwInit())
            throw std::runtime_error("Cannot initialize GLFW");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        auto *window =
            glfwCreateWindow(640, 400, "Mojive window lifecycle validation", nullptr, nullptr);
        if (!window)
            throw std::runtime_error("Cannot create primary window");
        auto renderer = probe::makeRenderer({probe::nativeWindow(window), argv[1]}, argv[3]);
        std::array<std::byte, 16> pixels{
            std::byte(255), std::byte(0),   std::byte(0),   std::byte(255),
            std::byte(0),   std::byte(255), std::byte(0),   std::byte(255),
            std::byte(0),   std::byte(0),   std::byte(255), std::byte(255),
            std::byte(255), std::byte(255), std::byte(255), std::byte(255)};
        auto texture = renderer->uploadTexture({2, 2}, pixels);
        auto capture = renderer->createTarget({320, 240}, 1);
        const int cycles = argc == 5 ? std::stoi(argv[4]) : 24;
        if (cycles < 1 || cycles > 240)
            throw std::invalid_argument("Invalid lifecycle cycle count");
        std::vector<uint64_t> memory;
        int resizes = 0, minimized = 0;
        for (int cycle = 0; cycle < cycles; ++cycle) {
            auto *peer =
                glfwCreateWindow(400, 260, "Mojive peer lifecycle validation", nullptr, nullptr);
            if (!peer)
                throw std::runtime_error("Cannot create peer window");
            auto surface = renderer->createSurface(probe::nativeWindow(peer));
            for (int frame = 0; frame < 18; ++frame) {
                if (frame == 5) {
                    glfwSetWindowSize(peer, 360 + cycle % 4 * 37, 230 + cycle % 3 * 29);
                    glfwPollEvents();
                    renderer->resize(surface, probe::nativeWindow(peer).size);
                    ++resizes;
                }
                if (frame == 9) {
                    glfwIconifyWindow(peer);
                    ++minimized;
                }
                if (frame == 12)
                    glfwRestoreWindow(peer);
                if (frame == 14)
                    glfwSetWindowSize(window, 640 + cycle % 3 * 21, 400 + cycle % 2 * 31);
                glfwPollEvents();
                present(*renderer, {}, probe::nativeWindow(window).size, texture);
                // A minimized or hidden platform window has no presentation demand.
                if (!glfwGetWindowAttrib(peer, GLFW_ICONIFIED))
                    present(*renderer, surface, probe::nativeWindow(peer).size, texture);
                renderer->advance();
                std::this_thread::sleep_for(std::chrono::milliseconds(8));
            }
            renderer->destroy(surface);
            glfwDestroyWindow(peer);
            auto token = present(*renderer, capture, {320, 240}, texture);
            auto image = waitForReadback(*renderer, renderer->readback(token, Product::Color));
            if (image.state != ReadbackState::Ready || int(image.image.pixels[0]) < 200)
                throw std::runtime_error("Surviving window device stopped producing valid output");
            present(*renderer, {}, probe::nativeWindow(window).size, texture);
            renderer->advance();
            memory.push_back(footprint());
        }
        auto output = std::filesystem::path(argv[2]);
        std::filesystem::create_directories(output);
        std::ofstream report(output / "windows.json");
        report << "{\"passed\":true,\"peer_reopens\":" << cycles
               << ",\"peer_surface_resizes\":" << resizes
               << ",\"minimize_restore_cycles\":" << minimized << ",\"physical_footprint_bytes\":[";
        for (size_t i = 0; i < memory.size(); ++i)
            report << (i ? "," : "") << memory[i];
        report << "]";
        renderer->destroy(capture);
        renderer->destroy(texture);
        renderer.reset();
        glfwDestroyWindow(window);
        glfwTerminate();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        report << ",\"physical_footprint_after_shutdown_bytes\":" << footprint() << "}\n";
        std::cout << argv[3] << " window lifecycle passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
