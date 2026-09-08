// Isolate SDL's native window/device lifecycle from Mojive, GLFW and ImGui.
#include <SDL3/SDL.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifdef __APPLE__
#include <mach/mach.h>
#endif
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
template <class T> static T checked(T value) {
    if (!value)
        throw std::runtime_error(SDL_GetError());
    return value;
}
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("Usage: sdl_window_audit output.json");
        checked(SDL_Init(SDL_INIT_VIDEO));
#ifdef __APPLE__
        auto *device = checked(SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_MSL, false, "metal"));
#else
        auto *device = checked(SDL_CreateGPUDevice(
            SDL_GPU_SHADERFORMAT_SPIRV | SDL_GPU_SHADERFORMAT_DXIL, false, nullptr));
#endif
        std::array<uint64_t, 65> memory{};
        for (size_t n = 0; n < 64; ++n) {
            auto *window = checked(SDL_CreateWindow("SDL lifecycle isolation", 400, 260, 0));
            checked(SDL_ClaimWindowForGPUDevice(device, window));
            auto *cmd = checked(SDL_AcquireGPUCommandBuffer(device));
            SDL_GPUTexture *texture = nullptr;
            checked(SDL_WaitAndAcquireGPUSwapchainTexture(cmd, window, &texture, nullptr, nullptr));
            if (texture) {
                SDL_GPUColorTargetInfo color{};
                color.texture = texture;
                color.clear_color = {.2f, .4f, .6f, 1};
                color.load_op = SDL_GPU_LOADOP_CLEAR;
                color.store_op = SDL_GPU_STOREOP_STORE;
                SDL_EndGPURenderPass(checked(SDL_BeginGPURenderPass(cmd, &color, 1, nullptr)));
            }
            checked(SDL_SubmitGPUCommandBuffer(cmd));
            checked(SDL_WaitForGPUIdle(device));
            SDL_ReleaseWindowFromGPUDevice(device, window);
            SDL_DestroyWindow(window);
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
            }
            SDL_Delay(20);
            memory[n] = footprint();
        }
        SDL_DestroyGPUDevice(device);
        SDL_Quit();
        SDL_Delay(1000);
        memory.back() = footprint();
        std::filesystem::path output = argv[1];
        std::filesystem::create_directories(output.parent_path());
        std::ofstream report(output);
        report << "{\"cycles\":64,\"physical_footprint_bytes\":[";
        for (size_t i = 0; i < memory.size(); ++i)
            report << (i ? "," : "") << memory[i];
        report << "],\"last_sample_after_device_shutdown\":true}\n";
        std::cout << "SDL lifecycle isolation completed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
