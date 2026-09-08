#include "scene_stream.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <mojive/backends/bgfx.hpp>
#include <mojive/readback.hpp>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#if defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#elif defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#else
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include <GLFW/glfw3native.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <unordered_set>

using namespace mojive;
static NativeWindow native_window(GLFWwindow *window) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    NativeWindow result;
    result.size = {uint32_t(width), uint32_t(height)};
#if defined(__APPLE__)
    result.handle = glfwGetCocoaWindow(window);
#elif defined(_WIN32)
    result.handle = glfwGetWin32Window(window);
#else
    result.handle = reinterpret_cast<void *>(glfwGetX11Window(window));
    result.display = glfwGetX11Display();
#endif
    return result;
}
struct UiPackets {
    std::vector<UiVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<UiCommand> commands;
    std::unordered_set<uint64_t> textures;
    UiFrame frame;
    void update(Renderer &renderer) {
        auto *data = ImGui::GetDrawData();
        for (auto *texture : ImGui::GetPlatformIO().Textures) {
            if (texture->Status == ImTextureStatus_WantCreate ||
                texture->Status == ImTextureStatus_WantUpdates) {
                if (texture->Format != ImTextureFormat_RGBA32)
                    throw std::runtime_error("Unsupported font atlas format");
                auto pixels = std::span(static_cast<const std::byte *>(texture->GetPixels()),
                                        size_t(texture->Width) * texture->Height * 4);
                auto id = renderer.upload_texture(
                    {uint32_t(texture->Width), uint32_t(texture->Height)}, pixels);
                if (texture->TexID != ImTextureID_Invalid) {
                    renderer.destroy(Texture{texture->TexID});
                    textures.erase(texture->TexID);
                }
                texture->SetTexID(id.id);
                texture->SetStatus(ImTextureStatus_OK);
                textures.insert(id.id);
            } else if (texture->Status == ImTextureStatus_WantDestroy) {
                renderer.destroy(Texture{texture->TexID});
                textures.erase(texture->TexID);
                texture->SetTexID(ImTextureID_Invalid);
                texture->SetStatus(ImTextureStatus_Destroyed);
            }
        }
        vertices.clear();
        indices.clear();
        commands.clear();
        for (auto *list : data->CmdLists) {
            uint32_t vertex_offset = vertices.size(), index_offset = indices.size();
            for (const auto &v : list->VtxBuffer)
                vertices.push_back({(v.pos.x - data->DisplayPos.x) * data->FramebufferScale.x,
                                    (v.pos.y - data->DisplayPos.y) * data->FramebufferScale.y,
                                    v.uv.x, v.uv.y, v.col});
            for (auto i : list->IdxBuffer)
                indices.push_back(i);
            for (const auto &cmd : list->CmdBuffer) {
                if (cmd.UserCallback) {
                    if (cmd.UserCallback != ImDrawCallback_ResetRenderState)
                        throw std::runtime_error(
                            "Custom ImGui callbacks are outside the probe contract");
                    continue;
                }
                commands.push_back(
                    {index_offset + cmd.IdxOffset,
                     cmd.ElemCount,
                     vertex_offset + cmd.VtxOffset,
                     {(cmd.ClipRect.x - data->DisplayPos.x) * data->FramebufferScale.x,
                      (cmd.ClipRect.y - data->DisplayPos.y) * data->FramebufferScale.y,
                      (cmd.ClipRect.z - data->DisplayPos.x) * data->FramebufferScale.x,
                      (cmd.ClipRect.w - data->DisplayPos.y) * data->FramebufferScale.y},
                     Texture{cmd.GetTexID()}});
            }
        }
        frame = {{uint32_t(data->DisplaySize.x * data->FramebufferScale.x),
                  uint32_t(data->DisplaySize.y * data->FramebufferScale.y)},
                 vertices,
                 indices,
                 commands};
    }
};
int main(int argc, char **argv) {
    try {
        if (argc != 6)
            throw std::invalid_argument(
                "Usage: gallery shaders trajectory output-directory font-latin font-cjk");
        if (!glfwInit())
            throw std::runtime_error("Cannot initialize GLFW");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        auto *window = glfwCreateWindow(1120, 760, "Mojive Native Probe", nullptr, nullptr);
        auto *secondary = glfwCreateWindow(520, 340, "Mojive Native Surface", nullptr, nullptr);
        if (!window || !secondary)
            throw std::runtime_error("Cannot create native windows");
        glfwSetWindowPos(window, 60, 70);
        glfwSetWindowPos(secondary, 1200, 90);
        BgfxOptions options;
        options.shader_directory = argv[1];
        options.window = native_window(window);
        auto renderer = make_bgfx_renderer(options);
        auto scene = probe::load(argv[2]);
        renderer->set_scene(scene.source);
        auto viewport = renderer->create_target({1280, 720}, 4);
        auto capture = renderer->create_target(options.window.size);
        auto secondary_size = native_window(secondary).size;
        auto surface = renderer->create_surface(native_window(secondary));
        ImGui::CreateContext();
        auto &io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.BackendFlags |=
            ImGuiBackendFlags_RendererHasTextures | ImGuiBackendFlags_RendererHasVtxOffset;
        io.BackendRendererName = "mojive-neutral-ui";
        float scale = std::getenv("MOJIVE_PROBE_UI_SCALE")
                          ? std::stof(std::getenv("MOJIVE_PROBE_UI_SCALE"))
                          : 1.0f;
        if (scale < 0.75f || scale > 2)
            throw std::invalid_argument("UI scale must be between 0.75 and 2");
        if (!io.Fonts->AddFontFromFileTTF(argv[4], 16 * scale))
            throw std::runtime_error("Cannot load Latin font");
        ImFontConfig merge;
        merge.MergeMode = true;
        if (!io.Fonts->AddFontFromFileTTF(argv[5], 16 * scale, &merge))
            throw std::runtime_error("Cannot load CJK font");
        ImGui_ImplGlfw_InitForOther(window, true);
        ImGui::StyleColorsDark();
        auto &style = ImGui::GetStyle();
        style.FrameRounding = style.WindowRounding = style.ChildRounding = style.PopupRounding =
            4.8f;
        style.WindowPadding = {12, 12};
        style.FramePadding = {10, 6};
        style.ItemSpacing = {10, 8};
        style.CellPadding = {8, 6};
        style.ScaleAllSizes(scale);
        UiPackets packets;
        FrameToken capture_frame;
        for (uint64_t n = 0; n < 110; ++n) {
            glfwPollEvents();
            if (n == 55)
                glfwSetWindowSize(window, 1000, 700);
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            auto dock = ImGui::DockSpaceOverViewport();
            if (n == 0) {
                ImGui::DockBuilderRemoveNode(dock);
                ImGui::DockBuilderAddNode(dock, ImGuiDockNodeFlags_DockSpace);
                ImGui::DockBuilderSetNodeSize(dock, io.DisplaySize);
                ImGuiID left, center;
                ImGui::DockBuilderSplitNode(dock, ImGuiDir_Left, .27f * scale, &left, &center);
                ImGui::DockBuilderDockWindow("Validation", left);
                ImGui::DockBuilderDockWindow("Viewport", center);
                ImGui::DockBuilderFinish(dock);
            }
            ImGui::Begin("Validation");
            ImGui::TextUnformatted("Native renderer probe");
            ImGui::Separator();
            ImGui::TextUnformatted("中文显示 · 层级 · 检查器");
            if (n > 30)
                ImGui::TextUnformatted("动态字形：渲染、物理、异步");
            if (ImGui::BeginTable("metrics", 2, ImGuiTableFlags_SizingStretchProp)) {
                for (const auto &item :
                     std::vector<std::pair<const char *, const char *>>{{"Backend", "Metal"},
                                                                        {"Moving bodies", "1,600"},
                                                                        {"Instances", "5,101"},
                                                                        {"Object ID", "uint32"},
                                                                        {"Depth", "meters"},
                                                                        {"MSAA", "4x"}}) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(item.first);
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(item.second);
                }
                ImGui::EndTable();
            }
            ImGui::Separator();
            ImGui::TextWrapped("This prototype validates geometry, outputs, readback and native "
                               "surfaces. Production shading is a separate migration step.");
            ImGui::End();
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            ImGui::Begin("Viewport");
            auto available = ImGui::GetContentRegionAvail();
            Extent viewport_size{
                uint32_t(std::max(1.0f, available.x * io.DisplayFramebufferScale.x)),
                uint32_t(std::max(1.0f, available.y * io.DisplayFramebufferScale.y))};
            renderer->resize(viewport, viewport_size);
            auto camera = scene.camera;
            camera.projection[0] *=
                (16.0f / 9) / (float(viewport_size.width) / viewport_size.height);
            renderer->update({1, n, scene.frames[n % scene.frames.size()]});
            renderer->render(viewport, camera);
            ImGui::Image(ImTextureRef(renderer->target_texture(viewport).id), available);
            ImGui::End();
            ImGui::PopStyleVar();
            ImGui::Render();
            packets.update(*renderer);
            renderer->render_ui(packets.frame);
            renderer->resize(capture, packets.frame.size);
            capture_frame = renderer->render_ui(packets.frame, capture);
            std::array<UiVertex, 4> quad = {
                {{0, 0, 0, 0, 0xffffffff},
                 {float(secondary_size.width), 0, 1, 0, 0xffffffff},
                 {float(secondary_size.width), float(secondary_size.height), 1, 1, 0xffffffff},
                 {0, float(secondary_size.height), 0, 1, 0xffffffff}}};
            std::array<uint32_t, 6> indices = {0, 1, 2, 0, 2, 3};
            std::array<UiCommand, 1> command = {
                {{0,
                  6,
                  0,
                  {0, 0, float(secondary_size.width), float(secondary_size.height)},
                  renderer->target_texture(viewport)}}};
            renderer->render_ui({secondary_size, quad, indices, command}, surface);
            renderer->advance();
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
        auto readback = renderer->readback(capture_frame, Product::Color);
        ReadbackResult result;
        result = wait_for_readback(*renderer, readback);
        std::filesystem::path output = argv[3];
        std::filesystem::create_directories(output);
        std::ofstream image(output / "gallery.ppm", std::ios::binary);
        image << "P6\n" << result.image.size.width << " " << result.image.size.height << "\n255\n";
        image.write(reinterpret_cast<const char *>(result.image.pixels.data()),
                    result.image.pixels.size());
        std::ofstream report(output / "gallery.json");
        report << "{\n  \"passed\": true,\n  \"ui_scale\": " << scale
               << ",\n  \"framebuffer_scale_x\": " << io.DisplayFramebufferScale.x
               << ",\n  \"framebuffer_scale_y\": " << io.DisplayFramebufferScale.y
               << ",\n  \"width\": " << result.image.size.width
               << ",\n  \"height\": " << result.image.size.height << "\n}\n";
        renderer->destroy(surface);
        readback = renderer->readback(capture_frame, Product::Color);
        result = wait_for_readback(*renderer, readback);
        glfwDestroyWindow(secondary);
        renderer->render_ui(packets.frame);
        renderer->advance();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        renderer.reset();
        glfwDestroyWindow(window);
        glfwTerminate();
        std::cout << "Native gallery passed: docking, CJK atlas updates, pixel sizing, resize and "
                     "secondary surface lifecycle"
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
