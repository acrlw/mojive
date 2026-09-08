#pragma once
#include <mojive/render.hpp>
namespace mojive {
struct SdlOptions {
    NativeWindow window;
    std::string shader_directory;
};
std::unique_ptr<Renderer> make_sdl_renderer(const SdlOptions &);
} // namespace mojive
