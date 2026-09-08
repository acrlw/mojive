#pragma once
#include <mojive/render.hpp>
#include <stdexcept>
#ifdef MOJIVE_HAS_BGFX
#include <mojive/backends/bgfx.hpp>
#endif
#ifdef MOJIVE_HAS_SDL
#include <mojive/backends/sdl.hpp>
#endif
namespace mojive::probe {
struct RendererOptions {
    NativeWindow window;
    std::string shader_directory;
};
inline std::unique_ptr<Renderer> make_renderer(const RendererOptions &options,
                                               const std::string &backend) {
#ifdef MOJIVE_HAS_BGFX
    if (backend == "bgfx")
        return make_bgfx_renderer({options.window, options.shader_directory});
#endif
#ifdef MOJIVE_HAS_SDL
    if (backend == "sdl")
        return make_sdl_renderer({options.window, options.shader_directory});
#endif
    throw std::invalid_argument("Backend was not built: " + backend);
}
} // namespace mojive::probe
