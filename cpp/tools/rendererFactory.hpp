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
    std::string shaderDirectory;
};
inline std::unique_ptr<Renderer> makeRenderer(const RendererOptions &options,
                                              const std::string &backend) {
#ifdef MOJIVE_HAS_BGFX
    if (backend == "bgfx")
        return makeBgfxRenderer({options.window, options.shaderDirectory});
#endif
#ifdef MOJIVE_HAS_SDL
    if (backend == "sdl")
        return makeSdlRenderer({options.window, options.shaderDirectory});
#endif
    throw std::invalid_argument("Backend was not built: " + backend);
}
} // namespace mojive::probe
