#pragma once
#include <mojive/render.hpp>
namespace mojive {
// Only the composition root chooses this adapter. Consumers use Renderer.
struct BgfxOptions {
    NativeWindow window;
    std::string shader_directory;
};
std::unique_ptr<Renderer> make_bgfx_renderer(const BgfxOptions &);
} // namespace mojive
