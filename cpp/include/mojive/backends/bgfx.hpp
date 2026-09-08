#pragma once
#include <mojive/render.hpp>
namespace mojive {
// Only the composition root chooses this adapter. Consumers use Renderer.
struct BgfxOptions {
    NativeWindow window;
    std::string shaderDirectory;
};
std::unique_ptr<Renderer> makeBgfxRenderer(const BgfxOptions &);
} // namespace mojive
