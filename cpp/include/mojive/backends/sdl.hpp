#pragma once
#include <mojive/render.hpp>
namespace mojive {
struct SdlOptions {
    NativeWindow window;
    std::string shaderDirectory;
};
std::unique_ptr<Renderer> makeSdlRenderer(const SdlOptions &);
} // namespace mojive
