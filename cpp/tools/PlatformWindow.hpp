#pragma once
#include <mojive/Render.hpp>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#if defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#elif defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#else
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#include <GLFW/glfw3native.h>
namespace mojive::probe {
inline NativeWindow nativeWindow(GLFWwindow *window) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    NativeWindow result;
    result.size = {uint32_t(width), uint32_t(height)};
#if defined(__APPLE__)
    result.handle = glfwGetCocoaWindow(window);
#elif defined(_WIN32)
    result.handle = glfwGetWin32Window(window);
#else
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
        result.handle = glfwGetWaylandWindow(window);
        result.display = glfwGetWaylandDisplay();
        result.system = WindowSystem::Wayland;
    } else {
        result.handle = reinterpret_cast<void *>(glfwGetX11Window(window));
        result.display = glfwGetX11Display();
    }
#endif
    return result;
}
} // namespace mojive::probe
