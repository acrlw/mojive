#pragma once
#include "imgui.h"

// Inspection for the controlled ImGui backends; arbitrary callbacks remain unsupported.
namespace mojive::imguiBridge {
enum class CallbackKind { None = 0, ResetRenderState = 1, Unsupported = 3 };
inline CallbackKind callbackKind(const ImDrawCmd &command) {
    if (!command.UserCallback)
        return CallbackKind::None;
    if (command.UserCallback == reinterpret_cast<ImDrawCallback>(-8) ||
        (ImGui::GetCurrentContext() &&
         command.UserCallback == ImGui::GetPlatformIO().DrawCallback_ResetRenderState))
        return CallbackKind::ResetRenderState;
    return CallbackKind::Unsupported;
}
} // namespace mojive::imguiBridge
