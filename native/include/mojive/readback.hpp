#pragma once
#include <chrono>
#include <mojive/render.hpp>
namespace mojive {
// Offline/validation convenience. Interactive owners should poll between events.
ReadbackResult wait_for_readback(Renderer &, ReadbackTicket,
                                 std::chrono::milliseconds timeout = std::chrono::seconds(10));
} // namespace mojive
