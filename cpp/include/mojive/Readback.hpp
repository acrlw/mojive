#pragma once
#include <chrono>
#include <mojive/Render.hpp>
namespace mojive {
// Offline/validation convenience. Interactive owners should poll between events.
ReadbackResult waitForReadback(Renderer &, ReadbackTicket,
                               std::chrono::milliseconds timeout = std::chrono::seconds(10));
} // namespace mojive
