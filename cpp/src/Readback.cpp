#include <mojive/Readback.hpp>
#include <stdexcept>
namespace mojive {
ReadbackResult waitForReadback(Renderer &renderer, ReadbackTicket ticket,
                               std::chrono::milliseconds timeout) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        renderer.advance();
        auto result = renderer.poll(ticket);
        if (result.state != ReadbackState::Pending)
            return result;
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("Readback timed out");
    }
}
} // namespace mojive
