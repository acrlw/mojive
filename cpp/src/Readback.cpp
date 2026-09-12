#include <cstring>
#include <mojive/Readback.hpp>
#include <stdexcept>
namespace mojive {
ReadbackState Renderer::readInto(FrameToken frame, ImageView destination, Region region) {
    auto result = waitForReadback(*this, readback(frame, destination.product, region));
    if (result.state == ReadbackState::Ready) {
        if (destination.size != result.image.size ||
            destination.pixels.size() != result.image.pixels.size())
            throw std::invalid_argument("Readback destination has the wrong size");
        std::memcpy(destination.pixels.data(), result.image.pixels.data(),
                    destination.pixels.size());
    }
    return result.state;
}
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
