#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

namespace mojive::bindingProbe {
// Shared C++ workloads; binding headers and Python objects stay in module files.
int ping(int value);
class OwnedState {
    std::vector<double> mValues;

  public:
    explicit OwnedState(std::span<const double> values);
    std::span<const double> values() const;
    double work(size_t repeats) const;
};
class FrameBatch {
    std::vector<float> mPacked;
    size_t mCount;
    std::thread::id mOwner = std::this_thread::get_id();

  public:
    explicit FrameBatch(size_t count);
    // Reuse two 80-byte instance records, matching the renderer's two passes.
    double pack(std::span<const float> matrices);
    size_t bytes() const;
};
} // namespace mojive::bindingProbe
