#include "workload.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace mojive::bindingProbe {
int ping(int value) {
    if (value == std::numeric_limits<int>::max())
        throw std::overflow_error("Integer overflow");
    return value + 1;
}
OwnedState::OwnedState(std::span<const double> values) : mValues(values.begin(), values.end()) {}
std::span<const double> OwnedState::values() const {
    return mValues;
}
double OwnedState::work(size_t repeats) const {
    double sum = 0;
    for (size_t n = 0; n < repeats; ++n)
        for (double value : mValues)
            sum += std::sin(value + n * 0.000001);
    return sum;
}
static size_t storageSize(size_t count) {
    if (count > 100000)
        throw std::invalid_argument("FrameBatch exceeds the probe instance limit");
    return count * 40;
}
FrameBatch::FrameBatch(size_t count) : mPacked(storageSize(count)), mCount(count) {
    for (size_t i = 0; i < count; ++i) {
        auto *out = mPacked.data() + i * 40;
        out[12] = .2f;
        out[13] = .4f;
        out[14] = .8f;
        out[15] = 1;
        out[32] = float((i + 1) & 65535);
        out[33] = float((i + 1) >> 16);
    }
}
double FrameBatch::pack(std::span<const float> matrices) {
    if (std::this_thread::get_id() != mOwner)
        throw std::logic_error("FrameBatch requires its owner thread");
    if (matrices.size() != mCount * 16)
        throw std::invalid_argument("Expected N contiguous 4x4 matrices");
    double sum = 0;
    for (size_t i = 0; i < mCount; ++i) {
        auto *src = matrices.data() + i * 16;
        auto *dst = mPacked.data() + i * 40;
        std::copy_n(src, 12, dst);
        std::copy_n(src, 12, dst + 20);
        sum += src[3] + src[7] + src[11];
    }
    return sum;
}
size_t FrameBatch::bytes() const {
    return mPacked.size() * sizeof(float);
}
} // namespace mojive::bindingProbe
