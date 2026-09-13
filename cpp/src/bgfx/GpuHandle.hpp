#pragma once

#include <bgfx/bgfx.h>
#include <utility>

namespace mojive {
// Backend-owned handles must be reset before the runtime calls bgfx::shutdown().
template <class T> class GpuHandle {
  public:
    T value = BGFX_INVALID_HANDLE;
    GpuHandle() = default;
    explicit GpuHandle(T handle) : value(handle) {}
    ~GpuHandle() {
        reset();
    }
    GpuHandle(const GpuHandle &) = delete;
    GpuHandle &operator=(const GpuHandle &) = delete;
    GpuHandle(GpuHandle &&other) noexcept {
        std::swap(value, other.value);
    }
    GpuHandle &operator=(GpuHandle &&other) noexcept {
        std::swap(value, other.value);
        return *this;
    }
    void reset(T handle = BGFX_INVALID_HANDLE) {
        if (bgfx::isValid(value))
            bgfx::destroy(value);
        value = handle;
    }
};
} // namespace mojive
