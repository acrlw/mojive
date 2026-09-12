#pragma once

#include <algorithm>
#include <bgfx/bgfx.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mojive {

// Frame-owned instance data, grouped by vertex layout. Writers fill the upload
// range directly; no intermediate per-draw buffer or second CPU copy is needed.
class InstanceStream {
    struct Upload {
        bgfx::DynamicVertexBufferHandle handle = BGFX_INVALID_HANDLE;
        bgfx::VertexLayout layout;
        uint32_t cursor = 0, capacity = 0;
        std::vector<std::byte> bytes;
    };
    std::unordered_map<uint16_t, Upload> mUploads;

    static void upload(const Upload &value) {
        if (value.cursor)
            bgfx::update(value.handle, 0,
                         bgfx::copy(value.bytes.data(), value.cursor * value.layout.getStride()));
    }

  public:
    InstanceStream() = default;
    InstanceStream(const InstanceStream &) = delete;
    InstanceStream &operator=(const InstanceStream &) = delete;
    ~InstanceStream() {
        clear();
    }

    // The returned range remains writable until the next allocation. Binding is
    // recorded now; the caller must submit its draw before allocating again.
    std::span<std::byte> allocate(uint32_t count, uint16_t stride) {
        if (!stride || stride % 16 || uint64_t(count) * stride > UINT32_MAX)
            throw std::length_error("Invalid instance stream extent");
        auto &value = mUploads[stride];
        if (!bgfx::isValid(value.handle)) {
            value.layout.begin();
            for (int i = 0; i < stride / 16; ++i)
                value.layout.add(bgfx::Attrib::Enum(bgfx::Attrib::TexCoord0 + i), 4,
                                 bgfx::AttribType::Float);
            value.layout.end();
        }
        // bgfx's automatic resize checks payload size, not startVertex + count.
        // Deferred destruction preserves earlier draws when this stream grows.
        if (count > value.capacity - value.cursor) {
            uint32_t capacity = std::min<uint64_t>(
                UINT32_MAX / stride, std::max<uint64_t>({count, 2ull * value.capacity, 1024}));
            auto handle = bgfx::createDynamicVertexBuffer(capacity, value.layout);
            if (!bgfx::isValid(handle))
                throw std::runtime_error("Cannot allocate instance stream");
            if (bgfx::isValid(value.handle)) {
                upload(value);
                bgfx::destroy(value.handle);
            }
            value.handle = handle;
            value.capacity = capacity;
            value.cursor = 0;
        }
        size_t offset = size_t(value.cursor) * stride;
        size_t length = size_t(count) * stride;
        value.bytes.resize(offset + length);
        bgfx::setInstanceDataBuffer(value.handle, value.cursor, count);
        value.cursor += count;
        return std::span(value.bytes).subspan(offset, length);
    }

    void flush() {
        // One upload per layout avoids a Metal staging buffer for every draw.
        for (const auto &[stride, value] : mUploads)
            upload(value);
    }

    void reset() {
        for (auto &[stride, value] : mUploads)
            value.cursor = 0;
    }

    // The renderer calls this before shutting down the bgfx device.
    void clear() {
        for (auto &[stride, value] : mUploads)
            if (bgfx::isValid(value.handle))
                bgfx::destroy(value.handle);
        mUploads.clear();
    }
};
} // namespace mojive
