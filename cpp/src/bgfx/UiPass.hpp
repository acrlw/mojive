#pragma once

#include "GpuHandle.hpp"
#include <mojive/Render.hpp>
#include <span>

namespace mojive {

struct UiPipeline {
    bgfx::ProgramHandle program;
    bgfx::UniformHandle sampler, textureInfo;
};
struct UiTexture {
    bgfx::TextureHandle handle = BGFX_INVALID_HANDLE;
    bool flipped = false;
};
std::array<uint16_t, 4> uiScissor(const UiCommand &, Extent);

// One bounded vertex/index upload serves all command spans in a composition.
// Shader reload and texture ownership stay with the renderer that supplies them.
class UiPass {
  public:
    explicit UiPass(size_t maxBytes = 64 * 1024 * 1024);
    void prepare(const UiFrame &);
    uint64_t submit(const UiFrame &, uint32_t firstCommand, uint32_t commandCount,
                    std::span<const UiTexture>, bgfx::ViewId, bgfx::FrameBufferHandle, UiPipeline,
                    bool clear = false);
    size_t uploadBytes() const {
        return mUploadBytes;
    }

  private:
    size_t mMaxBytes, mVertexCapacity = 0, mIndexCapacity = 0, mUploadBytes = 0;
    bgfx::VertexLayout mLayout;
    GpuHandle<bgfx::DynamicVertexBufferHandle> mVertices;
    GpuHandle<bgfx::DynamicIndexBufferHandle> mIndices;
};

} // namespace mojive
