#include "UiPass.hpp"
#include <algorithm>
#include <bgfx/defines.h>
#include <cmath>
#include <stdexcept>

namespace mojive {
std::array<uint16_t, 4> uiScissor(const UiCommand &command, Extent size) {
    const auto &clip = command.clip;
    const float left = std::clamp(clip[0], 0.f, float(size.width)),
                top = std::clamp(clip[1], 0.f, float(size.height)),
                right = std::clamp(clip[2], left, float(size.width)),
                bottom = std::clamp(clip[3], top, float(size.height));
    if (!command.indexCount || right <= left || bottom <= top)
        return {};
    return {uint16_t(std::floor(left)), uint16_t(std::floor(top)),
            uint16_t(std::ceil(right) - std::floor(left)),
            uint16_t(std::ceil(bottom) - std::floor(top))};
}
UiPass::UiPass(size_t maxBytes) : mMaxBytes(maxBytes) {
    mLayout.begin()
        .add(bgfx::Attrib::Position, 2, bgfx::AttribType::Float)
        .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
        .end();
}
void UiPass::prepare(const UiFrame &frame) {
    const auto vertices = std::max(mVertexCapacity, frame.vertices.size());
    const auto indices = std::max(mIndexCapacity, frame.indices.size());
    if (vertices > UINT32_MAX || indices > UINT32_MAX ||
        vertices * sizeof(UiVertex) + indices * sizeof(uint32_t) > mMaxBytes)
        throw std::length_error("UI vertex/index upload budget exceeded");
    auto grownVertices = vertices > mVertexCapacity
                             ? std::max({vertices, mVertexCapacity * 2, size_t(1024)})
                             : mVertexCapacity;
    auto grownIndices = indices > mIndexCapacity
                            ? std::max({indices, mIndexCapacity * 2, size_t(1024)})
                            : mIndexCapacity;
    if (grownVertices * sizeof(UiVertex) + grownIndices * sizeof(uint32_t) > mMaxBytes) {
        grownVertices = vertices;
        grownIndices = indices;
    }
    GpuHandle<bgfx::DynamicVertexBufferHandle> pendingVertices;
    GpuHandle<bgfx::DynamicIndexBufferHandle> pendingIndices;
    if (frame.vertices.size() > mVertexCapacity) {
        pendingVertices.reset(bgfx::createDynamicVertexBuffer(grownVertices, mLayout));
        if (!bgfx::isValid(pendingVertices.value))
            throw std::runtime_error("Cannot allocate UI vertex buffer");
    }
    if (frame.indices.size() > mIndexCapacity) {
        pendingIndices.reset(bgfx::createDynamicIndexBuffer(grownIndices, BGFX_BUFFER_INDEX32));
        if (!bgfx::isValid(pendingIndices.value))
            throw std::runtime_error("Cannot allocate UI index buffer");
    }
    if (bgfx::isValid(pendingVertices.value)) {
        mVertices = std::move(pendingVertices);
        mVertexCapacity = grownVertices;
    }
    if (bgfx::isValid(pendingIndices.value)) {
        mIndices = std::move(pendingIndices);
        mIndexCapacity = grownIndices;
    }
    if (!frame.vertices.empty())
        bgfx::update(mVertices.value, 0,
                     bgfx::copy(frame.vertices.data(), frame.vertices.size_bytes()));
    if (!frame.indices.empty())
        bgfx::update(mIndices.value, 0,
                     bgfx::copy(frame.indices.data(), frame.indices.size_bytes()));
    mUploadBytes = frame.vertices.size_bytes() + frame.indices.size_bytes();
}
uint64_t UiPass::submit(const UiFrame &frame, uint32_t firstCommand, uint32_t commandCount,
                        std::span<const UiTexture> textures, bgfx::ViewId view,
                        bgfx::FrameBufferHandle target, UiPipeline pipeline, bool clear) {
    const std::array<float, 16> projection{2.0f / frame.size.width,
                                           0,
                                           0,
                                           0,
                                           0,
                                           -2.0f / frame.size.height,
                                           0,
                                           0,
                                           0,
                                           0,
                                           1,
                                           0,
                                           -1,
                                           1,
                                           0,
                                           1};
    bgfx::setViewMode(view, bgfx::ViewMode::Sequential);
    bgfx::setViewRect(view, 0, 0, frame.size.width, frame.size.height);
    bgfx::setViewFrameBuffer(view, target);
    bgfx::setViewTransform(view, nullptr, projection.data());
    bgfx::setViewClear(view, clear ? BGFX_CLEAR_COLOR : BGFX_CLEAR_NONE, 0x14191eff);
    bgfx::touch(view);
    uint64_t draws = 0;
    for (uint32_t index = firstCommand; index < firstCommand + commandCount; ++index) {
        const auto &command = frame.commands[index];
        const auto clip = uiScissor(command, frame.size);
        if (!clip[2] || !clip[3])
            continue;
        bgfx::setScissor(clip[0], clip[1], clip[2], clip[3]);
        bgfx::setVertexBuffer(0, mVertices.value, command.vertexOffset,
                              frame.vertices.size() - command.vertexOffset);
        bgfx::setIndexBuffer(mIndices.value, command.firstIndex, command.indexCount);
        const float info[] = {textures[index].flipped ? 1.f : 0.f, 0, 0, 0};
        bgfx::setUniform(pipeline.textureInfo, info);
        bgfx::setTexture(0, pipeline.sampler, textures[index].handle);
        bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A |
                       BGFX_STATE_BLEND_FUNC_SEPARATE(
                           BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA,
                           BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_INV_SRC_ALPHA));
        bgfx::submit(view, pipeline.program);
        ++draws;
    }
    return draws;
}
} // namespace mojive
