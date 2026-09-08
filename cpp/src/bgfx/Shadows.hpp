#pragma once
#include <algorithm>
#include <array>
#include <bgfx/bgfx.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <mojive/Render.hpp>
#include <stdexcept>

namespace mojive {
inline glm::vec3 vector3(const std::array<float, 3> &v) {
    return {v[0], v[1], v[2]};
}
inline glm::mat4 safeLookAt(glm::vec3 eye, glm::vec3 target, glm::vec3 up) {
    auto f = target - eye;
    f = glm::length(f) > 1e-12f ? glm::normalize(f) : glm::vec3(0, 0, -1);
    if (glm::length(glm::cross(f, up)) < 1e-9f)
        up = std::abs(f.x) < .9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    return glm::lookAtRH(eye, eye + f, up);
}
struct ShadowMaps {
    bgfx::TextureHandle atlas = BGFX_INVALID_HANDLE, local = BGFX_INVALID_HANDLE;
    bgfx::FrameBufferHandle atlasFrame = BGFX_INVALID_HANDLE;
    std::vector<bgfx::FrameBufferHandle> localFrames;
    uint32_t pixels = 0, layers = 0;
    int directionalLight = -1, localCount = 0;
    std::array<glm::mat4, 3> matrices{};
    std::array<float, 4> splits{}, texels{};
    std::array<glm::mat4, 8> localMatrices{};
    std::array<std::array<glm::mat4, 6>, 8> pointMatrices{};
    std::array<std::array<float, 4>, 8> positions{}, localParams{};
    std::array<std::array<float, 4>, 100> slots{};
    std::array<int, 8> kinds{};
    void releaseLocal() {
        for (auto f : localFrames)
            bgfx::destroy(f);
        localFrames.clear();
        if (bgfx::isValid(local))
            bgfx::destroy(local);
        local = BGFX_INVALID_HANDLE;
        pixels = layers = 0;
    }
    void release() {
        releaseLocal();
        if (bgfx::isValid(atlasFrame))
            bgfx::destroy(atlasFrame);
        atlas = BGFX_INVALID_HANDLE;
        atlasFrame = BGFX_INVALID_HANDLE;
    }
    void prepare(const SceneSource &source, const Lighting &light, const SceneStyle &style,
                 const CameraView &camera) {
        directionalLight = -1;
        localCount = 0;
        for (auto &slot : slots)
            slot = {-1, 0, 0, 0};
        if (!style.shadows || !light.enabled)
            return;
        std::vector<int> selected;
        bool onlySpots = true;
        for (size_t i = 0; i < light.lights.size(); ++i) {
            const auto &l = light.lights[i];
            if (!l.castShadow)
                continue;
            if (l.type == 0 && directionalLight < 0)
                directionalLight = i;
            if (l.type > 0 && l.type <= 3 && selected.size() < 8) {
                selected.push_back(i);
                onlySpots &= l.type == 2;
            }
        }
        if (directionalLight >= 0) {
            if (!bgfx::isValid(atlas)) {
                atlas = bgfx::createTexture2D(4096, 4096, false, 1, bgfx::TextureFormat::D32F,
                                              BGFX_TEXTURE_RT | BGFX_SAMPLER_COMPARE_LEQUAL |
                                                  BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP);
                if (!bgfx::isValid(atlas))
                    throw std::runtime_error("Cannot allocate directional shadow atlas");
                atlasFrame = bgfx::createFrameBuffer(1, &atlas, true);
                if (!bgfx::isValid(atlasFrame)) {
                    bgfx::destroy(atlas);
                    atlas = BGFX_INVALID_HANDLE;
                    throw std::runtime_error("Cannot allocate shadow framebuffer");
                }
            }
            auto direction = vector3(light.lights[directionalLight].direction);
            direction =
                glm::length(direction) > 1e-6f ? glm::normalize(direction) : glm::vec3(0, 0, -1);
            auto up = std::abs(direction.z) < .99f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
            auto side = glm::normalize(glm::cross(direction, up));
            up = glm::cross(side, direction);
            glm::mat3 basis = glm::transpose(glm::mat3(side, up, -direction));
            const float divisors[3][3] = {{6, 2, 1}, {9, 3, 1}, {12, 4, 1}};
            for (size_t i = 0; i < 3; ++i) {
                float radius = std::max(source.extent, 1e-6f) * std::max(source.shadowClip, 1e-6f) /
                               divisors[std::clamp(style.shadowQuality, 0, 2)][i];
                float texel = 2 * radius / 2048;
                auto center = glm::transpose(basis) *
                              (glm::floor((basis * vector3(camera.focus)) / texel) * texel);
                float half = std::max(
                    source.extent + glm::length(center - vector3(source.center)) + radius, 1e-4f);
                matrices[i] = glm::orthoRH_NO(-radius, radius, -radius, radius, 0.f, 2 * half) *
                              safeLookAt(center - direction * half, center, up);
                splits[i] = radius;
                texels[i] = texel;
            }
        }
        uint32_t nextPixels = onlySpots ? 2048 : 1024, nextLayers = 0;
        for (int index : selected)
            nextLayers += light.lights[index].type == 2 ? 1 : 6;
        if (nextLayers && (nextPixels != pixels || nextLayers != layers)) {
            releaseLocal();
            pixels = nextPixels;
            layers = nextLayers;
            local = bgfx::createTexture2D(
                pixels, pixels, false, std::max(layers, 2u), bgfx::TextureFormat::R16F,
                BGFX_TEXTURE_RT | BGFX_SAMPLER_MIN_POINT | BGFX_SAMPLER_MAG_POINT |
                    BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP);
            if (!bgfx::isValid(local))
                throw std::runtime_error("Cannot allocate local shadow array");
            for (uint32_t layer = 0; layer < layers; ++layer) {
                bgfx::Attachment attachment;
                attachment.init(local, bgfx::Access::Write, layer);
                auto frame = bgfx::createFrameBuffer(1, &attachment, false);
                if (!bgfx::isValid(frame)) {
                    releaseLocal();
                    throw std::runtime_error("Cannot allocate local shadow framebuffer");
                }
                localFrames.push_back(frame);
            }
        }
        int layer = 0;
        for (int index : selected) {
            const auto &l = light.lights[index];
            int slot = localCount++;
            slots[index][0] = float(slot);
            kinds[slot] = l.type;
            float range = source.linearColors ? l.range : 0;
            positions[slot] = {l.position[0], l.position[1], l.position[2], range};
            float near = std::max(source.extent * .02f, 1e-3f);
            auto pos = vector3(l.position);
            if (l.type == 2) {
                float fov = 2 * std::clamp(l.cutoff, 1.f, 89.f) * .017453292519943295f;
                float far = range > near ? range : source.extent * 6;
                localMatrices[slot] =
                    glm::perspectiveRH_NO(fov, 1.f, near, far) *
                    safeLookAt(pos, pos + vector3(l.direction) * source.extent, {0, 0, 1});
                localParams[slot] = {2 * std::tan(fov * .5f) / pixels, l.radius, float(layer++),
                                     float(pixels)};
            } else {
                float far =
                    range > near
                        ? range
                        : std::max(glm::length(pos - vector3(source.center)) + 2 * source.extent,
                                   near * 2);
                auto projection = glm::perspectiveRH_NO(1.5707963267948966f, 1.f, near, far);
                const glm::vec3 direction[6] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                                {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
                const glm::vec3 up[6] = {{0, -1, 0}, {0, -1, 0}, {0, 0, 1},
                                         {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};
                for (int face = 0; face < 6; ++face)
                    pointMatrices[slot][face] =
                        projection * safeLookAt(pos, pos + direction[face], up[face]);
                localParams[slot] = {2.f / pixels, l.radius, float(layer), float(pixels)};
                layer += 6;
            }
        }
    }
};
class ShadowUniforms {
    enum Slot {
        Config,
        Matrix,
        Splits,
        Texels,
        LocalMatrix,
        LocalPosition,
        LocalParams,
        LocalSlots,
        Count
    };
    std::array<bgfx::UniformHandle, Count> mHandles;
    bgfx::UniformHandle mAtlasSampler = BGFX_INVALID_HANDLE, mLocalSampler = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle mAtlasFallback = BGFX_INVALID_HANDLE, mLocalFallback = BGFX_INVALID_HANDLE;

  public:
    ShadowUniforms() {
        mHandles.fill(bgfx::UniformHandle{bgfx::kInvalidHandle});
    }
    void initialize() {
        const char *names[] = {"u_shadowConfig", "u_shadowMatrix", "u_shadowSplits",
                               "u_shadowTexels", "u_localMatrix",  "u_localPosition",
                               "u_localParams",  "u_localSlots"};
        for (int i = 0; i < Count; ++i)
            mHandles[i] = bgfx::createUniform(
                names[i],
                i == Matrix || i == LocalMatrix ? bgfx::UniformType::Mat4 : bgfx::UniformType::Vec4,
                i == Matrix                            ? 3
                : i >= LocalMatrix && i <= LocalParams ? 8
                : i == LocalSlots                      ? 100
                                                       : 1);
        mAtlasSampler = bgfx::createUniform("s_shadowAtlas", bgfx::UniformType::Sampler);
        mLocalSampler = bgfx::createUniform("s_localShadow", bgfx::UniformType::Sampler);
        mAtlasFallback = bgfx::createTexture2D(1, 1, false, 1, bgfx::TextureFormat::D32F,
                                               BGFX_TEXTURE_RT | BGFX_SAMPLER_COMPARE_LEQUAL);
        mLocalFallback = bgfx::createTexture2D(1, 1, false, 2, bgfx::TextureFormat::R16F, 0);
    }
    void release() {
        for (auto h : mHandles)
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        for (auto h : {mAtlasSampler, mLocalSampler})
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        for (auto h : {mAtlasFallback, mLocalFallback})
            if (bgfx::isValid(h))
                bgfx::destroy(h);
    }
    void bind(const ShadowMaps &s, int quality) {
        const float config[4] = {float(s.directionalLight), float(s.localCount), float(quality),
                                 bgfx::getCaps()->originBottomLeft ? 0.f : 1.f};
        bgfx::setUniform(mHandles[Config], config);
        bgfx::setUniform(mHandles[Matrix], s.matrices.data(), 3);
        bgfx::setUniform(mHandles[Splits], s.splits.data());
        bgfx::setUniform(mHandles[Texels], s.texels.data());
        bgfx::setUniform(mHandles[LocalMatrix], s.localMatrices.data(), 8);
        bgfx::setUniform(mHandles[LocalPosition], s.positions.data(), 8);
        bgfx::setUniform(mHandles[LocalParams], s.localParams.data(), 8);
        bgfx::setUniform(mHandles[LocalSlots], s.slots.data(), 100);
        bgfx::setTexture(3, mAtlasSampler, bgfx::isValid(s.atlas) ? s.atlas : mAtlasFallback);
        bgfx::setTexture(4, mLocalSampler, bgfx::isValid(s.local) ? s.local : mLocalFallback);
    }
};
} // namespace mojive
