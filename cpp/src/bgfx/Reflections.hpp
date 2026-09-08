#pragma once
#include <algorithm>
#include <bgfx/bgfx.h>
#include <glm/gtc/type_ptr.hpp>
#include <mojive/Render.hpp>
#include <unordered_set>

namespace mojive {
struct ReflectionGroup {
    glm::vec4 plane;
    std::vector<uint32_t> instances;
};
struct ReflectionMaps {
    std::vector<bgfx::TextureHandle> colors;
    std::vector<bgfx::FrameBufferHandle> frames;
    bgfx::TextureHandle depth = BGFX_INVALID_HANDLE;
    Extent size;
    std::vector<ReflectionGroup> groups;
    std::vector<uint8_t> instanceLayers;
    std::unordered_set<uint64_t> excluded;
    void release() {
        for (auto h : frames)
            bgfx::destroy(h);
        for (auto h : colors)
            bgfx::destroy(h);
        if (bgfx::isValid(depth))
            bgfx::destroy(depth);
        frames.clear();
        colors.clear();
        depth = BGFX_INVALID_HANDLE;
    }
    static uint64_t bucket(uint32_t mesh, uint32_t material) {
        return uint64_t(mesh) << 32 | material;
    }
    void prepare(const SceneSource &source, const std::vector<std::array<float, 40>> &instances,
                 const SceneStyle &style, const CameraView &camera, Extent extent) {
        groups.clear();
        excluded.clear();
        instanceLayers.assign(instances.size(), 0);
        if (!style.reflections || source.planarKinds.empty())
            return;
        std::vector<uint32_t> candidates;
        for (uint32_t i = 0; i < instances.size(); ++i)
            if (source.planarKinds[i] && instances[i][35] > 0)
                candidates.push_back(i);
        std::stable_sort(candidates.begin(), candidates.end(),
                         [&](auto a, auto b) { return instances[a][35] > instances[b][35]; });
        for (auto index : candidates) {
            const auto &p = instances[index];
            const glm::vec3 x(p[0], p[4], p[8]), y(p[1], p[5], p[9]), z(p[2], p[6], p[10]);
            float determinant = glm::dot(x, glm::cross(y, z));
            if (std::abs(determinant) < 1e-15)
                continue;
            auto normal = glm::normalize(glm::cross(x, y) / determinant);
            auto point = glm::vec3(p[3], p[7], p[11]);
            if (source.planarKinds[index] == 2)
                point += z;
            glm::vec4 plane(normal, -glm::dot(normal, point));
            auto it = std::find_if(groups.begin(), groups.end(), [&](const auto &g) {
                return glm::all(glm::lessThanEqual(glm::abs(g.plane - plane), glm::vec4(1e-5))) ||
                       glm::all(glm::lessThanEqual(glm::abs(g.plane + plane), glm::vec4(1e-5)));
            });
            if (it != groups.end())
                it->instances.push_back(index);
            else if (groups.size() < 4)
                groups.push_back({plane, {index}});
        }
        auto view = glm::transpose(glm::make_mat4(camera.view.data()));
        auto eye = glm::inverse(view)[3];
        std::erase_if(groups, [&](const auto &g) { return glm::dot(g.plane, eye) <= 1e-4; });
        for (size_t layer = 0; layer < groups.size(); ++layer)
            for (auto index : groups[layer].instances) {
                instanceLayers[index] = layer + 1;
                excluded.insert(
                    bucket(source.instances[index].mesh,
                           source.materialIndices.empty() ? 0 : source.materialIndices[index]));
            }
        if (groups.empty())
            return;
        if (size == extent && frames.size() == groups.size())
            return;
        release();
        size = extent;
        depth = bgfx::createTexture2D(size.width, size.height, false, 1, bgfx::TextureFormat::D32F,
                                      BGFX_TEXTURE_RT);
        if (!bgfx::isValid(depth))
            throw std::runtime_error("Cannot allocate reflection depth");
        for (size_t i = 0; i < groups.size(); ++i) {
            auto color = bgfx::createTexture2D(
                size.width, size.height, false, 1, bgfx::TextureFormat::RGBA16F,
                BGFX_TEXTURE_RT | BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP);
            if (!bgfx::isValid(color)) {
                release();
                throw std::runtime_error("Cannot allocate reflection color");
            }
            colors.push_back(color);
            const bgfx::TextureHandle attachments[] = {color, depth};
            auto frame = bgfx::createFrameBuffer(2, attachments, false);
            if (!bgfx::isValid(frame)) {
                release();
                throw std::runtime_error("Cannot allocate reflection framebuffer");
            }
            frames.push_back(frame);
        }
    }
    CameraView camera(const CameraView &base, size_t layer) const {
        const auto plane = groups[layer].plane;
        const glm::vec3 normal(plane);
        glm::mat4 mirror(1);
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r)
                mirror[c][r] -= 2 * normal[c] * normal[r];
        mirror[3] = glm::vec4(-2 * plane.w * normal, 1);
        auto view = glm::transpose(glm::make_mat4(base.view.data())) * mirror;
        auto result = base;
        auto row = glm::transpose(view);
        std::memcpy(result.view.data(), &row[0][0], sizeof(result.view));
        return result;
    }
};
class ReflectionUniforms {
    bgfx::UniformHandle mConfig = BGFX_INVALID_HANDLE, mPlane = BGFX_INVALID_HANDLE;
    std::array<bgfx::UniformHandle, 4> mSamplers{
        bgfx::UniformHandle{bgfx::kInvalidHandle}, bgfx::UniformHandle{bgfx::kInvalidHandle},
        bgfx::UniformHandle{bgfx::kInvalidHandle}, bgfx::UniformHandle{bgfx::kInvalidHandle}};

  public:
    void initialize() {
        mConfig = bgfx::createUniform("u_reflectionConfig", bgfx::UniformType::Vec4);
        mPlane = bgfx::createUniform("u_reflectionPlane", bgfx::UniformType::Vec4);
        const char *names[] = {"s_reflection0", "s_reflection1", "s_reflection2", "s_reflection3"};
        for (int i = 0; i < 4; ++i)
            mSamplers[i] = bgfx::createUniform(names[i], bgfx::UniformType::Sampler);
    }
    void release() {
        for (auto h : {mConfig, mPlane})
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        for (auto h : mSamplers)
            if (bgfx::isValid(h))
                bgfx::destroy(h);
    }
    void bind(const ReflectionMaps &maps, int layer, bgfx::TextureHandle fallback) {
        const float config[] = {layer >= 0 ? 1.f : 0.f, float(maps.size.width),
                                float(maps.size.height), 0};
        bgfx::setUniform(mConfig, config);
        const glm::vec4 plane = layer >= 0 ? maps.groups[layer].plane : glm::vec4(0);
        bgfx::setUniform(mPlane, &plane[0]);
        for (int i = 0; i < 4; ++i)
            bgfx::setTexture(5 + i, mSamplers[i],
                             layer < 0 && size_t(i) < maps.groups.size() ? maps.colors[i]
                                                                         : fallback);
    }
};
} // namespace mojive
