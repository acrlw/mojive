#pragma once
#include <algorithm>
#include <array>
#include <bgfx/bgfx.h>
#include <cmath>
#include <mojive/Render.hpp>

namespace mojive {
class LightingUniforms {
    enum Slot {
        Options,
        CameraPosition,
        CameraDirection,
        Ambient,
        HeadlightDiffuse,
        HeadlightSpecular,
        Fog,
        FogColor,
        HazeColor,
        DepthRange,
        LightCount,
        LightPosition,
        LightDirection,
        LightDiffuse,
        LightSpecular,
        LightAttenuation,
        ImageLight,
        Count
    };
    std::array<bgfx::UniformHandle, Count> mHandles;
    std::array<std::array<std::array<float, 4>, 100>, 5> mLights{};
    static float linear(float value) {
        return value <= .04045f ? value / 12.92f : std::pow((value + .055f) / 1.055f, 2.4f);
    }
    void set(Slot slot, const std::array<float, 4> &value) {
        bgfx::setUniform(mHandles[slot], value.data());
    }

  public:
    LightingUniforms() {
        mHandles.fill(bgfx::UniformHandle{bgfx::kInvalidHandle});
    }
    void initialize() {
        const char *names[] = {"u_options",
                               "u_cameraPosition",
                               "u_cameraDirection",
                               "u_ambient",
                               "u_headlightDiffuse",
                               "u_headlightSpecular",
                               "u_fog",
                               "u_fogColor",
                               "u_hazeColor",
                               "u_depthRange",
                               "u_lightCount",
                               "u_lightPosition",
                               "u_lightDirection",
                               "u_lightDiffuse",
                               "u_lightSpecular",
                               "u_lightAttenuation",
                               "u_imageLight"};
        for (size_t i = 0; i < Count; ++i)
            mHandles[i] =
                bgfx::createUniform(names[i], bgfx::UniformType::Vec4,
                                    i >= LightPosition && i <= LightAttenuation ? 100 : 1);
    }
    void release() {
        for (auto handle : mHandles)
            if (bgfx::isValid(handle))
                bgfx::destroy(handle);
        mHandles.fill(bgfx::UniformHandle{bgfx::kInvalidHandle});
    }
    void prepare(const Lighting &light) {
        for (size_t i = 0; i < light.lights.size(); ++i) {
            const auto &l = light.lights[i];
            const double length =
                std::hypot(double(l.direction[0]), double(l.direction[1]), double(l.direction[2]));
            const std::array<float, 3> direction =
                length > 1e-9 ? std::array<float, 3>{float(l.direction[0] / length),
                                                     float(l.direction[1] / length),
                                                     float(l.direction[2] / length)}
                              : std::array<float, 3>{0, 0, -1};
            mLights[0][i] = {l.position[0], l.position[1], l.position[2], float(l.type)};
            mLights[1][i] = {direction[0], direction[1], direction[2],
                             std::cos(std::clamp(l.cutoff, 0.f, 180.f) * 0.017453292519943295f)};
            mLights[2][i] = {linear(l.diffuse[0]), linear(l.diffuse[1]), linear(l.diffuse[2]),
                             l.exponent};
            mLights[3][i] = {linear(l.specular[0]), linear(l.specular[1]), linear(l.specular[2]),
                             0};
            mLights[4][i] = {l.attenuation[0], l.attenuation[1], l.attenuation[2], l.range};
        }
    }
    void bind(const Lighting &light, const SceneStyle &style, const CameraView &camera,
              bool linearColors, float imageMip, const CameraView *headlightCamera = nullptr) {
        set(Options, {light.enabled ? 1.f : 0.f, linearColors ? 0.f : 1.f,
                      style.tonemap ? 1.f : 0.f, float(style.debugView)});
        const auto &v = camera.view;
        set(CameraPosition, {-(v[0] * v[3] + v[4] * v[7] + v[8] * v[11]),
                             -(v[1] * v[3] + v[5] * v[7] + v[9] * v[11]),
                             -(v[2] * v[3] + v[6] * v[7] + v[10] * v[11]), 1});
        const auto &directionView = headlightCamera ? headlightCamera->view : v;
        set(CameraDirection, {-directionView[8], -directionView[9], -directionView[10], 0});
        set(Ambient, {light.ambient[0], light.ambient[1], light.ambient[2], 0});
        set(HeadlightDiffuse, {linear(light.headlightDiffuse[0]), linear(light.headlightDiffuse[1]),
                               linear(light.headlightDiffuse[2]), light.headlightDiffuse[3]});
        set(HeadlightSpecular,
            {linear(light.headlightSpecular[0]), linear(light.headlightSpecular[1]),
             linear(light.headlightSpecular[2]), 0});
        set(Fog, {light.fog[0], light.fog[1], style.fog && light.fog[1] > light.fog[0] ? 1.f : 0.f,
                  style.haze ? light.fog[3] : 0.f});
        set(FogColor, {light.fogColor[0], light.fogColor[1], light.fogColor[2], 0});
        set(HazeColor, {light.hazeColor[0], light.hazeColor[1], light.hazeColor[2], 0});
        set(DepthRange, {camera.nearPlane, camera.farPlane, 0, 0});
        set(ImageLight,
            {light.imageTexture >= 0 ? light.imageIntensity / 5000.f : 0.f, imageMip, 0, 0});
        set(LightCount, {float(light.lights.size()), 0, 0, 0});
        if (!light.lights.empty())
            for (size_t i = 0; i < mLights.size(); ++i)
                bgfx::setUniform(mHandles[LightPosition + i], mLights[i].data(),
                                 light.lights.size());
    }
};
} // namespace mojive
