#pragma once
#include <algorithm>
#include <bgfx/bgfx.h>
#include <cmath>
#include <glm/gtc/matrix_inverse.hpp>
#include <mojive/Render.hpp>
#include <numbers>

namespace mojive {
class EnvironmentGeometry {
    bgfx::VertexLayout mLayout;
    bgfx::VertexBufferHandle mFullscreen = BGFX_INVALID_HANDLE;
    bgfx::VertexBufferHandle mCylinder = BGFX_INVALID_HANDLE, mHaze = BGFX_INVALID_HANDLE;
    int mSlices = 0;

  public:
    void initialize() {
        mLayout.begin().add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float).end();
        const float triangle[9] = {-1, -1, 0, 3, -1, 0, -1, 3, 0};
        mFullscreen = bgfx::createVertexBuffer(bgfx::copy(triangle, sizeof(triangle)), mLayout);
    }
    void release() {
        for (auto h : {mFullscreen, mCylinder, mHaze})
            if (bgfx::isValid(h))
                bgfx::destroy(h);
    }
    void prepare(int slices) {
        slices = std::clamp(slices, 3, 4096);
        if (slices == mSlices)
            return;
        for (auto h : {mCylinder, mHaze})
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        std::vector<glm::vec3> cylinder, haze;
        auto quad = [](auto &out, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {
            out.insert(out.end(), {a, b, c, a, c, d});
        };
        for (int i = 0; i < slices; ++i) {
            double a = 2 * std::numbers::pi * i / slices,
                   b = 2 * std::numbers::pi * (i + 1) / slices;
            float x0 = std::cos(a), y0 = std::sin(a), x1 = std::cos(b), y1 = std::sin(b);
            quad(cylinder, {x0, y0, -1}, {x1, y1, -1}, {x1, y1, 1}, {x0, y0, 1});
            cylinder.insert(
                cylinder.end(),
                {{0, 0, 1}, {x0, y0, 1}, {x1, y1, 1}, {0, 0, -1}, {x1, y1, -1}, {x0, y0, -1}});
            for (int layer = 0; layer < 2; ++layer)
                quad(haze, {x0, y0, layer}, {x1, y1, layer}, {x1, y1, layer + 1},
                     {x0, y0, layer + 1});
        }
        mCylinder = bgfx::createVertexBuffer(
            bgfx::copy(cylinder.data(), cylinder.size() * sizeof(glm::vec3)), mLayout);
        mHaze = bgfx::createVertexBuffer(bgfx::copy(haze.data(), haze.size() * sizeof(glm::vec3)),
                                         mLayout);
        mSlices = slices;
    }
    void bindSky(bool classic) {
        bgfx::setVertexBuffer(0, classic ? mCylinder : mFullscreen);
    }
    void bindHaze() {
        bgfx::setVertexBuffer(0, mHaze);
    }
};
} // namespace mojive
