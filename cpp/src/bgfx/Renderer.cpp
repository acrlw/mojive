#include "../Rgb.hpp"
#include "Environment.hpp"
#include "InstanceStream.hpp"
#include "Lighting.hpp"
#include "Reflections.hpp"
#include "Shadows.hpp"
#include "Timing.hpp"
#include "Visibility.hpp"
#include <algorithm>
#include <atomic>
#include <bgfx/bgfx.h>
#include <bgfx/defines.h>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <limits>
#include <map>
#include <mojive/backends/Bgfx.hpp>
#include <optional>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>

namespace mojive {
namespace {
std::atomic_flag runtimeActive = ATOMIC_FLAG_INIT;
std::atomic<uint64_t> resourceSequence{1};
uint64_t allocateId() {
    return resourceSequence.fetch_add(1, std::memory_order_relaxed);
}
constexpr uint64_t sampler =
    BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP | BGFX_SAMPLER_MIN_POINT | BGFX_SAMPLER_MAG_POINT;
constexpr uint64_t targetTextureBit = uint64_t{1} << 63;
Matrix columnMajor(const Matrix &input) {
    Matrix result;
    for (size_t r = 0; r < 4; ++r)
        for (size_t c = 0; c < 4; ++c)
            result[c * 4 + r] = input[r * 4 + c];
    return result;
}
bool complete(uint32_t current, uint32_t ready) {
    return std::bit_cast<int32_t>(current - ready) >= 0;
}
uint32_t word(const std::byte *p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
struct GpuMesh {
    bgfx::DynamicVertexBufferHandle vertices = BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle indices = BGFX_INVALID_HANDLE;
    bgfx::DynamicVertexBufferHandle wireVertices = BGFX_INVALID_HANDLE;
    size_t vertexCount = 0;
    MeshBounds bounds;
    std::shared_ptr<const Mesh> source;
    std::vector<Vertex> deformed;
    ~GpuMesh() {
        for (auto handle : {vertices, wireVertices})
            if (bgfx::isValid(handle))
                bgfx::destroy(handle);
        if (bgfx::isValid(indices))
            bgfx::destroy(indices);
    }
};
struct GpuTexture {
    bgfx::TextureHandle handle = BGFX_INVALID_HANDLE;
    ~GpuTexture() {
        if (bgfx::isValid(handle))
            bgfx::destroy(handle);
    }
};
struct GpuVertex {
    Vertex vertex;
    std::array<float, 2> uv = {0, 0};
};
struct WireVertex {
    GpuVertex vertex;
    std::array<float, 3> bary;
};
struct GpuBatch {
    uint32_t mesh = 0, material = 0;
    std::vector<uint32_t> instances;
};
using ShadowKey = std::tuple<uint64_t, uint64_t, int, bool, std::array<float, 3>>;
using DataKey = std::tuple<uint64_t, uint64_t, Matrix, Matrix, float, float>;
using ReflectionKey =
    std::tuple<uint64_t, uint64_t, uint64_t, Matrix, Matrix, float, float, std::array<float, 3>>;
struct GpuScene {
    uint64_t geometryRevision = 1, lightingRevision = 1, styleRevision = 1;
    std::optional<ShadowKey> shadowKey;
    SceneStyle style;
    Lighting lighting;
    OverlayFrame overlays;
    bgfx::DynamicVertexBufferHandle surfaceBuffer = BGFX_INVALID_HANDLE;
    std::array<bgfx::DynamicVertexBufferHandle, 9> debugBuffers;
    GpuScene() {
        debugBuffers.fill(bgfx::DynamicVertexBufferHandle{bgfx::kInvalidHandle});
    }
    ShadowMaps shadows;
    std::vector<bgfx::TextureHandle> textures;
    std::vector<std::shared_ptr<GpuTexture>> textureOwners;
    std::vector<GpuBatch> batches;
    SceneSource source;
    uint64_t sequence = 0;
    std::vector<std::shared_ptr<GpuMesh>> meshes;
    std::vector<std::array<float, 40>> instances;
    std::vector<MeshBounds> worldBounds;
};
struct GpuTarget {
    Scene scene;
    bgfx::FrameBufferHandle selectionFrame = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle selectionMask = BGFX_INVALID_HANDLE;
    ReflectionMaps reflections;
    std::optional<ReflectionKey> reflectionKey, colorKey;
    std::optional<DataKey> dataKey;
    uint8_t cachedData = 0;
    Extent size;
    uint32_t samples = 1, allocatedSamples = 1;
    bool surface = false, colorOnly = false, dataOnly = false, vsync = false;
    std::optional<Product> dataProduct;
    uint16_t view = 0;
    uint32_t viewFrame = UINT32_MAX;
    uint64_t generation = 1;
    uint32_t lastRender = UINT32_MAX, lastReadback = UINT32_MAX;
    FrameToken latest;
    bgfx::SwapChain swapChain;
    bgfx::FrameBufferHandle colorFb = BGFX_INVALID_HANDLE;
    std::array<bgfx::FrameBufferHandle, 4> dataFbs = {
        {BGFX_INVALID_HANDLE, BGFX_INVALID_HANDLE, BGFX_INVALID_HANDLE, BGFX_INVALID_HANDLE}};
    bgfx::TextureHandle color = BGFX_INVALID_HANDLE;
    std::array<bgfx::TextureHandle, 3> data;
};
struct ReadbackSlot {
    uint64_t ticket = 0;
    bool canceled = false, gpuPacked = false, direct = false;
    uint32_t ready = 0;
    FrameToken frame;
    Product product = Product::Color;
    Extent size;
    bgfx::DynamicIndexBufferHandle buffer = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle resolved = BGFX_INVALID_HANDLE;
    uint32_t rowPitch = 0;
    std::vector<std::byte> bytes;
    std::span<std::byte> destination;
};

class BgfxRenderer final : public Renderer {
    bool mInitialized = false;
    std::thread::id mOwner = std::this_thread::get_id();
    Capabilities mCaps;
    std::string mShaderDirectory;
    WindowSystem mWindowSystem = WindowSystem::Native;
    std::unordered_map<uint64_t, GpuScene> mScenes{{0, GpuScene{}}};
    std::unordered_map<const Mesh *, std::weak_ptr<GpuMesh>> mMeshCache;
    using TextureKey = std::tuple<const void *, uint32_t, uint32_t, bool, bool, bool>;
    struct TextureEntry {
        std::weak_ptr<const std::vector<std::byte>> pixels;
        std::weak_ptr<GpuTexture> gpu;
    };
    std::map<TextureKey, TextureEntry> mTextureCache;
    ResourceStats mResources;
    uint64_t mSubmission = 0;
    uint32_t mGpuFrame = 0;
    // Submit presentation with its rendered frame, including when the caller pauses.
    uint32_t mResetFlags = BGFX_RESET_MAXANISOTROPY | BGFX_RESET_FLIP_AFTER_RENDER;
    bool mPendingCommands = false;
    std::array<bgfx::ViewId, 256> mPassOrder{};
    std::array<bool, 256> mUsedPasses{};
    uint16_t mPassCount = 0;
    bgfx::ViewId mNextSceneView = 48;
    uint32_t mMainRender = UINT32_MAX;
    PassTiming mTiming;
    Target mTimingTarget;
    uint64_t mTimingSubmission = 0;
    void timePass(bgfx::ViewId view, RenderPass pass) {
        mTiming.record(view, mTimingTarget, mTimingSubmission, pass);
    }
    void orderPass(bgfx::ViewId view) {
        if (!mUsedPasses[view]) {
            mUsedPasses[view] = true;
            mPassOrder[mPassCount++] = view;
        }
    }
    InstanceStream mInstances;
    void flush() {
        mInstances.flush();
        // bgfx sorts views numerically unless explicitly remapped. Preserve API
        // dependency order even when a sampled target was allocated later.
        auto count = mPassCount;
        for (bgfx::ViewId id = 0; id < mPassOrder.size(); ++id)
            if (!mUsedPasses[id])
                mPassOrder[count++] = id;
        bgfx::setViewOrder(0, mPassOrder.size(), mPassOrder.data());
        mGpuFrame = bgfx::frame();
        mTiming.collect(mGpuFrame, *bgfx::getStats());
        mInstances.reset();
        mPendingCommands = false;
        mPassCount = 0;
        mNextSceneView = 48;
        mTargetViews = 0;
        mUsedPasses.fill(false);
    }
    void assignTargetView(GpuTarget &target) {
        if (target.viewFrame == mGpuFrame)
            return;
        // View and clear-palette slots belong to a submitted GPU frame, not to
        // long-lived targets. Flush before reusing them for another target.
        if (mTargetViews == 12)
            flush();
        target.view = mTargetViews++ * 4;
        target.viewFrame = mGpuFrame;
    }
    void beginTarget(GpuTarget &target) {
        if (target.lastRender == mGpuFrame || target.lastReadback == mGpuFrame)
            flush();
        assignTargetView(target);
        target.lastRender = mGpuFrame;
        mPendingCommands = true;
    }
    std::unordered_map<uint64_t, GpuTarget> mTargets;
    std::unordered_map<uint64_t, bgfx::TextureHandle> mTextures;
    std::vector<ReadbackSlot> mReadbacks;
    uint16_t mTargetViews = 0;
    FrameStats mStats;
    bgfx::VertexLayout mVertices, mUiVertices, mWireVertices;
    std::vector<uint32_t> mDrawIndices;
    std::array<bgfx::VertexLayout, 9> mDebugLayouts;
    std::array<bgfx::ProgramHandle, 9> mDebugPrograms = [] {
        std::array<bgfx::ProgramHandle, 9> handles;
        handles.fill(bgfx::ProgramHandle{bgfx::kInvalidHandle});
        return handles;
    }();
    std::vector<float> mDebugUpload;
    bgfx::VertexBufferHandle mDebugVertices = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle mDebugViewProj = BGFX_INVALID_HANDLE, mDebugProj = BGFX_INVALID_HANDLE,
                        mDebugParams = BGFX_INVALID_HANDLE, mDebugDepth = BGFX_INVALID_HANDLE,
                        mDebugAtlas = BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle mMaskProgram = BGFX_INVALID_HANDLE, mOutlineProgram = BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle mIdentityProgram = BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle mReadbackProgram = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle mReadbackImage = BGFX_INVALID_HANDLE, mReadbackRegion = BGFX_INVALID_HANDLE,
                        mReadbackLayout = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle mUiTextureInfo = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle mIdentitySampler = BGFX_INVALID_HANDLE, mIdentitySize = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle mMaskSampler = BGFX_INVALID_HANDLE, mOutlineSize = BGFX_INVALID_HANDLE,
                        mOutlineColor = BGFX_INVALID_HANDLE;
    ReflectionUniforms mReflections;
    bgfx::ProgramHandle mGizmoProgram = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle mGizmoColor = BGFX_INVALID_HANDLE, mGizmoParams = BGFX_INVALID_HANDLE;
    EnvironmentGeometry mEnvironment;
    bgfx::ProgramHandle mSkyProgram = BGFX_INVALID_HANDLE, mClassicSkyProgram = BGFX_INVALID_HANDLE,
                        mHazeProgram = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle mSkySampler = BGFX_INVALID_HANDLE, mSkyInverse = BGFX_INVALID_HANDLE,
                        mSkyEyeDistance = BGFX_INVALID_HANDLE;
    std::array<bgfx::UniformHandle, 4> mHazeUniforms = {
        bgfx::UniformHandle{bgfx::kInvalidHandle}, bgfx::UniformHandle{bgfx::kInvalidHandle},
        bgfx::UniformHandle{bgfx::kInvalidHandle}, bgfx::UniformHandle{bgfx::kInvalidHandle}};
    LightingUniforms mLighting;
    bgfx::VertexLayout mSurfaceLayout;
    ShadowUniforms mShadows;
    bgfx::ProgramHandle mShadowProgram = BGFX_INVALID_HANDLE,
                        mDistanceProgram = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle mShadowLightPosition = BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle mLitProgram = BGFX_INVALID_HANDLE, mWireProgram = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle mCubeSampler = BGFX_INVALID_HANDLE,
                        mImageLightSampler = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle mWhiteCube = BGFX_INVALID_HANDLE;
    std::vector<std::pair<float, uint32_t>> mTransparent;
    bgfx::UniformHandle mMaterial = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle mWhite = BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle mColorProgram = BGFX_INVALID_HANDLE, mUiProgram = BGFX_INVALID_HANDLE;
    std::array<bgfx::ProgramHandle, 4> mDataPrograms = {
        {BGFX_INVALID_HANDLE, BGFX_INVALID_HANDLE, BGFX_INVALID_HANDLE, BGFX_INVALID_HANDLE}};
    bgfx::UniformHandle mImageSampler = BGFX_INVALID_HANDLE;
    Extent mWindowSize;
    bool mHasWindow = false;
    bgfx::SwapChain mSwapChain;

    void owner() const {
        if (std::this_thread::get_id() != mOwner)
            throw std::logic_error("Renderer called outside its owner thread");
    }
    GpuScene &scene(Scene id) {
        auto it = mScenes.find(id.id);
        if (it == mScenes.end())
            throw std::invalid_argument("Unknown scene");
        return it->second;
    }
    void releaseScene(GpuScene &scene) {
        if (bgfx::isValid(scene.surfaceBuffer))
            bgfx::destroy(scene.surfaceBuffer);
        scene.shadows.release();
        for (auto h : scene.debugBuffers)
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        scene.textures.clear();
        scene.textureOwners.clear();
        scene.meshes.clear();
    }
    bool renderShadows(GpuScene &scene, const CameraView &camera) {
        ShadowKey key{scene.geometryRevision, scene.lightingRevision, scene.style.shadowQuality,
                      scene.style.shadows,
                      std::ranges::any_of(
                          scene.lighting.lights,
                          [](const Light &light) { return light.type == 0 && light.castShadow; })
                          ? camera.focus
                          : std::array<float, 3>{}};
        if (scene.shadowKey == key)
            return false;
        scene.shadowKey.reset();
        auto &shadow = scene.shadows;
        shadow.prepare(scene.source, scene.lighting, scene.style, camera);
        auto draw = [&](bgfx::ViewId view, const glm::mat4 &matrix, bgfx::FrameBufferHandle frame,
                        int x, int y, int pixels, const float *light) {
            orderPass(view);
            timePass(view, RenderPass::Shadow);
            const ClipFrustum frustum(matrix);
            auto projection = matrix;
            if (!bgfx::getCaps()->homogeneousDepth)
                for (int c = 0; c < 4; ++c)
                    projection[c][2] = (matrix[c][2] + matrix[c][3]) * .5f;
            const glm::mat4 identity(1);
            bgfx::setViewTransform(view, &identity[0][0], &projection[0][0]);
            bgfx::setViewRect(view, x, y, pixels, pixels);
            bgfx::setViewFrameBuffer(view, frame);
            bgfx::setViewMode(view, bgfx::ViewMode::Default);
            if (light) {
                const float far[4] = {65504, 0, 0, 1};
                bgfx::setPaletteColor(15, far);
                bgfx::setViewClear(view, BGFX_CLEAR_COLOR, 1.f, 0, 15);
            } else
                bgfx::setViewClear(view, BGFX_CLEAR_DEPTH, uint32_t{0}, 1.f);
            bgfx::touch(view);
            for (const auto &batch : scene.batches) {
                mDrawIndices.clear();
                for (auto index : batch.instances) {
                    if (scene.instances[index][19] < 1)
                        continue;
                    if (frustum.intersects(scene.worldBounds[index])) {
                        mDrawIndices.push_back(index);
                        ++mStats.shadowInstances;
                    } else
                        ++mStats.culledShadowInstances;
                }
                uint32_t count = mDrawIndices.size();
                if (!count)
                    continue;
                constexpr uint16_t stride = 12 * sizeof(float);
                auto *buffer = mInstances.allocate(count, stride).data();
                for (size_t i = 0; i < count; ++i)
                    std::memcpy(buffer + i * stride, scene.instances[mDrawIndices[i]].data(),
                                stride);
                const auto &mesh = *scene.meshes[batch.mesh];
                bgfx::setVertexBuffer(0, mesh.vertices);
                bgfx::setIndexBuffer(mesh.indices);
                uint64_t state = BGFX_STATE_CULL_CW;
                if (light) {
                    bgfx::setUniform(mShadowLightPosition, light);
                    state |= BGFX_STATE_WRITE_RGB |
                             BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_ONE) |
                             BGFX_STATE_BLEND_EQUATION(BGFX_STATE_BLEND_EQUATION_MIN);
                } else
                    state |= BGFX_STATE_DEPTH_TEST_LESS | BGFX_STATE_WRITE_Z;
                bgfx::setState(state);
                bgfx::submit(view, light ? mDistanceProgram : mShadowProgram);
                ++mStats.drawCalls;
                mStats.uploadBytes += count * stride;
            }
        };
        if (shadow.directionalLight >= 0)
            for (int i = 0; i < 3; ++i)
                draw(mNextSceneView++, shadow.matrices[i], shadow.atlasFrame, (i % 2) * 2048,
                     (bgfx::getCaps()->originBottomLeft ? 1 - i / 2 : i / 2) * 2048, 2048, nullptr);
        for (int slot = 0; slot < shadow.localCount; ++slot) {
            int base = int(shadow.localParams[slot][2]);
            for (int face = 0; face < (shadow.kinds[slot] == 2 ? 1 : 6); ++face)
                draw(mNextSceneView++,
                     shadow.kinds[slot] == 2 ? shadow.localMatrices[slot]
                                             : shadow.pointMatrices[slot][face],
                     shadow.localFrames[base + face], 0, 0, shadow.pixels,
                     shadow.positions[slot].data());
        }
        scene.shadowKey = key;
        return true;
    }
    void renderEnvironment(const GpuScene &scene, const GpuTarget &target,
                           const CameraView &camera) {
        const auto &light = scene.lighting;
        const int texture = light.skyboxTexture;
        if (!light.enabled || !scene.style.skybox || scene.style.debugView || texture < 0 ||
            size_t(texture) >= scene.textures.size() || !scene.source.textures[texture].cube)
            return;
        bool classic = !scene.source.linearColors;
        mEnvironment.prepare(light.hazeSlices);
        auto eye = glm::inverse(glm::transpose(glm::make_mat4(camera.view.data())))[3];
        const float eyeDistance[] = {eye.x, eye.y, eye.z, camera.farPlane * .7f};
        auto vp = glm::transpose(glm::make_mat4(camera.projection.data())) *
                  glm::transpose(glm::make_mat4(camera.view.data()));
        auto inverse = glm::inverse(vp);
        bgfx::setUniform(mSkyInverse, &inverse[0][0]);
        bgfx::setUniform(mSkyEyeDistance, eyeDistance);
        mLighting.bind(light, scene.style, camera, scene.source.linearColors, 0);
        bgfx::setTexture(0, mSkySampler, scene.textures[texture]);
        mEnvironment.bindSky(classic);
        uint64_t state = BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_DEPTH_TEST_LEQUAL;
        if (classic)
            state |= BGFX_STATE_WRITE_Z;
        if (target.samples > 1 && scene.style.msaa)
            state |= BGFX_STATE_MSAA;
        bgfx::setState(state);
        bgfx::submit(target.view, classic ? mClassicSkyProgram : mSkyProgram);
        ++mStats.drawCalls;
        if (!scene.style.haze || !light.horizonHaze || light.hazeDensity <= 0 ||
            scene.source.infinitePlanes.empty())
            return;
        const auto &pose = scene.instances[scene.source.infinitePlanes[0]];
        auto normal = glm::normalize(glm::vec3(pose[2], pose[6], pose[10]));
        auto basisX = glm::normalize(glm::vec3(pose[0], pose[4], pose[8]));
        auto basisY = glm::normalize(glm::vec3(pose[1], pose[5], pose[9]));
        float elevation = glm::dot(glm::vec3(eye) - glm::vec3(pose[3], pose[7], pose[11]), normal);
        if (elevation < 0)
            return;
        double radius = light.hazeDensity, alpha = std::atan2(1.0, radius),
               beta = .75 * std::numbers::pi - alpha;
        float transition = std::sqrt(.5) * radius * std::sin(alpha) / std::sin(beta);
        const float geometry[] = {camera.farPlane * .7f, elevation, float(radius), transition};
        bgfx::setUniform(mHazeUniforms[0], geometry);
        int slot = 1;
        for (auto axis : {basisX, basisY, normal}) {
            const glm::vec4 value(axis, 0);
            bgfx::setUniform(mHazeUniforms[slot++], &value[0]);
        }
        bgfx::setUniform(mSkyEyeDistance, eyeDistance);
        mLighting.bind(light, scene.style, camera, scene.source.linearColors, 0);
        mEnvironment.bindHaze();
        bgfx::setState(state | BGFX_STATE_BLEND_FUNC_SEPARATE(
                                   BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA,
                                   BGFX_STATE_BLEND_ZERO, BGFX_STATE_BLEND_ONE));
        bgfx::submit(target.view, mHazeProgram);
        ++mStats.drawCalls;
    }
    void renderOutline(const GpuScene &scene, GpuTarget &target, const Matrix &view,
                       const Matrix &projection) {
        const auto &style = scene.style;
        if (!style.selectedId || !style.outline || !style.selectionOutline)
            return;
        if (!bgfx::isValid(target.selectionFrame)) {
            uint64_t flags = BGFX_TEXTURE_RT | BGFX_TEXTURE_RT_MSAA_X4 | sampler;
            if (!bgfx::isTextureValid(0, false, 1, bgfx::TextureFormat::R8, flags))
                flags &= ~BGFX_TEXTURE_RT_MSAA_X4;
            target.selectionMask = bgfx::createTexture2D(target.size.width, target.size.height,
                                                         false, 1, bgfx::TextureFormat::R8, flags);
            if (!bgfx::isValid(target.selectionMask))
                throw std::runtime_error("Cannot allocate selection mask");
            target.selectionFrame = bgfx::createFrameBuffer(1, &target.selectionMask, true);
            if (!bgfx::isValid(target.selectionFrame)) {
                bgfx::destroy(target.selectionMask);
                target.selectionMask = BGFX_INVALID_HANDLE;
                throw std::runtime_error("Cannot allocate selection framebuffer");
            }
        }
        auto maskView = mNextSceneView++, compositeView = mNextSceneView++;
        timePass(maskView, RenderPass::Outline);
        timePass(compositeView, RenderPass::Outline);
        orderPass(maskView);
        orderPass(compositeView);
        bgfx::setViewRect(maskView, 0, 0, target.size.width, target.size.height);
        bgfx::setViewFrameBuffer(maskView, target.selectionFrame);
        bgfx::setViewTransform(maskView, view.data(), projection.data());
        bgfx::setViewClear(maskView, BGFX_CLEAR_COLOR, uint32_t{0});
        bgfx::touch(maskView);
        for (const auto &batch : scene.batches) {
            mDrawIndices.clear();
            for (auto index : batch.instances)
                if (scene.source.instances[index].objectId == style.selectedId &&
                    scene.instances[index][19] > 0)
                    mDrawIndices.push_back(index);
            uint32_t count = mDrawIndices.size();
            if (!count)
                continue;
            constexpr uint16_t stride = 12 * sizeof(float);
            auto *buffer = mInstances.allocate(count, stride).data();
            for (size_t i = 0; i < count; ++i)
                std::memcpy(buffer + i * stride, scene.instances[mDrawIndices[i]].data(), stride);
            bgfx::setVertexBuffer(0, scene.meshes[batch.mesh]->vertices);
            bgfx::setIndexBuffer(scene.meshes[batch.mesh]->indices);
            bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_CULL_CW | BGFX_STATE_MSAA);
            bgfx::submit(maskView, mMaskProgram);
            ++mStats.drawCalls;
            mStats.uploadBytes += count * stride;
        }
        bgfx::setViewRect(compositeView, 0, 0, target.size.width, target.size.height);
        bgfx::setViewFrameBuffer(compositeView, target.colorFb);
        bgfx::setViewClear(compositeView, BGFX_CLEAR_NONE);
        bgfx::setViewMode(compositeView, bgfx::ViewMode::Sequential);
        const float options[] = {float(target.size.width), float(target.size.height),
                                 style.selectionXray ? .12f : 0.f, 0};
        const float color[] = {1, .63f, .20f, 1};
        bgfx::setUniform(mOutlineSize, options);
        bgfx::setUniform(mOutlineColor, color);
        bgfx::setTexture(0, mMaskSampler, target.selectionMask);
        mEnvironment.bindSky(false);
        bgfx::setState(BGFX_STATE_WRITE_RGB |
                       BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_SRC_ALPHA,
                                                      BGFX_STATE_BLEND_INV_SRC_ALPHA,
                                                      BGFX_STATE_BLEND_ZERO, BGFX_STATE_BLEND_ONE));
        bgfx::submit(compositeView, mOutlineProgram);
        ++mStats.drawCalls;
    }
    void renderIdentityColor(const GpuScene &scene, GpuTarget &target) {
        auto pass = mNextSceneView++;
        timePass(pass, RenderPass::Identity);
        orderPass(pass);
        bgfx::setViewRect(pass, 0, 0, target.size.width, target.size.height);
        bgfx::setViewFrameBuffer(pass, target.colorFb);
        bgfx::setViewClear(pass, BGFX_CLEAR_NONE);
        uint32_t selected = scene.style.debugView == 6 ? scene.style.selectedId : 0;
        const float options[] = {float(target.size.width), float(target.size.height),
                                 float(selected & 65535), float(selected >> 16)};
        bgfx::setUniform(mIdentitySize, options);
        bgfx::setTexture(0, mIdentitySampler, target.data[0], sampler);
        mEnvironment.bindSky(false);
        bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A);
        bgfx::submit(pass, mIdentityProgram);
        ++mStats.drawCalls;
    }
    void renderSurfaces(GpuScene &scene, GpuTarget &target, const CameraView &camera) {
        const auto &frame = scene.overlays;
        for (const auto &batch : frame.surfaceBatches) {
            if (!batch.count || (batch.transparent && !scene.style.transparent))
                continue;
            auto &mesh = *scene.meshes[batch.mesh];
            bool wire = scene.style.wireframe || scene.style.debugView == 5;
            if (wire) {
                if (!bgfx::isValid(mesh.wireVertices))
                    updateWireMesh(scene, batch.mesh);
                bgfx::setVertexBuffer(0, mesh.wireVertices);
            } else {
                bgfx::setVertexBuffer(0, mesh.vertices);
                bgfx::setIndexBuffer(mesh.indices);
            }
            bgfx::setInstanceDataBuffer(scene.surfaceBuffer, batch.start, batch.count);
            int texture = scene.style.textures ? batch.texture : -1;
            bool cube = texture >= 0 && scene.source.textures[texture].cube;
            bgfx::setTexture(0, mImageSampler,
                             texture >= 0 && !cube ? scene.textures[texture] : mWhite);
            bgfx::setTexture(1, mCubeSampler, cube ? scene.textures[texture] : mWhiteCube);
            int image = scene.lighting.imageTexture;
            bgfx::setTexture(2, mImageLightSampler,
                             image >= 0 ? scene.textures[image] : mWhiteCube);
            mReflections.bind(target.reflections, -1, mWhite);
            mShadows.bind(scene.shadows, scene.style.shadowQuality);
            mLighting.bind(
                scene.lighting, scene.style, camera, scene.source.linearColors,
                image >= 0 ? std::floor(std::log2(scene.source.textures[image].size.width)) : 0);
            uint64_t state = BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_DEPTH_TEST_LESS;
            if (scene.style.cullFace)
                state |= BGFX_STATE_CULL_CW;
            if (scene.style.msaa && target.samples > 1)
                state |= BGFX_STATE_MSAA;
            if (batch.transparent)
                state |= scene.style.additive
                             ? BGFX_STATE_BLEND_FUNC_SEPARATE(
                                   BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_ONE,
                                   BGFX_STATE_BLEND_ZERO, BGFX_STATE_BLEND_ONE)
                             : BGFX_STATE_BLEND_FUNC_SEPARATE(
                                   BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA,
                                   BGFX_STATE_BLEND_ZERO, BGFX_STATE_BLEND_ONE);
            else
                state |= BGFX_STATE_WRITE_Z;
            if (scene.style.debugView == 4)
                state = (state & (BGFX_STATE_CULL_MASK | BGFX_STATE_MSAA)) | BGFX_STATE_WRITE_RGB |
                        BGFX_STATE_BLEND_ADD;
            bgfx::setState(state);
            bgfx::submit(target.view, wire ? mWireProgram : mLitProgram);
            ++mStats.drawCalls;
        }
    }
    void renderGizmos(const GpuScene &scene, const GpuTarget &target, const Matrix &view,
                      const Matrix &projection) {
        if (scene.overlays.gizmos.empty())
            return;
        auto pass = mNextSceneView++;
        timePass(pass, RenderPass::Gizmo);
        orderPass(pass);
        bgfx::setViewRect(pass, 0, 0, target.size.width, target.size.height);
        bgfx::setViewFrameBuffer(pass, target.colorFb);
        bgfx::setViewTransform(pass, view.data(), projection.data());
        bgfx::setViewClear(pass, BGFX_CLEAR_NONE);
        bgfx::setViewMode(pass, bgfx::ViewMode::Sequential);
        for (const auto &draw : scene.overlays.gizmos) {
            auto model = columnMajor(draw.transform);
            bgfx::setTransform(model.data());
            bgfx::setUniform(mGizmoColor, draw.color.data());
            const float params[] = {draw.maskRadius, bgfx::getCaps()->homogeneousDepth ? 1.f : 0.f,
                                    0, 0};
            bgfx::setUniform(mGizmoParams, params);
            bgfx::setVertexBuffer(0, scene.meshes[draw.mesh]->vertices);
            bgfx::setIndexBuffer(scene.meshes[draw.mesh]->indices);
            uint64_t state = BGFX_STATE_WRITE_RGB |
                             BGFX_STATE_BLEND_FUNC_SEPARATE(
                                 BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA,
                                 BGFX_STATE_BLEND_ZERO, BGFX_STATE_BLEND_ONE);
            if (draw.depthTest)
                state |= BGFX_STATE_DEPTH_TEST_LESS;
            if (draw.depthWrite)
                state |= BGFX_STATE_WRITE_Z;
            if (draw.cullFace)
                state |= BGFX_STATE_CULL_CW;
            if (target.samples > 1 && scene.style.msaa)
                state |= BGFX_STATE_MSAA;
            bgfx::setState(state);
            bgfx::submit(pass, mGizmoProgram);
            ++mStats.drawCalls;
        }
    }
    void renderDebug(const GpuScene &scene, const GpuTarget &target, const CameraView &camera,
                     const Matrix &view, const Matrix &projection) {
        const auto &frame = scene.overlays;
        if (frame.debug.empty())
            return;
        auto pass = mNextSceneView++;
        timePass(pass, RenderPass::Debug);
        orderPass(pass);
        bgfx::setViewRect(pass, 0, 0, target.size.width, target.size.height);
        bgfx::setViewFrameBuffer(pass, target.colorFb);
        bgfx::setViewTransform(pass, view.data(), projection.data());
        bgfx::setViewClear(pass, BGFX_CLEAR_NONE);
        bgfx::setViewMode(pass, bgfx::ViewMode::Sequential);
        auto canonicalProj = columnMajor(camera.projection);
        auto vp = glm::make_mat4(canonicalProj.data()) * glm::make_mat4(view.data());
        float pxScale = 2.f / (camera.projection[5] * target.size.height);
        const uint32_t vertices[] = {6, 15, 6, 24, 0, 96, 6, 3, 6};
        auto draw = [&](const DebugBatch &batch, bool ghost) {
            if (!batch.count)
                return;
            size_t path = size_t(batch.path);
            uint64_t state = BGFX_STATE_WRITE_RGB |
                             BGFX_STATE_BLEND_FUNC_SEPARATE(
                                 BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA,
                                 BGFX_STATE_BLEND_ZERO, BGFX_STATE_BLEND_ONE);
            if (batch.occlusion != Occlusion::Always)
                state |= ghost ? BGFX_STATE_DEPTH_TEST_GREATER : BGFX_STATE_DEPTH_TEST_LESS;
            if (target.samples > 1 && scene.style.msaa)
                state |= BGFX_STATE_MSAA;
            const float params[] = {float(target.size.width), float(target.size.height), pxScale,
                                    ghost ? .28f : 1.f};
            const float depth[] = {bgfx::getCaps()->homogeneousDepth ? 1.f : 0.f, 0, 0, 0};
            bgfx::setUniform(mDebugParams, params);
            bgfx::setUniform(mDebugDepth, depth);
            bgfx::setUniform(mDebugViewProj, &vp[0][0]);
            bgfx::setUniform(mDebugProj, canonicalProj.data());
            if (batch.path == DebugPath::Solid) {
                bgfx::setVertexBuffer(0, scene.meshes[batch.mesh]->vertices);
                bgfx::setIndexBuffer(scene.meshes[batch.mesh]->indices);
            } else
                bgfx::setVertexBuffer(0, mDebugVertices, 0, vertices[path]);
            if (batch.path == DebugPath::Text)
                bgfx::setTexture(0, mDebugAtlas, mTextures.at(frame.glyphAtlas.id),
                                 BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP);
            bgfx::setInstanceDataBuffer(scene.debugBuffers[path], batch.start, batch.count);
            bgfx::setState(state);
            bgfx::submit(pass, mDebugPrograms[path]);
            ++mStats.drawCalls;
        };
        size_t start = 0;
        while (start < frame.debug.size()) {
            size_t end = start + 1;
            while (end < frame.debug.size() &&
                   frame.debug[end].occlusion == frame.debug[start].occlusion &&
                   (frame.debug[end].path == DebugPath::Text) ==
                       (frame.debug[start].path == DebugPath::Text))
                ++end;
            for (size_t i = start; i < end; ++i)
                draw(frame.debug[i], false);
            if (frame.debug[start].occlusion == Occlusion::Ghost)
                for (size_t i = start; i < end; ++i)
                    draw(frame.debug[i], true);
            start = end;
        }
    }
    void updateWireMesh(GpuScene &scene, uint32_t index) {
        auto &gpu = *scene.meshes[index];
        const auto &mesh = *scene.source.meshes[index];
        const auto &vertices = gpu.deformed.empty() ? mesh.vertices : gpu.deformed;
        std::vector<WireVertex> expanded(mesh.indices.size());
        for (size_t i = 0; i < expanded.size(); ++i) {
            auto vertex = mesh.indices[i];
            expanded[i].vertex = {vertices[vertex], mesh.texcoords.empty()
                                                        ? std::array<float, 2>{0, 0}
                                                        : mesh.texcoords[vertex]};
            expanded[i].bary = {0, 0, 0};
            expanded[i].bary[i % 3] = 1;
        }
        const auto *memory = bgfx::copy(expanded.data(), expanded.size() * sizeof(WireVertex));
        if (bgfx::isValid(gpu.wireVertices))
            bgfx::update(gpu.wireVertices, 0, memory);
        else
            gpu.wireVertices = bgfx::createDynamicVertexBuffer(memory, mWireVertices);
        if (!bgfx::isValid(gpu.wireVertices))
            throw std::runtime_error("Cannot allocate wireframe geometry");
    }
    GpuTarget &target(Target id) {
        auto it = mTargets.find(id.id);
        if (it == mTargets.end())
            throw std::invalid_argument("Unknown render target");
        return it->second;
    }
    void extent(Extent size) const {
        if (!size.width || !size.height || size.width > mCaps.maxTextureSize ||
            size.height > mCaps.maxTextureSize)
            throw std::invalid_argument("Unsupported render target size");
    }
    bgfx::ShaderHandle shader(const std::string &file) {
        std::ifstream stream(file, std::ios::binary | std::ios::ate);
        if (!stream)
            throw std::runtime_error("Cannot open shader: " + file);
        auto length = stream.tellg();
        stream.seekg(0);
        if (length <= 0 || length > std::numeric_limits<uint32_t>::max() - 1)
            throw std::runtime_error("Invalid shader size");
        std::vector<char> bytes(static_cast<size_t>(length) + 1);
        if (!stream.read(bytes.data(), length))
            throw std::runtime_error("Cannot read shader: " + file);
        auto handle = bgfx::createShader(bgfx::copy(bytes.data(), uint32_t(bytes.size())));
        if (!bgfx::isValid(handle))
            throw std::runtime_error("Cannot create shader: " + file);
        return handle;
    }
    bgfx::ProgramHandle program(const std::string &dir, const char *vs, const char *fs) {
        auto v = shader(dir + "/" + vs + ".bin");
        bgfx::ShaderHandle f = BGFX_INVALID_HANDLE;
        try {
            f = shader(dir + "/" + fs + ".bin");
        } catch (...) {
            bgfx::destroy(v);
            throw;
        }
        auto result = bgfx::createProgram(v, f, true);
        if (!bgfx::isValid(result)) {
            bgfx::destroy(v);
            bgfx::destroy(f);
            throw std::runtime_error(std::string("Cannot link shader program: ") + fs);
        }
        return result;
    }
    void loadPrograms() {
        std::vector<std::pair<bgfx::ProgramHandle *, bgfx::ProgramHandle>> replacements;
        replacements.reserve(32);
        auto stage = [&](bgfx::ProgramHandle &destination, const char *vs, const char *fs) {
            replacements.emplace_back(&destination, program(mShaderDirectory, vs, fs));
        };
        try {
            const char *debugNames[] = {"Line",   "Arrow",    "Point",  "Stroke", "Solid",
                                        "Sector", "DragLink", "Screen", "Text"};
            for (size_t i = 0; i < mDebugPrograms.size(); ++i) {
                const auto name = std::string("debug") + debugNames[i];
                stage(mDebugPrograms[i], ("vs_" + name).c_str(), ("fs_" + name).c_str());
            }
            stage(mColorProgram, "vs_scene", "fs_color");
            const char *dataShaders[] = {"fs_data", "fs_id", "fs_segmentation", "fs_depth"};
            for (size_t i = 0; i < mDataPrograms.size(); ++i)
                stage(mDataPrograms[i], "vs_scene", dataShaders[i]);
            stage(mUiProgram, "vs_ui", "fs_ui");
            stage(mLitProgram, "vs_lit", "fs_lit");
            stage(mWireProgram, "vs_litWire", "fs_litWire");
            stage(mGizmoProgram, "vs_gizmo", "fs_gizmo");
            stage(mMaskProgram, "vs_shadow", "fs_mask");
            stage(mOutlineProgram, "vs_fullscreen", "fs_outline");
            stage(mIdentityProgram, "vs_fullscreen", "fs_identityColor");
            stage(mSkyProgram, "vs_sky", "fs_sky");
            stage(mClassicSkyProgram, "vs_skyClassic", "fs_sky");
            stage(mHazeProgram, "vs_haze", "fs_haze");
            stage(mShadowProgram, "vs_shadow", "fs_shadow");
            stage(mDistanceProgram, "vs_shadow", "fs_distance");
            if ((bgfx::getCaps()->supported & BGFX_CAPS_COMPUTE) &&
                std::endian::native == std::endian::little) {
                auto handle =
                    bgfx::createProgram(shader(mShaderDirectory + "/cs_readback.bin"), true);
                if (!bgfx::isValid(handle))
                    throw std::runtime_error("Cannot create readback conversion program");
                replacements.emplace_back(&mReadbackProgram, handle);
            }
        } catch (...) {
            for (auto [destination, handle] : replacements)
                bgfx::destroy(handle);
            throw;
        }
        // Publish only a complete program set. A compile/load failure leaves
        // every scene and UI consumer on its previous working programs.
        for (auto [destination, handle] : replacements) {
            if (bgfx::isValid(*destination))
                bgfx::destroy(*destination);
            *destination = handle;
        }
    }
    void releaseTarget(GpuTarget &t) {
        t.reflectionKey.reset();
        t.colorKey.reset();
        t.dataKey.reset();
        t.cachedData = 0;
        t.reflections.release();
        if (bgfx::isValid(t.selectionFrame))
            bgfx::destroy(t.selectionFrame);
        t.selectionFrame = BGFX_INVALID_HANDLE;
        t.selectionMask = BGFX_INVALID_HANDLE;
        if (bgfx::isValid(t.colorFb))
            bgfx::destroy(t.colorFb);
        // The complete framebuffer owns the shared attachments; destroy it last.
        for (size_t i = t.dataFbs.size(); i-- > 0;) {
            if (bgfx::isValid(t.dataFbs[i]))
                bgfx::destroy(t.dataFbs[i]);
            t.dataFbs[i] = BGFX_INVALID_HANDLE;
        }
        t.colorFb = BGFX_INVALID_HANDLE;
    }
    void allocateTarget(GpuTarget &t) {
        uint64_t flags = sampler;
        t.allocatedSamples = scene(t.scene).style.msaa ? t.samples : 1;
        uint64_t msaa = t.allocatedSamples == 2    ? BGFX_TEXTURE_RT_MSAA_X2
                        : t.allocatedSamples == 4  ? BGFX_TEXTURE_RT_MSAA_X4
                        : t.allocatedSamples == 8  ? BGFX_TEXTURE_RT_MSAA_X8
                        : t.allocatedSamples == 16 ? BGFX_TEXTURE_RT_MSAA_X16
                                                   : BGFX_TEXTURE_RT;
        // RT and RT_MSAA_X* encode one field, not independent bits. OR-ing RT
        // into MSAA_X2 requests X4, and OR-ing it into X8 requests X16.
        if (!bgfx::isTextureValid(0, false, 1, bgfx::TextureFormat::RGBA8, flags | msaa))
            throw std::runtime_error("Requested color/MSAA target unsupported");
        std::vector<bgfx::TextureHandle> orphaned;
        auto texture = [&](bgfx::TextureFormat::Enum format, uint64_t options) {
            auto h = bgfx::createTexture2D(t.size.width, t.size.height, false, 1, format, options);
            if (!bgfx::isValid(h))
                throw std::runtime_error("Cannot allocate render target texture");
            orphaned.push_back(h);
            return h;
        };
        try {
            t.color = texture(bgfx::TextureFormat::RGBA8, flags | msaa);
            auto colorDepth =
                texture(bgfx::TextureFormat::D32F, flags | msaa | BGFX_TEXTURE_RT_WRITE_ONLY);
            bgfx::TextureHandle color_attachments[] = {t.color, colorDepth};
            t.colorFb = bgfx::createFrameBuffer(2, color_attachments, true);
            if (!bgfx::isValid(t.colorFb))
                throw std::runtime_error("Cannot create color framebuffer");
            orphaned.clear();
            const bgfx::TextureFormat::Enum formats[] = {
                bgfx::TextureFormat::RGBA8, bgfx::TextureFormat::RGBA16, bgfx::TextureFormat::R32F};
            for (size_t i = 0; i < t.data.size(); ++i)
                t.data[i] = texture(formats[i], flags | BGFX_TEXTURE_RT);
            auto dataDepth = texture(bgfx::TextureFormat::D32F,
                                     flags | BGFX_TEXTURE_RT | BGFX_TEXTURE_RT_WRITE_ONLY);
            bgfx::TextureHandle attachments[] = {t.data[0], t.data[1], t.data[2], dataDepth};
            t.dataFbs[0] = bgfx::createFrameBuffer(4, attachments, true);
            if (!bgfx::isValid(t.colorFb) || !bgfx::isValid(t.dataFbs[0]))
                throw std::runtime_error("Cannot create framebuffer");
            orphaned.clear();
            // Export only requested attachments; retain the full set for combined requests.
            bgfx::TextureHandle idAttachments[] = {t.data[0], dataDepth};
            bgfx::TextureHandle segmentationAttachments[] = {t.data[1], dataDepth};
            bgfx::TextureHandle depthAttachments[] = {t.data[2], dataDepth};
            t.dataFbs[1] = bgfx::createFrameBuffer(2, idAttachments, false);
            t.dataFbs[2] = bgfx::createFrameBuffer(2, segmentationAttachments, false);
            t.dataFbs[3] = bgfx::createFrameBuffer(2, depthAttachments, false);
            for (auto framebuffer : t.dataFbs)
                if (!bgfx::isValid(framebuffer))
                    throw std::runtime_error("Cannot create data framebuffer");
        } catch (...) {
            for (auto texture : orphaned)
                bgfx::destroy(texture);
            releaseTarget(t);
            throw;
        }
    }
    void cancel(Target id = {}) {
        for (auto &request : mReadbacks)
            if (!id.id || request.frame.target == id)
                request.canceled = true;
    }
    void releaseSlot(ReadbackSlot &slot) {
        if (bgfx::isValid(slot.buffer))
            bgfx::destroy(slot.buffer);
        slot.buffer = BGFX_INVALID_HANDLE;
        if (bgfx::isValid(slot.resolved))
            bgfx::destroy(slot.resolved);
        slot.resolved = BGFX_INVALID_HANDLE;
    }
    bgfx::TextureHandle textureHandle(Texture texture) const {
        if (texture.id & targetTextureBit) {
            auto it = mTargets.find(texture.id & ~targetTextureBit);
            if (it == mTargets.end())
                throw std::invalid_argument("Expired target texture");
            return it->second.color;
        }
        auto it = mTextures.find(texture.id);
        if (it == mTextures.end())
            throw std::invalid_argument("Unknown UI texture");
        return it->second;
    }

  public:
    void initialize(const BgfxOptions &options) {
        mShaderDirectory = options.shaderDirectory;
        mWindowSystem = options.window.system;
        if (runtimeActive.test_and_set())
            throw std::logic_error("Only one bgfx runtime may exist; create peer targets instead");
        bgfx::Init init;
#if defined(__APPLE__)
        init.type = bgfx::RendererType::Metal;
#elif defined(_WIN32)
        init.type = bgfx::RendererType::Direct3D12;
#else
        init.type = bgfx::RendererType::Vulkan;
#endif
        init.fallback = false;
        init.platformData.type = mWindowSystem == WindowSystem::Wayland
                                     ? bgfx::NativeWindowHandleType::Wayland
                                     : bgfx::NativeWindowHandleType::Default;
        init.profile = true;
        init.reset = mResetFlags;
        // Overlap CPU submission with GPU execution without an unbounded frame queue.
        init.swapChain.maxFrameLatency = 2;
        init.swapChain.nwh = options.window.handle;
        init.swapChain.ndt = options.window.display;
        init.swapChain.width = options.window.handle ? options.window.size.width : 0;
        init.swapChain.height = options.window.handle ? options.window.size.height : 0;
        mSwapChain = init.swapChain;
        mWindowSize = options.window.size;
        mHasWindow = options.window.handle != nullptr;
        // The runtime already owns a render thread; avoid a second CPU frame queue.
        bgfx::renderFrame();
        if (!bgfx::init(init)) {
            runtimeActive.clear();
            throw std::runtime_error("Cannot initialize native renderer");
        }
        mInitialized = true;
        bgfx::setDebug(BGFX_DEBUG_PROFILER);
        const auto *caps = bgfx::getCaps();
        mCaps.backend = bgfx::getRendererName(caps->rendererType);
        mCaps.device = std::to_string(caps->vendorId) + ":" + std::to_string(caps->deviceId);
        mCaps.maxTextureSize = std::min<uint32_t>(caps->limits.maxTextureSize, UINT16_MAX);
        mCaps.readback = bgfx::isTextureValid(0, false, 1, bgfx::TextureFormat::RGBA8,
                                              BGFX_TEXTURE_READ_BACK | BGFX_TEXTURE_BLIT_DST);
        mCaps.instancing = caps->limits.maxInstanceData >= 8;
        mCaps.multipleScenes = true;
        mCaps.multipleWindows = (caps->supported & BGFX_CAPS_SWAP_CHAIN) != 0;
        auto rt = [&](bgfx::TextureFormat::Enum f) {
            return (caps->formats[f] & BGFX_CAPS_FORMAT_TEXTURE_FRAMEBUFFER) != 0;
        };
        mCaps.integerTarget = rt(bgfx::TextureFormat::R32U);
        mCaps.signedPairTarget = rt(bgfx::TextureFormat::RG32I);
        mCaps.floatTarget = rt(bgfx::TextureFormat::R32F);
        if (!mCaps.readback || !mCaps.instancing || !mCaps.floatTarget ||
            !rt(bgfx::TextureFormat::RGBA16) || caps->limits.maxFBAttachments < 4)
            throw std::runtime_error("Device lacks a required probe capability");
        mVertices.begin()
            .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::Normal, 3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
            .end();
        mWireVertices.begin()
            .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::Normal, 3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord1, 3, bgfx::AttribType::Float)
            .end();
        mUiVertices.begin()
            .add(bgfx::Attrib::Position, 2, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
            .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
            .end();
        mDebugPrograms.fill(bgfx::ProgramHandle{bgfx::kInvalidHandle});
        for (size_t i = 0; i < 9; ++i) {
            mDebugLayouts[i].begin();
            for (uint32_t col = 0; col < (debugRecordFloats[i] + 3) / 4; ++col)
                mDebugLayouts[i].add(bgfx::Attrib::Enum(bgfx::Attrib::TexCoord0 + col), 4,
                                     bgfx::AttribType::Float);
            mDebugLayouts[i].end();
        }
        std::array<GpuVertex, 96> debugVertices{};
        for (size_t i = 0; i < debugVertices.size(); ++i)
            debugVertices[i].vertex.position[0] = float(i);
        mDebugVertices = bgfx::createVertexBuffer(
            bgfx::copy(debugVertices.data(), sizeof(debugVertices)), mVertices);
        mDebugViewProj = bgfx::createUniform("u_debugViewProj", bgfx::UniformType::Mat4);
        mDebugProj = bgfx::createUniform("u_debugProj", bgfx::UniformType::Mat4);
        mDebugParams = bgfx::createUniform("u_debugParams", bgfx::UniformType::Vec4);
        mDebugDepth = bgfx::createUniform("u_debugDepth", bgfx::UniformType::Vec4);
        mDebugAtlas = bgfx::createUniform("s_debugAtlas", bgfx::UniformType::Sampler);
        mUiTextureInfo = bgfx::createUniform("u_uiTextureInfo", bgfx::UniformType::Vec4);
        mSurfaceLayout.begin();
        for (int i = 0; i < 8; ++i)
            mSurfaceLayout.add(bgfx::Attrib::Enum(bgfx::Attrib::TexCoord0 + i), 4,
                               bgfx::AttribType::Float);
        mSurfaceLayout.end();
        mGizmoColor = bgfx::createUniform("u_gizmoColor", bgfx::UniformType::Vec4);
        mGizmoParams = bgfx::createUniform("u_gizmoParams", bgfx::UniformType::Vec4);
        if ((bgfx::getCaps()->supported & BGFX_CAPS_COMPUTE) &&
            std::endian::native == std::endian::little) {
            mReadbackImage = bgfx::createUniform("s_readbackImage", bgfx::UniformType::Sampler);
            mReadbackRegion = bgfx::createUniform("u_readbackRegion", bgfx::UniformType::Vec4);
            mReadbackLayout = bgfx::createUniform("u_readbackLayout", bgfx::UniformType::Vec4);
        }
        mIdentitySampler = bgfx::createUniform("s_identity", bgfx::UniformType::Sampler);
        mIdentitySize = bgfx::createUniform("u_identitySize", bgfx::UniformType::Vec4);
        mMaskSampler = bgfx::createUniform("s_selectionMask", bgfx::UniformType::Sampler);
        mOutlineSize = bgfx::createUniform("u_outlineSize", bgfx::UniformType::Vec4);
        mOutlineColor = bgfx::createUniform("u_outlineColor", bgfx::UniformType::Vec4);
        mReflections.initialize();
        mEnvironment.initialize();
        mSkySampler = bgfx::createUniform("s_sky", bgfx::UniformType::Sampler);
        mSkyInverse = bgfx::createUniform("u_skyInverse", bgfx::UniformType::Mat4);
        mSkyEyeDistance = bgfx::createUniform("u_skyEyeDistance", bgfx::UniformType::Vec4);
        const char *hazeNames[] = {"u_hazeGeometry", "u_hazeBasisX", "u_hazeBasisY",
                                   "u_hazeNormal"};
        for (int i = 0; i < 4; ++i)
            mHazeUniforms[i] = bgfx::createUniform(hazeNames[i], bgfx::UniformType::Vec4);
        mLighting.initialize();
        mShadows.initialize();
        mShadowLightPosition =
            bgfx::createUniform("u_shadowLightPosition", bgfx::UniformType::Vec4);
        mCubeSampler = bgfx::createUniform("s_cube", bgfx::UniformType::Sampler);
        mImageLightSampler = bgfx::createUniform("s_imageLight", bgfx::UniformType::Sampler);
        const std::array<uint32_t, 6> cubeWhite = {0xffffffff, 0xffffffff, 0xffffffff,
                                                   0xffffffff, 0xffffffff, 0xffffffff};
        mWhiteCube = bgfx::createTextureCube(1, false, 1, bgfx::TextureFormat::RGBA8, 0,
                                             bgfx::copy(cubeWhite.data(), sizeof(cubeWhite)));
        mImageSampler = bgfx::createUniform("s_image", bgfx::UniformType::Sampler);
        mMaterial = bgfx::createUniform("u_material", bgfx::UniformType::Vec4);
        const uint32_t white = 0xffffffff;
        mWhite = bgfx::createTexture2D(1, 1, false, 1, bgfx::TextureFormat::RGBA8, 0,
                                       bgfx::copy(&white, 4));
        bgfx::setPaletteColor(0, uint32_t{0});
        bgfx::setPaletteColor(1, 0xffffffff);
        loadPrograms();
    }
    ~BgfxRenderer() override {
        if (!mInitialized)
            return;
        // Resource uploads can open a Metal blit encoder before the first render.
        // Finish that frame before queuing destruction, including failed startup.
        flush();
        // Pending readback destinations must survive until the render thread has stopped.
        for (auto &[id, t] : mTargets)
            releaseTarget(t);
        for (auto &[id, value] : mScenes)
            releaseScene(value);
        for (auto &r : mReadbacks)
            releaseSlot(r);
        for (auto &[id, t] : mTextures)
            bgfx::destroy(t);
        for (auto h : {mMaskProgram, mOutlineProgram, mIdentityProgram, mReadbackProgram})
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        for (auto h : {mMaskSampler, mOutlineSize, mOutlineColor})
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        mInstances.clear();
        for (auto h : {mIdentitySampler, mIdentitySize, mReadbackImage, mReadbackRegion,
                       mReadbackLayout, mUiTextureInfo})
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        if (bgfx::isValid(mGizmoProgram))
            bgfx::destroy(mGizmoProgram);
        for (auto h : {mGizmoColor, mGizmoParams})
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        mReflections.release();
        for (auto h : mDebugPrograms)
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        if (bgfx::isValid(mDebugVertices))
            bgfx::destroy(mDebugVertices);
        for (auto h : {mDebugViewProj, mDebugProj, mDebugParams, mDebugDepth, mDebugAtlas})
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        mEnvironment.release();
        for (auto h : {mSkyProgram, mClassicSkyProgram, mHazeProgram})
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        for (auto h : {mSkySampler, mSkyInverse, mSkyEyeDistance})
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        for (auto h : mHazeUniforms)
            if (bgfx::isValid(h))
                bgfx::destroy(h);
        mLighting.release();
        mShadows.release();
        for (auto handle : {mShadowProgram, mDistanceProgram})
            if (bgfx::isValid(handle))
                bgfx::destroy(handle);
        if (bgfx::isValid(mShadowLightPosition))
            bgfx::destroy(mShadowLightPosition);
        if (bgfx::isValid(mWireProgram))
            bgfx::destroy(mWireProgram);
        if (bgfx::isValid(mLitProgram))
            bgfx::destroy(mLitProgram);
        if (bgfx::isValid(mCubeSampler))
            bgfx::destroy(mCubeSampler);
        if (bgfx::isValid(mImageLightSampler))
            bgfx::destroy(mImageLightSampler);
        if (bgfx::isValid(mWhiteCube))
            bgfx::destroy(mWhiteCube);
        if (bgfx::isValid(mColorProgram))
            bgfx::destroy(mColorProgram);
        for (auto program : mDataPrograms)
            if (bgfx::isValid(program))
                bgfx::destroy(program);
        if (bgfx::isValid(mUiProgram))
            bgfx::destroy(mUiProgram);
        if (bgfx::isValid(mImageSampler))
            bgfx::destroy(mImageSampler);
        if (bgfx::isValid(mMaterial))
            bgfx::destroy(mMaterial);
        if (bgfx::isValid(mWhite))
            bgfx::destroy(mWhite);
        flush();
        bgfx::shutdown();
        runtimeActive.clear();
    }
    const Capabilities &capabilities() const override {
        return mCaps;
    }
    void setOverlays(Scene id, OverlayFrame frame) override {
        owner();
        auto &current = scene(id);
        for (const auto &row : frame.surfaces)
            for (float v : row)
                if (!std::isfinite(v))
                    throw std::invalid_argument("Non-finite surface data");
        for (const auto &batch : frame.surfaceBatches) {
            if (batch.mesh >= current.meshes.size() || batch.start > frame.surfaces.size() ||
                batch.count > frame.surfaces.size() - batch.start || batch.texture < -1 ||
                batch.texture >= int(current.textures.size()))
                throw std::invalid_argument("Invalid surface batch");
        }
        for (const auto &draw : frame.gizmos) {
            if (!std::isfinite(draw.maskRadius) || draw.maskRadius < 0)
                throw std::invalid_argument("Invalid gizmo mask radius");
            if (draw.mesh >= current.meshes.size())
                throw std::invalid_argument("Invalid gizmo mesh");
            for (float value : draw.transform)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Non-finite gizmo transform");
            for (float value : draw.color)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Non-finite gizmo color");
        }
        for (size_t path = 0; path < frame.streams.size(); ++path) {
            const auto &stream = frame.streams[path];
            if (stream.size() % debugRecordFloats[path])
                throw std::invalid_argument("Invalid debug stream length");
            for (float value : stream)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Non-finite debug data");
        }
        for (const auto &b : frame.debug) {
            size_t path = size_t(b.path);
            if (path >= frame.streams.size() || int(b.occlusion) < 0 || int(b.occlusion) > 2)
                throw std::invalid_argument("Invalid debug batch type");
            size_t count = frame.streams[path].size() / debugRecordFloats[path];
            if (b.start > count || b.count > count - b.start ||
                (b.path == DebugPath::Solid && b.mesh >= current.meshes.size()))
                throw std::invalid_argument("Debug batch exceeds stream");
            if (b.path == DebugPath::Text && !mTextures.contains(frame.glyphAtlas.id))
                throw std::invalid_argument("Missing glyph atlas");
        }
        if (mPendingCommands)
            flush();
        for (size_t path = 0; path < frame.streams.size(); ++path) {
            auto &handle = current.debugBuffers[path];
            auto floats = debugRecordFloats[path], stride = (floats + 3) / 4 * 4;
            uint32_t count = frame.streams[path].size() / floats;
            if (!count)
                continue;
            if (!bgfx::isValid(handle))
                handle = bgfx::createDynamicVertexBuffer(std::max(count, 64u), mDebugLayouts[path],
                                                         BGFX_BUFFER_ALLOW_RESIZE);
            mDebugUpload.resize(size_t(count) * stride);
            for (size_t i = 0; i < count; ++i)
                std::memcpy(mDebugUpload.data() + i * stride,
                            frame.streams[path].data() + i * floats, floats * sizeof(float));
            bgfx::update(handle, 0,
                         bgfx::copy(mDebugUpload.data(), mDebugUpload.size() * sizeof(float)));
            mStats.uploadBytes += mDebugUpload.size() * sizeof(float);
        }
        if (!frame.surfaces.empty()) {
            if (!bgfx::isValid(current.surfaceBuffer))
                current.surfaceBuffer =
                    bgfx::createDynamicVertexBuffer(std::max(size_t(64), frame.surfaces.size()),
                                                    mSurfaceLayout, BGFX_BUFFER_ALLOW_RESIZE);
            bgfx::update(current.surfaceBuffer, 0,
                         bgfx::copy(frame.surfaces.data(),
                                    frame.surfaces.size() * sizeof(frame.surfaces[0])));
        }
        current.overlays = std::move(frame);
    }
    void setLighting(Scene id, const Lighting &lighting) override {
        owner();
        if (lighting.lights.size() > 100)
            throw std::invalid_argument("Scene supports at most 100 active lights");
        const auto &source = scene(id).source;
        for (int index : {lighting.imageTexture, lighting.skyboxTexture})
            if (index < -1 || index >= int(source.textures.size()) ||
                (index >= 0 && !source.textures[index].cube))
                throw std::invalid_argument("Environment requires a scene cube texture");
        auto finite = [](const auto &values) {
            for (float value : values)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Non-finite light data");
        };
        for (const auto &values :
             {lighting.ambient, lighting.headlightSpecular, lighting.fogColor, lighting.hazeColor})
            finite(values);
        finite(lighting.headlightDiffuse);
        finite(lighting.fog);
        if (!std::isfinite(lighting.imageIntensity) || lighting.imageIntensity < 0 ||
            !std::isfinite(lighting.hazeDensity) || lighting.hazeDensity < 0 ||
            lighting.hazeSlices < 3 || lighting.hazeSlices > 4096)
            throw std::invalid_argument("Invalid environment lighting");
        for (const auto &light : lighting.lights) {
            for (const auto &values : {light.position, light.direction, light.diffuse,
                                       light.specular, light.attenuation})
                finite(values);
            finite(std::array{light.cutoff, light.exponent, light.range, light.radius});
            if (light.type < 0 || light.type > 3 || light.range < 0 || light.radius < 0 ||
                light.exponent < 0)
                throw std::invalid_argument("Invalid light parameters");
        }
        auto &current = scene(id);
        if (current.lighting != lighting) {
            current.lighting = lighting;
            ++current.lightingRevision;
        }
    }
    void configure(Scene id, const SceneStyle &style) override {
        owner();
        if (style.debugView < 0 || style.debugView > 7 || style.shadowQuality < 0 ||
            style.shadowQuality > 2)
            throw std::invalid_argument("Invalid render mode");
        for (float value : style.background)
            if (!std::isfinite(value) || value < 0 || value > 1)
                throw std::invalid_argument("Background must contain normalized finite channels");

        auto &current = scene(id);
        if (current.style != style) {
            current.style = style;
            ++current.styleRevision;
        }
    }
    ResourceStats resourceStats() const override {
        owner();
        return mResources;
    }
    std::shared_ptr<GpuMesh> uploadMesh(std::shared_ptr<const Mesh> input, bool cache = true) {
        if (cache) {
            auto found = mMeshCache.find(input.get());
            if (found != mMeshCache.end())
                if (auto existing = found->second.lock())
                    return existing;
        }
        validateMesh(*input);
        auto mesh = std::make_shared<GpuMesh>();
        std::vector<GpuVertex> vertices(input->vertices.size());
        for (size_t v = 0; v < vertices.size(); ++v)
            vertices[v] = {input->vertices[v], input->texcoords.empty() ? std::array<float, 2>{0, 0}
                                                                        : input->texcoords[v]};
        mesh->vertices = bgfx::createDynamicVertexBuffer(
            bgfx::copy(vertices.data(), vertices.size() * sizeof(GpuVertex)), mVertices);
        mesh->indices = bgfx::createIndexBuffer(
            bgfx::copy(input->indices.data(), input->indices.size() * sizeof(uint32_t)),
            BGFX_BUFFER_INDEX32);
        if (!bgfx::isValid(mesh->vertices) || !bgfx::isValid(mesh->indices))
            throw std::runtime_error("Cannot allocate scene mesh");
        mesh->vertexCount = input->vertices.size();
        mesh->bounds = meshBounds(input->vertices);
        mesh->source = std::move(input);
        ++mResources.meshUploads;
        mResources.uploadBytes +=
            vertices.size() * sizeof(GpuVertex) + mesh->source->indices.size() * sizeof(uint32_t);
        if (cache)
            mMeshCache[mesh->source.get()] = mesh;
        return mesh;
    }
    std::shared_ptr<GpuTexture> uploadSceneTexture(const TextureSource &texture) {
        const TextureKey key{texture.rgba.get(), texture.size.width, texture.size.height,
                             texture.mipmaps,    texture.srgb,       texture.cube};
        auto found = mTextureCache.find(key);
        if (found != mTextureCache.end() && found->second.pixels.lock() == texture.rgba)
            if (auto existing = found->second.gpu.lock())
                return existing;
        extent(texture.size);
        const auto flags =
            (texture.srgb ? BGFX_TEXTURE_SRGB : 0) |
            (texture.cube
                 ? uint64_t(BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP | BGFX_SAMPLER_W_CLAMP)
                 : uint64_t(0));
        auto *storage = new decltype(texture.rgba)(texture.rgba);
        auto memory = bgfx::makeRef(
            (*storage)->data(), (*storage)->size(),
            [](void *, void *user) { delete static_cast<decltype(storage)>(user); }, storage);
        auto result = std::make_shared<GpuTexture>();
        result->handle =
            texture.cube
                ? bgfx::createTextureCube(texture.size.width, texture.mipmaps, 1,
                                          bgfx::TextureFormat::RGBA8, flags, memory)
                : bgfx::createTexture2D(texture.size.width, texture.size.height, texture.mipmaps, 1,
                                        bgfx::TextureFormat::RGBA8, flags, memory);
        if (!bgfx::isValid(result->handle))
            throw std::runtime_error("Cannot allocate scene texture");
        ++mResources.textureUploads;
        mResources.uploadBytes += texture.rgba->size();
        mTextureCache[key] = {texture.rgba, result};
        return result;
    }
    GpuScene uploadScene(const SceneSource &source) {
        validateScene(source, false);
        std::erase_if(mMeshCache, [](const auto &item) { return item.second.expired(); });
        std::erase_if(mTextureCache, [](const auto &item) { return item.second.gpu.expired(); });
        GpuScene result;
        // Keep shared immutable geometry for deformation and wire expansion.
        // Texture pixels are upload inputs; retain only their sampling metadata.
        result.source = {.revision = source.revision,
                         .meshes = source.meshes,
                         .instances = source.instances,
                         .materials = source.materials,
                         .textures = {},
                         .materialIndices = source.materialIndices,
                         .infinitePlanes = source.infinitePlanes,
                         .planarKinds = source.planarKinds,
                         .linearColors = source.linearColors,
                         .visualMaterials = source.visualMaterials,
                         .cubeCoords = source.cubeCoords,
                         .extent = source.extent,
                         .shadowClip = source.shadowClip,
                         .center = source.center};
        for (const auto &texture : source.textures)
            result.source.textures.push_back(
                {texture.size, texture.mipmaps, texture.srgb, texture.cube, {}});
        result.instances.resize(source.instances.size());
        result.worldBounds.resize(source.instances.size());
        result.meshes.resize(source.meshes.size());
        try {
            for (size_t i = 0; i < source.meshes.size(); ++i)
                result.meshes[i] = uploadMesh(source.meshes[i]);
            for (const auto &texture : source.textures) {
                auto owner = uploadSceneTexture(texture);
                result.textures.push_back(owner->handle);
                result.textureOwners.push_back(std::move(owner));
            }
            std::unordered_map<uint64_t, size_t> batches;
            for (uint32_t i = 0; i < source.instances.size(); ++i) {
                const auto &input = source.instances[i];
                const uint32_t material =
                    source.materialIndices.empty() ? 0 : source.materialIndices[i];
                const uint64_t key = (uint64_t(input.mesh) << 32) | material;
                auto [it, fresh] = batches.emplace(key, result.batches.size());
                if (fresh)
                    result.batches.push_back({input.mesh, material, {}});
                result.batches[it->second].instances.push_back(i);
                auto &data = result.instances[i];
                const auto matrix = identity();
                std::copy(matrix.begin(), matrix.end(), data.begin());
                result.worldBounds[i] = result.meshes[input.mesh]->bounds;
                std::copy(input.color.begin(), input.color.end(), data.begin() + 16);
                data[28] = data[29] = 1;
                auto visual =
                    source.materials.empty() ? Material{-1, 0, 0, 0} : source.materials[material];
                data[32] = visual.emission;
                data[33] = visual.specular;
                data[34] = visual.shininess;
                if (!source.visualMaterials.empty())
                    std::copy(source.visualMaterials[i].begin(), source.visualMaterials[i].end(),
                              data.begin() + 32);
                if (!source.cubeCoords.empty())
                    std::copy(source.cubeCoords[i].begin(), source.cubeCoords[i].end(),
                              data.begin() + 36);
                auto encode = [&](size_t offset, uint32_t value) {
                    data[offset] = float(value & 0xffff);
                    data[offset + 1] = float(value >> 16);
                };
                encode(20, input.objectId);
                encode(22, std::bit_cast<uint32_t>(input.segmentation[0]));
                encode(24, std::bit_cast<uint32_t>(input.segmentation[1]));
            }
        } catch (...) {
            releaseScene(result);
            throw;
        }
        return result;
    }
    Scene createScene(const SceneSource &source) override {
        owner();
        Scene id{allocateId()};
        auto [it, inserted] = mScenes.try_emplace(id.id);
        try {
            it->second = uploadScene(source);
        } catch (...) {
            mScenes.erase(it);
            throw;
        }
        return id;
    }
    void setScene(const SceneSource &source) override {
        setScene(Scene{}, source);
    }
    void setScene(Scene id, const SceneSource &source) override {
        owner();
        auto &current = scene(id);
        auto replacement = uploadScene(source);
        replacement.style = current.style;
        replacement.lighting = current.lighting;
        // Environment indices are local to the old source texture table.
        replacement.lighting.skyboxTexture = -1;
        replacement.lighting.imageTexture = -1;
        if (mPendingCommands)
            flush();
        for (auto &[targetId, target] : mTargets) {
            if (target.scene == id) {
                cancel(Target{targetId});
                target.latest = {};
                target.reflectionKey.reset();
                target.colorKey.reset();
                target.dataKey.reset();
                target.cachedData = 0;
            }
        }
        releaseScene(current);
        current = std::move(replacement);
    }
    void destroy(Scene id) override {
        owner();
        if (!id.id)
            throw std::invalid_argument("Cannot destroy the default scene");
        auto &current = scene(id);
        std::vector<Target> attached;
        for (const auto &[targetId, target] : mTargets)
            if (target.scene == id)
                attached.push_back({targetId});
        for (auto target : attached)
            destroy(target);
        if (mPendingCommands)
            flush();
        releaseScene(current);
        mScenes.erase(id.id);
    }
    void update(const SceneFrame &frame) override {
        update(Scene{}, frame);
    }
    void update(Scene id, const SceneFrame &frame) override {
        owner();
        auto &current = scene(id);
        validateFrame(current.source, frame);
        current.sequence = frame.sequence;
        bool changed = false;
        for (size_t i = 0; i < current.instances.size(); ++i) {
            auto previous = current.instances[i];
            std::copy(frame.transforms[i].begin(), frame.transforms[i].end(),
                      current.instances[i].begin());
            if (!frame.colors.empty()) {
                current.source.instances[i].color = frame.colors[i];
                std::copy(frame.colors[i].begin(), frame.colors[i].end(),
                          current.instances[i].begin() + 16);
            }
            if (!frame.materials.empty())
                std::copy(frame.materials[i].begin(), frame.materials[i].end(),
                          current.instances[i].begin() + 32);
            if (!frame.cubeCoords.empty())
                std::copy(frame.cubeCoords[i].begin(), frame.cubeCoords[i].end(),
                          current.instances[i].begin() + 36);
            if (!frame.texcoords.empty())
                std::copy(frame.texcoords[i].begin(), frame.texcoords[i].end(),
                          current.instances[i].begin() + 28);
            changed |= previous != current.instances[i];
            current.worldBounds[i] =
                transformBounds(current.meshes[current.source.instances[i].mesh]->bounds,
                                current.instances[i].data());
        }
        if (changed)
            ++current.geometryRevision;
    }
    void updateMesh(uint32_t index, std::span<const Vertex> vertices) override {
        updateMesh(Scene{}, index, vertices);
    }
    void updateMesh(Scene id, uint32_t index, std::span<const Vertex> vertices) override {
        owner();
        auto &current = scene(id);
        if (index >= current.meshes.size() || vertices.size() != current.meshes[index]->vertexCount)
            throw std::invalid_argument("Dynamic mesh topology changed");
        if (mPendingCommands)
            flush();
        for (const auto &vertex : vertices)
            for (const auto &values : {vertex.position, vertex.normal})
                for (float value : values)
                    if (!std::isfinite(value))
                        throw std::invalid_argument("Non-finite dynamic mesh vertex");
        ++current.geometryRevision;
        // Dynamic vertices are scene-local. An immutable GPU mesh may also be
        // used by a peer scene, so detach before its first deformation.
        auto &mesh = current.meshes[index];
        if (mesh->deformed.empty()) {
            if (mesh.use_count() > 1)
                mesh = uploadMesh(current.source.meshes[index], false);
            else
                mMeshCache.erase(mesh->source.get());
        }
        mesh->deformed.assign(vertices.begin(), vertices.end());
        current.meshes[index]->bounds = meshBounds(vertices);
        for (size_t i = 0; i < current.instances.size(); ++i)
            if (current.source.instances[i].mesh == index)
                current.worldBounds[i] =
                    transformBounds(current.meshes[index]->bounds, current.instances[i].data());
        if (bgfx::isValid(current.meshes[index]->wireVertices))
            updateWireMesh(current, index);
        std::vector<GpuVertex> upload(vertices.size());
        const auto &uv = current.source.meshes[index]->texcoords;
        for (size_t i = 0; i < vertices.size(); ++i)
            upload[i] = {vertices[i], uv.empty() ? std::array<float, 2>{0, 0} : uv[i]};
        bgfx::update(current.meshes[index]->vertices, 0,
                     bgfx::copy(upload.data(), upload.size() * sizeof(GpuVertex)));
        mStats.uploadBytes += upload.size() * sizeof(GpuVertex);
    }
    Target createTarget(Extent size, uint32_t samples) override {
        return createTarget(Scene{}, size, samples);
    }
    Target createTarget(Scene sceneId, Extent size, uint32_t samples) override {
        owner();
        scene(sceneId);
        extent(size);
        if (samples != 1 && samples != 2 && samples != 4 && samples != 8 && samples != 16)
            throw std::invalid_argument("Sample count must be 1, 2, 4, 8, or 16");
        GpuTarget t;
        t.size = size;
        t.samples = samples;
        t.scene = sceneId;
        allocateTarget(t);
        Target id{allocateId()};
        mTargets.emplace(id.id, std::move(t));
        return id;
    }
    Target createSurface(NativeWindow window) override {
        owner();
        if (window.system != mWindowSystem)
            throw std::invalid_argument("Native windows must use the runtime's window system");
        extent(window.size);
        if (!window.handle || !mCaps.multipleWindows)
            throw std::invalid_argument("Native surface unavailable");
        GpuTarget t;
        t.size = window.size;
        t.surface = true;
        auto &surface = t.swapChain;
        surface.nwh = window.handle;
        surface.ndt = window.display;
        surface.width = window.size.width;
        surface.height = window.size.height;
        surface.maxFrameLatency = 2;
        t.colorFb = bgfx::createFrameBuffer(surface);
        if (!bgfx::isValid(t.colorFb))
            throw std::runtime_error("Cannot create native surface");
        Target id{allocateId()};
        mTargets.emplace(id.id, std::move(t));
        return id;
    }
    void setVsync(Target id, bool enabled) override {
        owner();
        auto &t = target(id);
        if (!t.surface)
            throw std::invalid_argument("VSync requires a native surface");
        t.vsync = enabled;
        updateVsync();
    }
    void updateVsync() {
        // bgfx owns one presentation policy per device. A synchronized peer
        // keeps all shared swap chains synchronized until its request is removed.
        bool enabled = std::any_of(mTargets.begin(), mTargets.end(), [](const auto &entry) {
            return entry.second.surface && entry.second.vsync;
        });
        uint32_t flags = BGFX_RESET_MAXANISOTROPY | BGFX_RESET_FLIP_AFTER_RENDER |
                         (enabled ? BGFX_RESET_VSYNC : 0);
        if (flags != mResetFlags) {
            if (mPendingCommands)
                flush();
            mResetFlags = flags;
            bgfx::reset(mResetFlags, &mSwapChain);
        }
#if defined(__APPLE__)
        // An unsynchronized Metal drawable may remain owned by the compositor
        // after GPU completion. A spare image prevents that ownership from
        // serializing submission; GPU work is still bounded to two frames.
        for (auto &[id, target] : mTargets) {
            const uint8_t imageCount = enabled ? 2 : 3;
            if (target.surface && target.swapChain.maxFrameLatency != imageCount) {
                target.swapChain.maxFrameLatency = imageCount;
                bgfx::updateSwapChain(target.colorFb, target.swapChain);
            }
        }
#endif
    }
    void resize(Target id, Extent size) override {
        owner();
        extent(size);
        auto &t = target(id);
        if (t.size == size)
            return;
        if (mPendingCommands)
            flush();
        if (t.surface) {
            t.swapChain.width = size.width;
            t.swapChain.height = size.height;
            bgfx::updateSwapChain(t.colorFb, t.swapChain);
            t.size = size;
            ++t.generation;
            t.latest = {};
            return;
        }
        GpuTarget replacement;
        replacement.scene = t.scene;
        replacement.view = t.view;
        replacement.samples = t.samples;
        replacement.size = size;
        replacement.generation = t.generation + 1;
        allocateTarget(replacement);
        cancel(id);
        releaseTarget(t);
        t = std::move(replacement);
    }
    void destroy(Target id) override {
        owner();
        auto &t = target(id);
        if (mPendingCommands)
            flush();
        cancel(id);
        releaseTarget(t);
        mTiming.erase(id);
        mTargets.erase(id.id);
        updateVsync();
    }
    FrameToken render(Target id, const CameraView &camera) override {
        return renderRequested(id, camera, {});
    }
    FrameToken renderRequested(Target id, const CameraView &camera,
                               RenderRequest request) override {
        size_t dataIndex = 0;
        if (request.dataProduct)
            switch (*request.dataProduct) {
            case Product::ObjectId:
                dataIndex = 1;
                break;
            case Product::Segmentation:
                dataIndex = 2;
                break;
            case Product::MetricDepth:
                dataIndex = 3;
                break;
            default:
                throw std::invalid_argument("Invalid scene data product");
            }
        const auto before = mStats;
        if (!request.color && !request.sceneData)
            throw std::invalid_argument("Empty render request");
        owner();
        validateCamera(camera);
        auto &t = target(id);
        auto &current = scene(t.scene);
        const bool identityColor = request.color && current.style.debugView >= 6;
        if (identityColor) {
            request.sceneData = true;
            request.dataProduct.reset();
            dataIndex = 0;
        }
        if (t.surface)
            throw std::invalid_argument(
                "Render scenes to an offscreen target, then present its texture");
        if (request.color && t.allocatedSamples != (current.style.msaa ? t.samples : 1)) {
            // Metal/Vulkan derive raster sample count from the attachments; the
            // draw-state MSAA bit alone cannot disable multisample coverage.
            if (mPendingCommands)
                flush();
            GpuTarget replacement;
            replacement.scene = t.scene;
            replacement.view = t.view;
            replacement.samples = t.samples;
            replacement.size = t.size;
            replacement.generation = t.generation + 1;
            allocateTarget(replacement);
            releaseTarget(t);
            t = std::move(replacement);
        }
        if (mNextSceneView + 59 >= 250)
            flush();
        beginTarget(t);
        mTimingTarget = id;
        mTimingSubmission = mSubmission + 1;
        ReflectionKey reflectionKey{current.geometryRevision, current.lightingRevision,
                                    current.styleRevision,    camera.view,
                                    camera.projection,        camera.nearPlane,
                                    camera.farPlane,          camera.focus};
        bool reflectionDirty = t.reflectionKey != reflectionKey;
        // Retain static scene color only when no separately updated overlays are present.
        const bool cacheColor = current.overlays.surfaceBatches.empty() &&
                                current.overlays.gizmos.empty() && current.overlays.debug.empty();
        const bool renderColor = request.color && (!cacheColor || t.colorKey != reflectionKey);
        const DataKey dataKey{current.geometryRevision, current.styleRevision, camera.view,
                              camera.projection,        camera.nearPlane,      camera.farPlane};
        if (t.dataKey != dataKey)
            t.cachedData = 0;
        const uint8_t dataMask = dataIndex ? 1u << (dataIndex - 1) : 7;
        const bool renderData = request.sceneData && (t.cachedData & dataMask) != dataMask;
        bool shadowRendered = false;
        if (renderColor && !identityColor) {
            mLighting.prepare(current.lighting);
            shadowRendered = renderShadows(current, camera);
            if (reflectionDirty)
                t.reflections.prepare(current.source, current.instances, current.style, camera,
                                      t.size);
        }
        t.colorOnly = !request.sceneData;
        t.dataOnly = !request.color;
        t.dataProduct = request.dataProduct;
        float clear_depth[4] = {camera.farPlane, 0, 0, 1};
        bgfx::setPaletteColor(2 + t.view / 4, clear_depth);
        auto view = columnMajor(camera.view), projection = camera.projection;
        if (!bgfx::getCaps()->homogeneousDepth)
            for (size_t c = 0; c < 4; ++c)
                projection[8 + c] = (projection[8 + c] + projection[12 + c]) * 0.5f;
        projection = columnMajor(projection);
        for (uint16_t pass = 0; pass < 2; ++pass) {
            if (pass ? !renderData : (!renderColor || identityColor))
                continue;
            timePass(t.view + pass, pass ? RenderPass::SceneData : RenderPass::Color);
            bgfx::setViewRect(t.view + pass, 0, 0, t.size.width, t.size.height);
            bgfx::setViewFrameBuffer(t.view + pass, pass ? t.dataFbs[dataIndex] : t.colorFb);
            bgfx::setViewTransform(t.view + pass, view.data(), projection.data());
            if (pass && t.dataProduct == Product::MetricDepth)
                bgfx::setViewClear(t.view + pass, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 1.0f, 0,
                                   2 + t.view / 4);
            else if (pass && t.dataProduct)
                bgfx::setViewClear(t.view + pass, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH,
                                   *t.dataProduct == Product::Segmentation ? UINT32_MAX : 0, 1.0f);
            else if (pass)
                bgfx::setViewClear(t.view + pass, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 1.0f, 0, 0,
                                   1, 2 + t.view / 4);
            else
                bgfx::setViewClear(
                    t.view + pass, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH,
                    current.style.debugView == 4
                        ? uint32_t{0x000000ff}
                        : (uint32_t(std::lround(current.style.background[0] * 255)) << 24) |
                              (uint32_t(std::lround(current.style.background[1] * 255)) << 16) |
                              (uint32_t(std::lround(current.style.background[2] * 255)) << 8) |
                              uint32_t(std::lround(current.style.background[3] * 255)),
                    1.0f);
            bgfx::touch(t.view + pass);
        }
        bgfx::setViewMode(t.view, bgfx::ViewMode::Sequential);
        auto drawCamera = camera;
        ClipFrustum drawFrustum(camera);
        auto visible = [&](uint32_t index) {
            if (drawFrustum.intersects(current.worldBounds[index]))
                return true;
            ++mStats.culledInstances;
            return false;
        };
        int reflectionLayer = -1;
        auto colorView = t.view;
        auto draw = [&](uint32_t meshIndex, uint32_t materialIndex,
                        std::span<const uint32_t> indices, uint16_t pass, bool transparent) {
            uint32_t count = indices.size();
            if (!count)
                return;
            bool wire = !pass && (current.style.wireframe || current.style.debugView == 5);
            bool lit = !pass && (current.lighting.enabled || wire);
            uint16_t stride = (lit ? 32 : 20) * sizeof(float);
            auto *buffer = mInstances.allocate(count, stride).data();
            for (size_t i = 0; i < count; ++i) {
                auto *destination = buffer + i * stride;
                const auto &source = current.instances[indices[i]];
                std::memcpy(destination, source.data(), 12 * sizeof(float));
                if (pass)
                    std::memcpy(destination + 12 * sizeof(float), source.data() + 20,
                                8 * sizeof(float));
                else {
                    std::memcpy(destination + 12 * sizeof(float), source.data() + 16,
                                4 * sizeof(float));
                    std::memcpy(destination + 16 * sizeof(float), source.data() + 28,
                                (lit ? 12 : 4) * sizeof(float));
                    if (lit) {
                        float info[4] = {
                            current.style.selectionFill && current.style.selectedId != 0 &&
                                    current.source.instances[indices[i]].objectId ==
                                        current.style.selectedId
                                ? 1.f
                                : 0.f,
                            reflectionLayer < 0 ? float(t.reflections.instanceLayers[indices[i]])
                                                : 0.f,
                            !current.source.planarKinds.empty() &&
                                    current.source.planarKinds[indices[i]] == 2
                                ? 1.f
                                : 0.f,
                            0};
                        std::memcpy(destination + 28 * sizeof(float), info, sizeof(info));
                    }
                }
            }
            mStats.uploadBytes += count * stride;
            auto &mesh = *current.meshes[meshIndex];
            if (wire) {
                if (!bgfx::isValid(mesh.wireVertices))
                    updateWireMesh(current, meshIndex);
                bgfx::setVertexBuffer(0, mesh.wireVertices);
            } else {
                bgfx::setVertexBuffer(0, mesh.vertices);
                bgfx::setIndexBuffer(mesh.indices);
            }
            uint64_t state = BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_DEPTH_TEST_LESS;
            if (pass || !transparent)
                state |= BGFX_STATE_WRITE_Z;
            if (current.style.cullFace)
                state |= reflectionLayer >= 0 ? BGFX_STATE_CULL_CCW : BGFX_STATE_CULL_CW;
            if (!pass && t.samples > 1 && current.style.msaa)
                state |= BGFX_STATE_MSAA;
            if (!pass) {
                const auto material = current.source.materials.empty()
                                          ? Material{-1, 0, 0, 0}
                                          : current.source.materials[materialIndex];
                const bool textureOn = current.style.textures && material.texture >= 0;
                const bool cube = textureOn && current.source.textures[material.texture].cube;
                bgfx::setTexture(0, mImageSampler,
                                 textureOn && !cube ? current.textures[material.texture] : mWhite);
                if (lit) {
                    bgfx::setTexture(1, mCubeSampler,
                                     cube ? current.textures[material.texture] : mWhiteCube);
                    int image = current.lighting.imageTexture;
                    bool validImage = image >= 0 && size_t(image) < current.textures.size() &&
                                      current.source.textures[image].cube;
                    bgfx::setTexture(2, mImageLightSampler,
                                     validImage ? current.textures[image] : mWhiteCube);
                    mReflections.bind(t.reflections, reflectionLayer, mWhite);
                    mShadows.bind(current.shadows, current.style.shadowQuality);
                    mLighting.bind(
                        current.lighting, current.style, drawCamera, current.source.linearColors,
                        validImage
                            ? std::floor(std::log2(current.source.textures[image].size.width))
                            : 0,
                        &camera);
                } else {
                    const float values[4] = {material.emission, material.specular,
                                             material.shininess,
                                             current.source.linearColors ? 1.f : 0.f};
                    bgfx::setUniform(mMaterial, values);
                }
                if (transparent)
                    state |= current.style.additive
                                 ? BGFX_STATE_BLEND_FUNC_SEPARATE(
                                       BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_ONE,
                                       BGFX_STATE_BLEND_ZERO, BGFX_STATE_BLEND_ONE)
                                 : BGFX_STATE_BLEND_FUNC_SEPARATE(
                                       BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA,
                                       BGFX_STATE_BLEND_ZERO, BGFX_STATE_BLEND_ONE);
                if (current.style.debugView == 4)
                    state = (state & (BGFX_STATE_CULL_MASK | BGFX_STATE_MSAA)) |
                            BGFX_STATE_WRITE_RGB | BGFX_STATE_BLEND_ADD;
            }
            bgfx::setState(state);
            bgfx::submit(pass ? t.view + pass : colorView, pass ? mDataPrograms[dataIndex]
                                                                : (wire  ? mWireProgram
                                                                   : lit ? mLitProgram
                                                                         : mColorProgram));
            ++mStats.drawCalls;
            mStats.instances += count;
        };
        auto drawTransparent = [&]() {
            if (current.style.transparent) {
                mTransparent.clear();
                const auto &v = drawCamera.view;
                const float eye[3] = {-(v[0] * v[3] + v[4] * v[7] + v[8] * v[11]),
                                      -(v[1] * v[3] + v[5] * v[7] + v[9] * v[11]),
                                      -(v[2] * v[3] + v[6] * v[7] + v[10] * v[11])};
                for (uint32_t i = 0; i < current.instances.size(); ++i) {
                    const auto &data = current.instances[i];
                    if (data[19] <= 0 || data[19] >= 1 || !visible(i))
                        continue;
                    float distance = 0;
                    for (size_t axis = 0; axis < 3; ++axis) {
                        float d = data[axis * 4 + 3] - eye[axis];
                        distance += d * d;
                    }
                    mTransparent.emplace_back(distance, i);
                }
                std::stable_sort(mTransparent.begin(), mTransparent.end(),
                                 [](auto a, auto b) { return a.first > b.first; });
                for (const auto &[distance, index] : mTransparent) {
                    uint32_t mat = current.source.materialIndices.empty()
                                       ? 0
                                       : current.source.materialIndices[index];
                    if (reflectionLayer >= 0 &&
                        t.reflections.excluded.contains(
                            ReflectionMaps::bucket(current.source.instances[index].mesh, mat)))
                        continue;
                    draw(current.source.instances[index].mesh, mat,
                         std::span<const uint32_t>(&index, 1), 0, true);
                }
            }
        };
        for (size_t layer = 0; request.color && !identityColor && reflectionDirty &&
                               layer < t.reflections.groups.size();
             ++layer) {
            reflectionLayer = layer;
            drawCamera = t.reflections.camera(camera, layer);
            drawFrustum = ClipFrustum(drawCamera);
            colorView = mNextSceneView++;
            timePass(colorView, RenderPass::Reflection);
            orderPass(colorView);
            auto reflectedView = columnMajor(drawCamera.view);
            bgfx::setViewRect(colorView, 0, 0, t.size.width, t.size.height);
            bgfx::setViewFrameBuffer(colorView, t.reflections.frames[layer]);
            bgfx::setViewTransform(colorView, reflectedView.data(), projection.data());
            bgfx::setViewClear(colorView, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, uint32_t{0x000000ff},
                               1.f);
            bgfx::setViewMode(colorView, bgfx::ViewMode::Sequential);
            bgfx::touch(colorView);
            for (const auto &batch : current.batches) {
                if (t.reflections.excluded.contains(
                        ReflectionMaps::bucket(batch.mesh, batch.material)))
                    continue;
                mDrawIndices.clear();
                for (auto index : batch.instances)
                    if (current.instances[index][19] >= 1 && visible(index))
                        mDrawIndices.push_back(index);
                draw(batch.mesh, batch.material, mDrawIndices, 0, false);
            }
            drawTransparent();
        }
        reflectionLayer = -1;
        drawCamera = camera;
        drawFrustum = ClipFrustum(camera);
        colorView = t.view;
        orderPass(t.view);
        orderPass(t.view + 1);
        for (uint16_t pass = 0; pass < 2; ++pass) {
            if (pass ? !renderData : (!renderColor || identityColor))
                continue;
            for (const auto &batch : current.batches) {
                mDrawIndices.clear();
                for (auto index : batch.instances) {
                    float alpha = current.instances[index][19];
                    if ((alpha >= 1 ||
                         (pass && current.style.transparentIds && current.style.transparent)) &&
                        visible(index))
                        mDrawIndices.push_back(index);
                }
                draw(batch.mesh, batch.material, mDrawIndices, pass, false);
            }
        }
        if (renderColor) {
            if (identityColor) {
                renderIdentityColor(current, t);
            } else {
                renderEnvironment(current, t, camera);
                renderSurfaces(current, t, camera);
                drawTransparent();
                renderOutline(current, t, view, projection);
                renderDebug(current, t, camera, view, projection);
                renderGizmos(current, t, view, projection);
            }
            t.reflectionKey = reflectionKey;
            t.colorKey = cacheColor ? std::optional{reflectionKey} : std::nullopt;
        }
        if (renderData) {
            t.dataKey = dataKey;
            t.cachedData |= dataMask;
        }
        t.latest = {
            id,           t.generation, current.source.revision, current.sequence, camera.revision,
            ++mSubmission};
        t.latest.statistics = {mStats.drawCalls - before.drawCalls,
                               mStats.instances - before.instances,
                               mStats.uploadBytes - before.uploadBytes, -1};
        auto &stats = t.latest.statistics;
        stats.passes = mTiming.get(id);
        stats.shadowInstances = mStats.shadowInstances - before.shadowInstances;
        stats.culledShadowInstances = mStats.culledShadowInstances - before.culledShadowInstances;
        stats.culledInstances = mStats.culledInstances - before.culledInstances;
        if (request.color && !identityColor && current.style.reflections &&
            !t.reflections.groups.empty()) {
            stats.reflectionRendered = renderColor && reflectionDirty;
            stats.reflectionReused = !stats.reflectionRendered;
        }
        if (request.color && !identityColor && current.style.shadows) {
            stats.shadowRendered = shadowRendered;
            stats.shadowReused = !shadowRendered;
        }
        return t.latest;
    }
    ReadbackTicket readback(FrameToken frame, Product product, Region region) override {
        return submitReadback(frame, product, region, {});
    }
    ReadbackTicket submitReadback(FrameToken frame, Product product, Region region,
                                  std::span<std::byte> destination) {
        owner();
        auto &t = target(frame.target);
        if (t.surface ||
            (t.dataProduct && product != Product::Color && product != Product::ColorAlpha &&
             product != *t.dataProduct) ||
            (t.colorOnly && product != Product::Color && product != Product::ColorAlpha) ||
            (t.dataOnly && (product == Product::Color || product == Product::ColorAlpha)))
            throw std::invalid_argument("Requested output is unavailable");
        if (frame != t.latest || !frame.submission)
            throw std::invalid_argument("Frame is no longer available for readback");
        auto free = std::find_if(mReadbacks.begin(), mReadbacks.end(),
                                 [](const auto &r) { return r.ticket == 0; });
        if (free == mReadbacks.end() && mReadbacks.size() >= 8)
            throw std::runtime_error("Readback queue is full; poll before submitting more");
        if (region.x >= t.size.width || region.y >= t.size.height)
            throw std::invalid_argument("Readback origin exceeds target");
        if (!region.width)
            region.width = t.size.width - region.x;
        if (!region.height)
            region.height = t.size.height - region.y;
        if (region.x > t.size.width || region.y > t.size.height ||
            region.width > t.size.width - region.x || region.height > t.size.height - region.y)
            throw std::invalid_argument("Readback region exceeds target");
        size_t slot = std::distance(mReadbacks.begin(), free);
        if (free == mReadbacks.end())
            mReadbacks.emplace_back();
        auto &r = mReadbacks[slot];
        bool reuse = r.size == Extent{region.width, region.height} && r.product == product &&
                     bgfx::isValid(r.buffer);
        if (!reuse)
            releaseSlot(r);
        r.ticket = allocateId();
        r.canceled = false;
        r.destination = destination;
        r.direct = false;
        r.frame = frame;
        r.product = product;
        r.gpuPacked = bgfx::isValid(mReadbackProgram) &&
                      (product == Product::Color || bgfx::getCaps()->originBottomLeft);
        r.size = {region.width, region.height};
        auto source = product == Product::ObjectId       ? t.data[0]
                      : product == Product::Segmentation ? t.data[1]
                      : product == Product::MetricDepth  ? t.data[2]
                                                         : t.color;
        uint32_t y =
            bgfx::getCaps()->originBottomLeft ? t.size.height - region.y - region.height : region.y;
        bgfx::TextureRegion src{};
        src.init(source, region.x, y, region.width, region.height);
        assignTargetView(t);
        orderPass(t.view + 2);
        if (r.gpuPacked) {
            // Pack RGB and orient data products on the GPU. The CPU can hand off
            // the owned buffer without another conversion or full-image allocation.
            const bool rgb = product == Product::Color;
            r.rowPitch = rgb ? ((r.size.width + 3) / 4) * 12 : r.size.width * pixelBytes(product);
            const uint32_t size = r.rowPitch * r.size.height;
            if (!reuse)
                r.buffer = bgfx::createDynamicIndexBuffer(size / 4, BGFX_BUFFER_INDEX32 |
                                                                        BGFX_BUFFER_COMPUTE_WRITE);
            if (!bgfx::isValid(r.buffer)) {
                r.ticket = 0;
                throw std::runtime_error("Cannot allocate readback conversion buffer");
            }
            const float rectangle[] = {float(region.x), float(region.y), float(region.width),
                                       float(region.height)};
            const float format = rgb                                ? 0.f
                                 : product == Product::MetricDepth  ? 2.f
                                 : product == Product::Segmentation ? 3.f
                                                                    : 1.f;
            const float layout[] = {float(t.size.height),
                                    bgfx::getCaps()->originBottomLeft ? 1.0f : 0.0f, format, 0};
            bgfx::setUniform(mReadbackRegion, rectangle);
            bgfx::setUniform(mReadbackLayout, layout);
            bgfx::setTexture(0, mReadbackImage, source, sampler);
            bgfx::setBuffer(1, r.buffer, bgfx::Access::Write);
            const uint32_t pixelsPerGroup = rgb ? 256 : 64;
            bgfx::dispatch(t.view + 2, mReadbackProgram,
                           (r.size.width + pixelsPerGroup - 1) / pixelsPerGroup, r.size.height);
            bgfx::BufferRegion dst{};
            dst.init(r.buffer, 0, size);
            r.direct = !destination.empty() && r.rowPitch == r.size.width * pixelBytes(product);
            if (!r.direct)
                r.bytes.resize(size);
            r.ready = bgfx::read(dst, r.direct ? destination.data() : r.bytes.data());
            t.lastReadback = mGpuFrame;
            mPendingCommands = true;
            return {r.ticket};
        }
        // Buffer copies cannot source an MSAA resource. Preserve bgfx's resolved
        // color image with a GPU-only copy before transferring linear storage.
        if ((product == Product::Color || product == Product::ColorAlpha) &&
            t.allocatedSamples > 1) {
            if (!bgfx::isValid(r.resolved))
                r.resolved = bgfx::createTexture2D(r.size.width, r.size.height, false, 1,
                                                   bgfx::TextureFormat::RGBA8,
                                                   BGFX_TEXTURE_BLIT_DST | sampler);
            if (!bgfx::isValid(r.resolved)) {
                r.ticket = 0;
                throw std::runtime_error("Cannot allocate resolved readback texture");
            }
            bgfx::TextureRegion resolved{};
            resolved.init(r.resolved);
            bgfx::blit(t.view + 2, resolved, src);
            src = resolved;
        }
        bgfx::BufferRegion dst{};
        // Honor each backend's copy alignment and read linear storage directly.
        // A staging texture adds a texture-to-CPU conversion on Metal.
        dst.init(src);
        if (!reuse)
            r.buffer = bgfx::createDynamicIndexBuffer(
                (dst.size + 3) / 4, BGFX_BUFFER_INDEX32 | BGFX_BUFFER_COMPUTE_WRITE);
        if (!bgfx::isValid(r.buffer)) {
            r.ticket = 0;
            throw std::runtime_error("Cannot allocate readback buffer");
        }
        dst.handle = r.buffer;
        r.rowPitch = dst.rowPitch;
        r.bytes.resize(dst.size);
        bgfx::blit(t.view + 2, dst, src);
        r.ready = bgfx::read(dst, r.bytes.data());
        t.lastReadback = mGpuFrame;
        mPendingCommands = true;
        return {r.ticket};
    }
    ReadbackState readInto(FrameToken frame, ImageView destination, Region region) override {
        owner();
        const auto size = target(frame.target).size;
        if (region.x >= size.width || region.y >= size.height)
            throw std::invalid_argument("Readback origin exceeds target");
        const Extent output = {region.width ? region.width : size.width - region.x,
                               region.height ? region.height : size.height - region.y};
        if (destination.size != output ||
            destination.pixels.size() !=
                size_t(output.width) * output.height * pixelBytes(destination.product))
            throw std::invalid_argument("Readback destination has the wrong size");
        auto ticket = submitReadback(frame, destination.product, region, destination.pixels);
        // The caller's storage must remain alive until completion. Unlike an owned
        // asynchronous ticket, synchronous delivery cannot time out and abandon it.
        for (;;) {
            advance();
            auto result = poll(ticket);
            if (result.state != ReadbackState::Pending)
                return result.state;
        }
    }
    ReadbackResult poll(ReadbackTicket ticket) override {
        owner();
        if (!ticket.id)
            throw std::invalid_argument("Invalid readback ticket");
        auto it = std::find_if(mReadbacks.begin(), mReadbacks.end(),
                               [&](const auto &r) { return r.ticket == ticket.id; });
        if (it == mReadbacks.end())
            throw std::invalid_argument("Unknown or consumed readback ticket");
        if (!complete(mGpuFrame, it->ready))
            return {ReadbackState::Pending, it->frame, {}};
        ReadbackResult result;
        result.frame = it->frame;
        result.state = it->canceled ? ReadbackState::Canceled : ReadbackState::Ready;
        if (!it->canceled && !it->direct) {
            auto &image = result.image;
            image.product = it->product;
            image.size = it->size;
            auto stride = pixelBytes(image.product);
            const bool flip = !it->gpuPacked && bgfx::getCaps()->originBottomLeft;
            const bool rawWords =
                it->gpuPacked || image.product == Product::ColorAlpha ||
                image.product == Product::MetricDepth ||
                ((image.product == Product::ObjectId || image.product == Product::Segmentation) &&
                 std::endian::native == std::endian::little);
            if (it->destination.empty() && !flip && rawWords &&
                it->rowPitch == image.size.width * stride) {
                image.pixels = std::move(it->bytes);
            } else {
                auto output = it->destination;
                if (output.empty()) {
                    image.pixels.resize(size_t(image.size.width) * image.size.height * stride);
                    output = image.pixels;
                }
                for (size_t y = 0; y < image.size.height; ++y) {
                    const size_t offset = (flip ? image.size.height - 1 - y : y) * it->rowPitch;
                    const auto *source = it->bytes.data() + offset;
                    auto *destination = output.data() + y * image.size.width * stride;
                    if (rawWords) {
                        std::memcpy(destination, source, image.size.width * stride);
                    } else if (image.product == Product::Color) {
                        detail::copyRgb(destination, source, image.size.width);
                    } else if (image.product == Product::Segmentation) {
                        for (size_t x = 0; x < image.size.width; ++x) {
                            uint16_t words[4];
                            std::memcpy(words, source + x * 8, 8);
                            const uint32_t values[] = {
                                uint32_t(words[0]) | uint32_t(words[1]) << 16,
                                uint32_t(words[2]) | uint32_t(words[3]) << 16};
                            std::memcpy(destination + x * 8, values, 8);
                        }
                    } else {
                        for (size_t x = 0; x < image.size.width; ++x) {
                            const uint32_t value = word(source + x * 4);
                            std::memcpy(destination + x * 4, &value, 4);
                        }
                    }
                }
            }
        }
        it->destination = {};
        it->ticket = 0;
        return result;
    }
    FrameStats advance() override {
        owner();
        flush();
        const auto *stats = bgfx::getStats();
        if (stats->gpuTimerFreq > 0 && stats->gpuTimeEnd >= stats->gpuTimeBegin)
            mStats.gpuMs =
                1000.0 * double(stats->gpuTimeEnd - stats->gpuTimeBegin) / stats->gpuTimerFreq;
        auto result = mStats;
        mStats = {};
        return result;
    }
    void reloadShaders() override {
        owner();
        if (mPendingCommands)
            flush();
        loadPrograms();
        for (auto &[id, value] : mScenes)
            value.shadowKey.reset();
        for (auto &[id, value] : mTargets) {
            value.colorKey.reset();
            value.reflectionKey.reset();
            value.dataKey.reset();
            value.cachedData = 0;
        }
    }
    Texture targetTexture(Target id) const override {
        owner();
        if (!mTargets.contains(id.id))
            throw std::invalid_argument("Unknown target texture");
        return {id.id | targetTextureBit};
    }
    Texture uploadTexture(Extent size, std::span<const std::byte> rgba) override {
        owner();
        extent(size);
        if (rgba.size() != size_t(size.width) * size.height * 4)
            throw std::invalid_argument("Invalid RGBA texture bytes");
        // Font coverage and ImGui textured AA require bilinear filtering.
        // Integer data targets retain their separate point-sampling policy.
        auto texture = bgfx::createTexture2D(
            size.width, size.height, false, 1, bgfx::TextureFormat::RGBA8,
            BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP, bgfx::copy(rgba.data(), rgba.size()));
        Texture id{allocateId()};
        mTextures.emplace(id.id, texture);
        return id;
    }
    void destroy(Texture id) override {
        owner();
        auto it = mTextures.find(id.id);
        if (it == mTextures.end())
            throw std::invalid_argument("Unknown owned texture");
        bgfx::destroy(it->second);
        mTextures.erase(it);
    }
    FrameToken renderUi(const UiFrame &ui, Target output) override {
        owner();
        extent(ui.size);
        validateUi(ui);
        if (!mHasWindow && !output.id)
            throw std::logic_error("UI rendering requires a window");
        if (!output.id && ui.size != mWindowSize) {
            mSwapChain.width = ui.size.width;
            mSwapChain.height = ui.size.height;
            bgfx::reset(mResetFlags, &mSwapChain);
            mWindowSize = ui.size;
        }
        auto projection = identity();
        projection[0] = 2.0f / ui.size.width;
        projection[3] = -1;
        projection[5] = -2.0f / ui.size.height;
        projection[7] = 1;
        projection = columnMajor(projection);
        bgfx::ViewId view = 250;
        bgfx::FrameBufferHandle framebuffer = BGFX_INVALID_HANDLE;
        FrameToken token;
        if (output.id) {
            auto &t = target(output);
            if (t.size != ui.size)
                throw std::invalid_argument("UI extent must match its target");
            beginTarget(t);
            view = t.view + 1;
            framebuffer = t.colorFb;
            t.colorOnly = true;
            t.dataOnly = false;
            t.dataProduct.reset();
            t.colorKey.reset();
            token = {output, t.generation, 0, 0, 0, ++mSubmission};
        }
        if (!output.id) {
            if (mMainRender == mGpuFrame)
                flush();
            mMainRender = mGpuFrame;
            token.submission = ++mSubmission;
        }
        token.statistics.passes = mTiming.get(output);
        if (output.id)
            target(output).latest = token;
        mTiming.record(view, output, token.submission, RenderPass::Ui);
        orderPass(view);
        bgfx::setViewMode(view, bgfx::ViewMode::Sequential);
        bgfx::setViewRect(view, 0, 0, ui.size.width, ui.size.height);
        bgfx::setViewFrameBuffer(view, framebuffer);
        bgfx::setViewTransform(view, nullptr, projection.data());
        bgfx::setViewClear(view, BGFX_CLEAR_COLOR, 0x14191eff);
        bgfx::touch(view);
        mPendingCommands = true;
        if (ui.vertices.empty() || ui.indices.empty())
            return token;
        bgfx::TransientVertexBuffer vertices;
        bgfx::TransientIndexBuffer indices;
        if (bgfx::getAvailTransientVertexBuffer(ui.vertices.size(), mUiVertices) !=
                ui.vertices.size() ||
            bgfx::getAvailTransientIndexBuffer(ui.indices.size(), true) != ui.indices.size())
            throw std::runtime_error("UI upload capacity exhausted");
        bgfx::allocTransientVertexBuffer(&vertices, ui.vertices.size(), mUiVertices);
        bgfx::allocTransientIndexBuffer(&indices, ui.indices.size(), true);
        std::memcpy(vertices.data, ui.vertices.data(), ui.vertices.size_bytes());
        std::memcpy(indices.data, ui.indices.data(), ui.indices.size_bytes());
        for (const auto &command : ui.commands) {
            auto clip = command.clip;
            float left = std::clamp(clip[0], 0.0f, float(ui.size.width)),
                  top = std::clamp(clip[1], 0.0f, float(ui.size.height));
            float right = std::clamp(clip[2], left, float(ui.size.width)),
                  bottom = std::clamp(clip[3], top, float(ui.size.height));
            if (right <= left || bottom <= top)
                continue;
            bgfx::setScissor(left, top, right - left, bottom - top);
            bgfx::setVertexBuffer(0, &vertices, command.vertexOffset,
                                  ui.vertices.size() - command.vertexOffset);
            bgfx::setIndexBuffer(&indices, command.firstIndex, command.indexCount);
            const float textureInfo[] = {
                bgfx::getCaps()->originBottomLeft && (command.texture.id & targetTextureBit) ? 1.f
                                                                                             : 0.f,
                0, 0, 0};
            bgfx::setUniform(mUiTextureInfo, textureInfo);
            bgfx::setTexture(0, mImageSampler, textureHandle(command.texture));
            bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A |
                           BGFX_STATE_BLEND_FUNC_SEPARATE(
                               BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA,
                               BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_INV_SRC_ALPHA));
            bgfx::submit(view, mUiProgram);
        }
        return token;
    }
};
} // namespace
std::unique_ptr<Renderer> makeBgfxRenderer(const BgfxOptions &options) {
    auto renderer = std::make_unique<BgfxRenderer>();
    renderer->initialize(options);
    return renderer;
}
} // namespace mojive
