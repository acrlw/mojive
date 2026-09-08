#include <SDL3/SDL.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <deque>
#include <fstream>
#include <mojive/backends/sdl.hpp>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace mojive {
namespace {
std::atomic<uint64_t> nextId{1};
constexpr uint64_t targetBit = uint64_t{1} << 63;
uint64_t allocateId() {
    return nextId.fetch_add(1, std::memory_order_relaxed);
}
template <class T> T checked(T value) {
    if (!value)
        throw std::runtime_error(std::string("SDL GPU: ") + SDL_GetError());
    return value;
}
struct Buffer {
    SDL_GPUBuffer *gpu = nullptr;
    uint32_t size = 0;
};
struct MeshData {
    Buffer vertices, indices;
    std::vector<uint32_t> instances;
};
struct TargetData {
    Extent size;
    uint32_t samples = 1;
    uint64_t generation = 1;
    SDL_Window *window = nullptr;
    SDL_GPUTexture *color = nullptr, *msaa = nullptr, *depth = nullptr, *dataDepth = nullptr;
    std::array<SDL_GPUTexture *, 4> data{};
    FrameToken latest;
    bool colorOnly = false;
};
using Fence = std::shared_ptr<SDL_GPUFence>;
struct Readback {
    uint64_t ticket = 0;
    FrameToken frame;
    Product product = Product::Color;
    Extent size;
    bool canceled = false;
    uint32_t pitch = 0, count = 1, capacity = 0;
    SDL_GPUTransferBuffer *buffer = nullptr;
    Fence fence;
};
class SdlRenderer final : public Renderer {
    SDL_GPUDevice *mDevice = nullptr;
    bool mVideo = false;
    SDL_GPUCommandBuffer *mCommand = nullptr;
    std::thread::id mOwner = std::this_thread::get_id();
    Capabilities mCaps;
    SceneSource mScene;
    uint64_t mSequence = 0, mSubmission = 0;
    std::string mShaderDirectory;
    SDL_GPUShaderFormat mShaderFormat = SDL_GPU_SHADERFORMAT_INVALID;
    std::vector<MeshData> mMeshes;
    std::vector<std::array<float, 28>> mInstances;
    std::vector<float> mPacked;
    Buffer mInstanceColor, mInstanceData, mUiVertices, mUiIndices;
    SDL_GPUTransferBuffer *mUpload = nullptr;
    uint32_t mUploadCapacity = 0;
    std::unordered_map<uint64_t, TargetData> mTargets;
    std::unordered_map<uint64_t, SDL_GPUTexture *> mTextures;
    std::array<Readback, 8> mReadbacks;
    std::deque<Fence> mInFlight;
    std::unordered_map<uint64_t, SDL_GPUGraphicsPipeline *> mPipelines;
    SDL_GPUSampler *mSampler = nullptr;
    FrameStats mStats;
    void owner() const {
        if (std::this_thread::get_id() != mOwner)
            throw std::logic_error("Renderer called outside its owner thread");
    }
    SDL_GPUCommandBuffer *command() {
        if (!mCommand)
            mCommand = checked(SDL_AcquireGPUCommandBuffer(mDevice));
        return mCommand;
    }
    void submit() {
        if (!mCommand)
            return;
        while (!mInFlight.empty() && SDL_QueryGPUFence(mDevice, mInFlight.front().get()))
            mInFlight.pop_front();
        // Bound outstanding GPU work even when there is no swapchain or readback.
        if (mInFlight.size() >= 2) {
            auto *first = mInFlight.front().get();
            checked(SDL_WaitForGPUFences(mDevice, true, &first, 1));
            mInFlight.pop_front();
        }
        auto *raw = checked(SDL_SubmitGPUCommandBufferAndAcquireFence(mCommand));
        mCommand = nullptr;
        Fence fence(raw, [this](auto *f) { SDL_ReleaseGPUFence(mDevice, f); });
        for (auto &r : mReadbacks)
            if (r.ticket && !r.fence)
                r.fence = fence;
        mInFlight.push_back(fence);
    }

    void idle() {
        submit();
        checked(SDL_WaitForGPUIdle(mDevice));
        mInFlight.clear();
    }
    void extent(Extent s) const {
        if (!s.width || !s.height || s.width > mCaps.maxTextureSize ||
            s.height > mCaps.maxTextureSize)
            throw std::invalid_argument("Unsupported render target size");
    }
    TargetData &target(Target id) {
        auto i = mTargets.find(id.id);
        if (i == mTargets.end())
            throw std::invalid_argument("Unknown render target");
        return i->second;
    }
    SDL_GPUTransferBuffer *transfer(uint32_t bytes, SDL_GPUTransferBufferUsage usage) {
        SDL_GPUTransferBufferCreateInfo info{};
        info.usage = usage;
        info.size = bytes;
        return checked(SDL_CreateGPUTransferBuffer(mDevice, &info));
    }
    void *mapUpload(uint32_t size) {
        if (size > mUploadCapacity) {
            auto *replacement = transfer(size, SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD);
            if (mUpload)
                SDL_ReleaseGPUTransferBuffer(mDevice, mUpload);
            mUpload = replacement;
            mUploadCapacity = size;
        }
        return checked(SDL_MapGPUTransferBuffer(mDevice, mUpload, true));
    }
    void release(Buffer &b) {
        if (b.gpu)
            SDL_ReleaseGPUBuffer(mDevice, b.gpu);
        b = {};
    }
    void uploadBuffer(Buffer &b, SDL_GPUBufferUsageFlags usage, const void *data, uint32_t size) {
        if (!size)
            return;
        if (size > b.size) {
            release(b);
            SDL_GPUBufferCreateInfo info{};
            info.size = size;
            info.usage = usage;
            b.gpu = checked(SDL_CreateGPUBuffer(mDevice, &info));
            b.size = size;
        }
        std::memcpy(mapUpload(size), data, size);
        SDL_UnmapGPUTransferBuffer(mDevice, mUpload);
        auto *pass = checked(SDL_BeginGPUCopyPass(command()));
        SDL_GPUTransferBufferLocation src{mUpload, 0};
        SDL_GPUBufferRegion dst{b.gpu, 0, size};
        SDL_UploadToGPUBuffer(pass, &src, &dst, true);
        SDL_EndGPUCopyPass(pass);
        mStats.uploadBytes += size;
    }
    SDL_GPUTexture *texture(Extent size, SDL_GPUTextureFormat format,
                            SDL_GPUTextureUsageFlags usage, uint32_t samples = 1) {
        SDL_GPUTextureCreateInfo info{};
        info.type = SDL_GPU_TEXTURETYPE_2D;
        info.format = format;
        info.usage = usage;
        info.width = size.width;
        info.height = size.height;
        info.layer_count_or_depth = 1;
        info.num_levels = 1;
        info.sample_count = samples == 4 ? SDL_GPU_SAMPLECOUNT_4 : SDL_GPU_SAMPLECOUNT_1;
        if (!SDL_GPUTextureSupportsFormat(mDevice, format, info.type, usage) ||
            !SDL_GPUTextureSupportsSampleCount(mDevice, format, info.sample_count))
            throw std::runtime_error("Unsupported SDL texture format/sample count");
        return checked(SDL_CreateGPUTexture(mDevice, &info));
    }
    void releaseTarget(TargetData &t, bool wait = true) {
        for (auto *p : {t.color, t.msaa, t.depth, t.dataDepth})
            if (p)
                SDL_ReleaseGPUTexture(mDevice, p);
        for (auto *p : t.data)
            if (p)
                SDL_ReleaseGPUTexture(mDevice, p);
        t.color = t.msaa = t.depth = t.dataDepth = nullptr;
        t.data = {};
        if (t.window) {
            if (wait)
                idle();
            SDL_ReleaseWindowFromGPUDevice(mDevice, t.window);
            SDL_DestroyWindow(t.window);
            t.window = nullptr;
        }
    }
    void allocateTarget(TargetData &t) {
        constexpr auto color = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
        t.color = texture(t.size, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, color);
        if (t.samples == 4)
            t.msaa = texture(t.size, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
                             SDL_GPU_TEXTUREUSAGE_COLOR_TARGET, 4);
        t.depth = texture(t.size, SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
                          SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET, t.samples);
        t.dataDepth = texture(t.size, SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
                              SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET);
        for (size_t i = 0; i < 4; ++i)
            t.data[i] = texture(t.size,
                                i == 3 ? SDL_GPU_TEXTUREFORMAT_R32_FLOAT
                                       : SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
                                color);
    }
    void cancel(Target id = {}) {
        for (auto &r : mReadbacks)
            if (!id.id || r.frame.target == id)
                r.canceled = true;
    }
    SDL_GPUShader *shader(const char *entry, bool vertex, bool ui) {
        SDL_GPUShaderCreateInfo info{};
        const char *extension = mShaderFormat == SDL_GPU_SHADERFORMAT_MSL    ? ".msl"
                                : mShaderFormat == SDL_GPU_SHADERFORMAT_DXIL ? ".dxil"
                                                                             : ".spv";
        std::ifstream file(mShaderDirectory + "/" + entry + extension, std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot open SDL shader: " + std::string(entry) + extension);
        std::string code(std::istreambuf_iterator<char>(file), {});
        if (mShaderFormat == SDL_GPU_SHADERFORMAT_MSL)
            code.push_back('\0');
        info.code = reinterpret_cast<const Uint8 *>(code.data());
        info.code_size = code.size();
        info.entrypoint = mShaderFormat == SDL_GPU_SHADERFORMAT_MSL ? entry : "main";
        info.format = mShaderFormat;
        info.stage = vertex ? SDL_GPU_SHADERSTAGE_VERTEX : SDL_GPU_SHADERSTAGE_FRAGMENT;
        info.num_uniform_buffers = vertex ? 1 : 0;
        info.num_samplers = ui && !vertex ? 1 : 0;
        return checked(SDL_CreateGPUShader(mDevice, &info));
    }
    SDL_GPUGraphicsPipeline *
    pipeline(int kind, uint32_t samples,
             SDL_GPUTextureFormat format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM) {
        uint64_t key = uint64_t(format) * 100 + samples * 10 + kind;
        if (auto i = mPipelines.find(key); i != mPipelines.end())
            return i->second;
        bool ui = kind == 2;
        SDL_GPUGraphicsPipelineCreateInfo info{};
        auto *vs = shader(ui ? "ui_vertex" : "scene_vertex", true, ui);
        SDL_GPUShader *fs = nullptr;
        try {
            fs = shader(ui          ? "ui_fragment"
                        : kind == 1 ? "data_fragment"
                                    : "color_fragment",
                        false, ui);
        } catch (...) {
            SDL_ReleaseGPUShader(mDevice, vs);
            throw;
        }
        info.vertex_shader = vs;
        info.fragment_shader = fs;
        std::array<SDL_GPUVertexBufferDescription, 2> buffers{
            {{0, uint32_t(ui ? sizeof(UiVertex) : sizeof(Vertex)), SDL_GPU_VERTEXINPUTRATE_VERTEX,
              0},
             {1, 80, SDL_GPU_VERTEXINPUTRATE_INSTANCE, 0}}};
        std::array<SDL_GPUVertexAttribute, 7> attributes{};
        attributes[0] = {
            0, 0, ui ? SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2 : SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0};
        attributes[1] = {
            1, 0, ui ? SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2 : SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
            ui ? 8u : 12u};
        if (ui)
            attributes[2] = {2, 0, SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM, 16};
        else
            for (uint32_t i = 2; i < 7; ++i)
                attributes[i] = {i, 1, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, (i - 2) * 16};
        info.vertex_input_state = {buffers.data(), ui ? 1u : 2u, attributes.data(), ui ? 3u : 7u};
        info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
        info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
        info.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
        info.rasterizer_state.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
        info.rasterizer_state.enable_depth_clip = true;
        info.multisample_state.sample_count =
            samples == 4 ? SDL_GPU_SAMPLECOUNT_4 : SDL_GPU_SAMPLECOUNT_1;
        info.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS;
        info.depth_stencil_state.enable_depth_test = info.depth_stencil_state.enable_depth_write =
            !ui;
        std::array<SDL_GPUColorTargetDescription, 4> colors{};
        for (size_t i = 0; i < 4; ++i)
            colors[i].format = i == 3 ? SDL_GPU_TEXTUREFORMAT_R32_FLOAT : format;
        if (ui) {
            auto &b = colors[0].blend_state;
            b.enable_blend = true;
            b.src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA;
            b.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
            b.color_blend_op = b.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
            b.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
            b.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
        }
        info.target_info.color_target_descriptions = colors.data();
        info.target_info.num_color_targets = kind == 1 ? 4 : 1;
        info.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
        info.target_info.has_depth_stencil_target = !ui;
        auto *result = SDL_CreateGPUGraphicsPipeline(mDevice, &info);
        SDL_ReleaseGPUShader(mDevice, vs);
        SDL_ReleaseGPUShader(mDevice, fs);
        mPipelines[key] = checked(result);
        return result;
    }
    SDL_GPUTexture *textureHandle(Texture id) const {
        if (id.id & targetBit) {
            auto i = mTargets.find(id.id & ~targetBit);
            if (i == mTargets.end() || i->second.window)
                throw std::invalid_argument("Expired target texture");
            return i->second.color;
        }
        auto i = mTextures.find(id.id);
        if (i == mTextures.end())
            throw std::invalid_argument("Unknown UI texture");
        return i->second;
    }
    FrameToken token(Target id, TargetData &t, uint64_t camera) {
        t.latest = {id, t.generation, mScene.revision, mSequence, camera, ++mSubmission};
        return t.latest;
    }

  public:
    void initialize(const SdlOptions &options) {
        checked(SDL_InitSubSystem(SDL_INIT_VIDEO));
        mVideo = true;
#if defined(__APPLE__)
        mShaderFormat = SDL_GPU_SHADERFORMAT_MSL;
        const char *driver = "metal";
#elif defined(_WIN32)
        mShaderFormat = SDL_GPU_SHADERFORMAT_DXIL;
        const char *driver = "direct3d12";
#else
        mShaderFormat = SDL_GPU_SHADERFORMAT_SPIRV;
        const char *driver = "vulkan";
#endif
        mDevice = checked(SDL_CreateGPUDevice(mShaderFormat, false, driver));
        checked(SDL_SetGPUAllowedFramesInFlight(mDevice, 2));
        mCaps = {std::string("SDL3 ") + SDL_GetGPUDeviceDriver(mDevice),
                 "",
                 true,
                 true,
                 true,
                 false,
                 false,
                 true,
                 16384};
        mCaps.integerTarget =
            SDL_GPUTextureSupportsFormat(mDevice, SDL_GPU_TEXTUREFORMAT_R32_UINT,
                                         SDL_GPU_TEXTURETYPE_2D, SDL_GPU_TEXTUREUSAGE_COLOR_TARGET);
        mCaps.signedPairTarget =
            SDL_GPUTextureSupportsFormat(mDevice, SDL_GPU_TEXTUREFORMAT_R32G32_INT,
                                         SDL_GPU_TEXTURETYPE_2D, SDL_GPU_TEXTUREUSAGE_COLOR_TARGET);
        mCaps.device = SDL_GetStringProperty(SDL_GetGPUDeviceProperties(mDevice),
                                             SDL_PROP_GPU_DEVICE_NAME_STRING, "Unknown GPU");
        mShaderDirectory = options.shaderDirectory;
        SDL_GPUSamplerCreateInfo sampler{};
        sampler.min_filter = sampler.mag_filter = SDL_GPU_FILTER_LINEAR;
        sampler.address_mode_u = sampler.address_mode_v = sampler.address_mode_w =
            SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
        mSampler = checked(SDL_CreateGPUSampler(mDevice, &sampler));
        pipeline(0, 1);
        pipeline(0, 4);
        pipeline(1, 1);
        pipeline(2, 1);
        if (options.window.handle) {
            auto id = createSurface(options.window);
            mTargets[0] = mTargets.at(id.id);
            mTargets.erase(id.id);
        }
    }
    ~SdlRenderer() override {
        if (mDevice) {
            if (mCommand)
                SDL_CancelGPUCommandBuffer(mCommand);
            SDL_WaitForGPUIdle(mDevice);
            mInFlight.clear();
            for (auto &r : mReadbacks) {
                r.fence.reset();
                if (r.buffer)
                    SDL_ReleaseGPUTransferBuffer(mDevice, r.buffer);
            }
            mCommand = nullptr;
            for (auto &[id, t] : mTargets)
                releaseTarget(t, false);
            for (auto &[id, t] : mTextures)
                SDL_ReleaseGPUTexture(mDevice, t);
            for (auto &[key, p] : mPipelines)
                SDL_ReleaseGPUGraphicsPipeline(mDevice, p);
            for (auto &m : mMeshes) {
                release(m.vertices);
                release(m.indices);
            }
            release(mInstanceColor);
            release(mInstanceData);
            release(mUiVertices);
            release(mUiIndices);
            if (mUpload)
                SDL_ReleaseGPUTransferBuffer(mDevice, mUpload);
            if (mSampler)
                SDL_ReleaseGPUSampler(mDevice, mSampler);
            SDL_DestroyGPUDevice(mDevice);
        }
        if (mVideo)
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
    }
    const Capabilities &capabilities() const override {
        return mCaps;
    }
    void setScene(const SceneSource &scene) override {
        owner();
        validateScene(scene);
        cancel();
        for (auto &m : mMeshes) {
            release(m.vertices);
            release(m.indices);
        }
        mMeshes.clear();
        mScene = scene;
        mInstances.resize(scene.instances.size());
        mMeshes.resize(scene.meshes.size());
        for (size_t i = 0; i < mMeshes.size(); ++i) {
            auto &m = mMeshes[i];
            auto &s = scene.meshes[i];
            uploadBuffer(m.vertices, SDL_GPU_BUFFERUSAGE_VERTEX, s.vertices.data(),
                         s.vertices.size() * sizeof(Vertex));
            uploadBuffer(m.indices, SDL_GPU_BUFFERUSAGE_INDEX, s.indices.data(),
                         s.indices.size() * 4);
        }
        for (uint32_t i = 0; i < scene.instances.size(); ++i) {
            const auto &s = scene.instances[i];
            mMeshes[s.mesh].instances.push_back(i);
            auto &out = mInstances[i];
            out = {};
            auto mat = identity();
            std::copy_n(mat.begin(), 12, out.begin());
            std::copy(s.color.begin(), s.color.end(), out.begin() + 12);
            std::array<uint32_t, 3> ids{s.objectId, std::bit_cast<uint32_t>(s.segmentation[0]),
                                        std::bit_cast<uint32_t>(s.segmentation[1])};
            for (size_t j = 0; j < 3; ++j) {
                out[20 + j * 2] = float(ids[j] & 65535);
                out[21 + j * 2] = float(ids[j] >> 16);
            }
        }
        mPacked.resize(mInstances.size() * 20);
        mSequence = 0;
        for (auto &[id, t] : mTargets)
            t.latest = {};
    }
    void update(const SceneFrame &frame) override {
        owner();
        validateFrame(mScene, frame);
        mSequence = frame.sequence;
        for (size_t i = 0; i < mInstances.size(); ++i)
            std::copy_n(frame.transforms[i].begin(), 12, mInstances[i].begin());
    }
    void updateMesh(uint32_t index, std::span<const Vertex> vertices) override {
        owner();
        if (index >= mMeshes.size() || vertices.size() != mScene.meshes[index].vertices.size())
            throw std::invalid_argument("Invalid dynamic mesh update");
        uploadBuffer(mMeshes[index].vertices, SDL_GPU_BUFFERUSAGE_VERTEX, vertices.data(),
                     vertices.size_bytes());
    }
    Target createTarget(Extent size, uint32_t samples) override {
        owner();
        extent(size);
        if (samples != 1 && samples != 4)
            throw std::invalid_argument("Unsupported MSAA count");
        TargetData t;
        t.size = size;
        t.samples = samples;
        try {
            allocateTarget(t);
        } catch (...) {
            releaseTarget(t);
            throw;
        }
        auto id = allocateId();
        mTargets.emplace(id, t);
        return {id};
    }
    Target createSurface(NativeWindow window) override {
        owner();
        extent(window.size);
        if (!window.handle)
            throw std::invalid_argument("Missing native window");
        auto props = checked(SDL_CreateProperties());
        checked(
            SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_HIGH_PIXEL_DENSITY_BOOLEAN, true));
#if defined(__APPLE__)
        checked(SDL_SetPointerProperty(props, SDL_PROP_WINDOW_CREATE_COCOA_WINDOW_POINTER,
                                       window.handle));
        checked(SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_METAL_BOOLEAN, true));
#elif defined(_WIN32)
        checked(SDL_SetPointerProperty(props, SDL_PROP_WINDOW_CREATE_WIN32_HWND_POINTER,
                                       window.handle));
#else
        checked(SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X11_WINDOW_NUMBER,
                                      reinterpret_cast<uintptr_t>(window.handle)));
#endif
        auto *wrapped = SDL_CreateWindowWithProperties(props);
        SDL_DestroyProperties(props);
        checked(wrapped);
        if (!SDL_ClaimWindowForGPUDevice(mDevice, wrapped)) {
            SDL_DestroyWindow(wrapped);
            checked(false);
        }
        TargetData t;
        t.window = wrapped;
        t.size = window.size;
        t.colorOnly = true;
        auto id = allocateId();
        mTargets.emplace(id, t);
        return {id};
    }
    void resize(Target id, Extent size) override {
        owner();
        extent(size);
        auto &t = target(id);
        if (t.size == size)
            return;
        if (t.window) {
            // SDL updates the claimed swapchain from the platform window's drawable
            // size on acquisition; keep its identity and resources across resizes.
            t.size = size;
            ++t.generation;
            t.latest = {};
            return;
        }
        cancel(id);
        releaseTarget(t);
        t.size = size;
        ++t.generation;
        t.latest = {};
        allocateTarget(t);
    }
    void destroy(Target id) override {
        owner();
        auto &t = target(id);
        cancel(id);
        releaseTarget(t);
        mTargets.erase(id.id);
    }
    FrameToken render(Target id, const CameraView &camera) override {
        owner();
        auto &t = target(id);
        if (t.window)
            throw std::invalid_argument("Render the scene to an offscreen target");
        t.colorOnly = false;
        std::array<float, 32> uniforms;
        std::copy(camera.view.begin(), camera.view.end(), uniforms.begin());
        std::copy(camera.projection.begin(), camera.projection.end(), uniforms.begin() + 16);
        for (int kind = 0; kind < 2; ++kind) {
            size_t at = 0;
            for (const auto &mesh : mMeshes)
                for (auto index : mesh.instances) {
                    auto &instance = mInstances[index];
                    std::copy_n(instance.begin(), 12, mPacked.begin() + at);
                    std::copy_n(instance.begin() + (kind ? 20 : 12), 8, mPacked.begin() + at + 12);
                    at += 20;
                }
            auto &buffer = kind ? mInstanceData : mInstanceColor;
            uploadBuffer(buffer, SDL_GPU_BUFFERUSAGE_VERTEX, mPacked.data(),
                         mPacked.size() * sizeof(float));
            std::array<SDL_GPUColorTargetInfo, 4> colors{};
            for (size_t i = 0; i < (kind ? 4 : 1); ++i) {
                auto &c = colors[i];
                c.texture = kind ? t.data[i] : (t.msaa ? t.msaa : t.color);
                c.load_op = SDL_GPU_LOADOP_CLEAR;
                c.store_op = SDL_GPU_STOREOP_STORE;
                c.clear_color = kind ? (i == 0   ? SDL_FColor{0, 0, 0, 0}
                                        : i == 3 ? SDL_FColor{camera.farPlane, 0, 0, 0}
                                                 : SDL_FColor{1, 1, 1, 1})
                                     : SDL_FColor{32.0f / 255, 38.0f / 255, 47.0f / 255, 1};
                if (!kind && t.msaa) {
                    c.resolve_texture = t.color;
                    c.store_op = SDL_GPU_STOREOP_RESOLVE;
                }
            }
            SDL_GPUDepthStencilTargetInfo depth{};
            depth.texture = kind ? t.dataDepth : t.depth;
            depth.clear_depth = 1;
            depth.load_op = SDL_GPU_LOADOP_CLEAR;
            depth.store_op = SDL_GPU_STOREOP_DONT_CARE;
            depth.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
            depth.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;
            auto *pass =
                checked(SDL_BeginGPURenderPass(command(), colors.data(), kind ? 4 : 1, &depth));
            SDL_BindGPUGraphicsPipeline(pass, pipeline(kind, kind ? 1 : t.samples));
            SDL_PushGPUVertexUniformData(command(), 0, uniforms.data(), sizeof(uniforms));
            uint32_t offset = 0;
            for (size_t i = 0; i < mMeshes.size(); ++i) {
                auto &m = mMeshes[i];
                if (m.instances.empty())
                    continue;
                SDL_GPUBufferBinding bindings[] = {{m.vertices.gpu, 0}, {buffer.gpu, offset}};
                SDL_BindGPUVertexBuffers(pass, 0, bindings, 2);
                SDL_GPUBufferBinding indices{m.indices.gpu, 0};
                SDL_BindGPUIndexBuffer(pass, &indices, SDL_GPU_INDEXELEMENTSIZE_32BIT);
                SDL_DrawGPUIndexedPrimitives(pass, mScene.meshes[i].indices.size(),
                                             m.instances.size(), 0, 0, 0);
                offset += m.instances.size() * 80;
                ++mStats.drawCalls;
            }
            SDL_EndGPURenderPass(pass);
        }
        mStats.instances += mInstances.size() * 2;
        return token(id, t, camera.revision);
    }
    ReadbackTicket readback(FrameToken frame, Product product, Region region) override {
        owner();
        auto &t = target(frame.target);
        if (t.window || frame != t.latest || !frame.submission ||
            frame.sceneRevision != mScene.revision)
            throw std::invalid_argument("Stale or unreadable frame");
        if (t.colorOnly && product != Product::Color && product != Product::ColorAlpha)
            throw std::invalid_argument("UI frame has no scene data products");
        if (region.x >= t.size.width || region.y >= t.size.height)
            throw std::invalid_argument("Invalid readback origin");
        if (!region.width)
            region.width = t.size.width - region.x;
        if (!region.height)
            region.height = t.size.height - region.y;
        if (!region.width || !region.height || region.x >= t.size.width ||
            region.y >= t.size.height || region.width > t.size.width - region.x ||
            region.height > t.size.height - region.y)
            throw std::invalid_argument("Invalid readback region");
        auto it = std::find_if(mReadbacks.begin(), mReadbacks.end(),
                               [](const auto &r) { return !r.ticket; });
        if (it == mReadbacks.end())
            throw std::runtime_error("Readback queue is full");
        auto &r = *it;
        r.frame = frame;
        r.product = product;
        r.size = {region.width, region.height};
        r.canceled = false;
        r.fence.reset();
        r.pitch = (region.width * 4 + 255) & ~255u;
        r.count = product == Product::Segmentation ? 2 : 1;
        uint32_t bytes = r.pitch * region.height * r.count;
        if (bytes > r.capacity) {
            auto *replacement = transfer(bytes, SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD);
            if (r.buffer)
                SDL_ReleaseGPUTransferBuffer(mDevice, r.buffer);
            r.buffer = replacement;
            r.capacity = bytes;
        }
        auto *pass = checked(SDL_BeginGPUCopyPass(command()));
        for (uint32_t i = 0; i < r.count; ++i) {
            auto *image = (product == Product::Color || product == Product::ColorAlpha) ? t.color
                          : product == Product::MetricDepth                             ? t.data[3]
                          : product == Product::ObjectId ? t.data[0]
                                                         : t.data[1 + i];
            SDL_GPUTextureRegion src{image,         0, 0, region.x, region.y, 0, region.width,
                                     region.height, 1};
            SDL_GPUTextureTransferInfo dst{r.buffer, i * r.pitch * region.height, r.pitch / 4,
                                           region.height};
            SDL_DownloadFromGPUTexture(pass, &src, &dst);
        }
        SDL_EndGPUCopyPass(pass);
        r.ticket = allocateId();
        return {r.ticket};
    }
    ReadbackResult poll(ReadbackTicket ticket) override {
        owner();
        auto it = std::find_if(mReadbacks.begin(), mReadbacks.end(),
                               [&](const auto &r) { return r.ticket == ticket.id && r.ticket; });
        if (it == mReadbacks.end())
            throw std::invalid_argument("Unknown readback ticket");
        auto &r = *it;
        ReadbackResult result;
        result.frame = r.frame;
        if (!r.fence || !SDL_QueryGPUFence(mDevice, r.fence.get()))
            return result;
        result.state = r.canceled ? ReadbackState::Canceled : ReadbackState::Ready;
        if (!r.canceled) {
            result.image = {r.product, r.size, {}};
            auto stride = pixelBytes(r.product);
            result.image.pixels.resize(size_t(r.size.width) * r.size.height * stride);
            auto *data = static_cast<const std::byte *>(
                checked(SDL_MapGPUTransferBuffer(mDevice, r.buffer, false)));
            for (uint32_t y = 0; y < r.size.height; ++y)
                for (uint32_t x = 0; x < r.size.width; ++x) {
                    auto *src = data + size_t(y) * r.pitch + x * 4;
                    auto *dst =
                        result.image.pixels.data() + (size_t(y) * r.size.width + x) * stride;
                    std::memcpy(dst, src, r.product == Product::Color ? 3 : 4);
                    if (r.count == 2)
                        std::memcpy(dst + 4, src + size_t(r.pitch) * r.size.height, 4);
                }
            SDL_UnmapGPUTransferBuffer(mDevice, r.buffer);
        }
        r.ticket = 0;
        r.fence.reset();
        return result;
    }
    FrameStats advance() override {
        owner();
        submit();
        auto stats = mStats;
        mStats = {};
        return stats;
    }
    Texture targetTexture(Target id) const override {
        owner();
        textureHandle({id.id | targetBit});
        return {id.id | targetBit};
    }
    Texture uploadTexture(Extent size, std::span<const std::byte> rgba) override {
        owner();
        extent(size);
        if (rgba.size() != size_t(size.width) * size.height * 4)
            throw std::invalid_argument("Invalid RGBA texture bytes");
        auto *tex =
            texture(size, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, SDL_GPU_TEXTUREUSAGE_SAMPLER);
        std::memcpy(mapUpload(rgba.size()), rgba.data(), rgba.size());
        SDL_UnmapGPUTransferBuffer(mDevice, mUpload);
        auto *pass = checked(SDL_BeginGPUCopyPass(command()));
        SDL_GPUTextureTransferInfo src{mUpload, 0, size.width, size.height};
        SDL_GPUTextureRegion dst{tex, 0, 0, 0, 0, 0, size.width, size.height, 1};
        SDL_UploadToGPUTexture(pass, &src, &dst, false);
        SDL_EndGPUCopyPass(pass);
        auto id = allocateId();
        mTextures[id] = tex;
        return {id};
    }
    void destroy(Texture id) override {
        owner();
        auto i = mTextures.find(id.id);
        if (i == mTextures.end())
            throw std::invalid_argument("Unknown UI texture");
        SDL_ReleaseGPUTexture(mDevice, i->second);
        mTextures.erase(i);
    }
    FrameToken renderUi(const UiFrame &frame, Target output) override {
        owner();
        validateUi(frame);
        auto &t = target(output);
        t.colorOnly = true;
        auto *image = t.color;
        auto format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
        if (t.window) {
            checked(SDL_WaitAndAcquireGPUSwapchainTexture(command(), t.window, &image,
                                                          &t.size.width, &t.size.height));
            if (!image)
                return {};
            format = SDL_GetGPUSwapchainTextureFormat(mDevice, t.window);
        } else if (frame.size != t.size)
            throw std::invalid_argument("UI frame extent mismatch");
        uploadBuffer(mUiVertices, SDL_GPU_BUFFERUSAGE_VERTEX, frame.vertices.data(),
                     frame.vertices.size_bytes());
        uploadBuffer(mUiIndices, SDL_GPU_BUFFERUSAGE_INDEX, frame.indices.data(),
                     frame.indices.size_bytes());
        SDL_GPUColorTargetInfo color{};
        color.texture = image;
        color.clear_color = {20.0f / 255, 25.0f / 255, 30.0f / 255, 1};
        color.load_op = SDL_GPU_LOADOP_CLEAR;
        color.store_op = SDL_GPU_STOREOP_STORE;
        auto *pass = checked(SDL_BeginGPURenderPass(command(), &color, 1, nullptr));
        SDL_BindGPUGraphicsPipeline(pass, pipeline(2, 1, format));
        std::array<float, 4> uniform{float(frame.size.width), float(frame.size.height), 0, 0};
        SDL_PushGPUVertexUniformData(command(), 0, uniform.data(), sizeof(uniform));
        if (!frame.vertices.empty() && !frame.indices.empty()) {
            SDL_GPUBufferBinding vb{mUiVertices.gpu, 0}, ib{mUiIndices.gpu, 0};
            SDL_BindGPUVertexBuffers(pass, 0, &vb, 1);
            SDL_BindGPUIndexBuffer(pass, &ib, SDL_GPU_INDEXELEMENTSIZE_32BIT);
            for (auto &c : frame.commands) {
                int x = std::max(0, int(c.clip[0])), y = std::max(0, int(c.clip[1]));
                int right = std::min(int(frame.size.width), int(c.clip[2])),
                    bottom = std::min(int(frame.size.height), int(c.clip[3]));
                if (right <= x || bottom <= y || !c.indexCount)
                    continue;
                SDL_Rect clip{x, y, right - x, bottom - y};
                SDL_SetGPUScissor(pass, &clip);
                SDL_GPUTextureSamplerBinding binding{textureHandle(c.texture), mSampler};
                SDL_BindGPUFragmentSamplers(pass, 0, &binding, 1);
                SDL_DrawGPUIndexedPrimitives(pass, c.indexCount, 1, c.firstIndex, c.vertexOffset,
                                             0);
                ++mStats.drawCalls;
            }
        }
        SDL_EndGPURenderPass(pass);
        return token(output, t, 0);
    }
};
} // namespace
std::unique_ptr<Renderer> makeSdlRenderer(const SdlOptions &options) {
    auto renderer = std::make_unique<SdlRenderer>();
    renderer->initialize(options);
    return renderer;
}
} // namespace mojive
