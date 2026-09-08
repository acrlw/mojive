#include <algorithm>
#include <atomic>
#include <bgfx/bgfx.h>
#include <bgfx/defines.h>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <mojive/backends/bgfx.hpp>
#include <stdexcept>
#include <thread>
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
    size_t vertexCount = 0;
};
struct GpuVertex {
    Vertex vertex;
    std::array<float, 2> uv = {0, 0};
};
struct GpuBatch {
    uint32_t mesh = 0, material = 0;
    std::vector<uint32_t> instances;
};
struct GpuScene {
    SceneStyle style;
    std::vector<bgfx::TextureHandle> textures;
    std::vector<GpuBatch> batches;
    SceneSource source;
    uint64_t sequence = 0;
    std::vector<GpuMesh> meshes;
    std::vector<std::array<float, 32>> instances;
};
struct GpuTarget {
    Scene scene;
    Extent size;
    uint32_t samples = 1;
    bool surface = false, colorOnly = false;
    uint16_t view = 0;
    uint64_t generation = 1;
    uint32_t lastRender = UINT32_MAX, lastReadback = UINT32_MAX;
    FrameToken latest;
    bgfx::SwapChain swapChain;
    bgfx::FrameBufferHandle colorFb = BGFX_INVALID_HANDLE, dataFb = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle color = BGFX_INVALID_HANDLE;
    std::array<bgfx::TextureHandle, 4> data;
};
struct ReadbackSlot {
    uint64_t ticket = 0;
    bool canceled = false;
    uint32_t ready = 0;
    FrameToken frame;
    Product product = Product::Color;
    Extent size;
    uint32_t count = 1;
    std::array<bgfx::TextureHandle, 2> textures = {bgfx::TextureHandle{bgfx::kInvalidHandle},
                                                   bgfx::TextureHandle{bgfx::kInvalidHandle}};
    std::array<std::vector<std::byte>, 2> bytes;
};

class BgfxRenderer final : public Renderer {
    bool mInitialized = false;
    std::thread::id mOwner = std::this_thread::get_id();
    Capabilities mCaps;
    std::unordered_map<uint64_t, GpuScene> mScenes{{0, GpuScene{}}};
    uint64_t mSubmission = 0;
    uint32_t mGpuFrame = 0;
    bool mPendingCommands = false;
    std::array<bgfx::ViewId, 256> mPassOrder{};
    std::array<bool, 256> mUsedPasses{};
    uint16_t mPassCount = 0;
    uint32_t mMainRender = UINT32_MAX;
    void orderPass(bgfx::ViewId view) {
        if (!mUsedPasses[view]) {
            mUsedPasses[view] = true;
            mPassOrder[mPassCount++] = view;
        }
    }
    void flush() {
        // bgfx sorts views numerically unless explicitly remapped. Preserve API
        // dependency order even when a sampled target was allocated later.
        auto count = mPassCount;
        for (bgfx::ViewId id = 0; id < mPassOrder.size(); ++id)
            if (!mUsedPasses[id])
                mPassOrder[count++] = id;
        bgfx::setViewOrder(0, mPassOrder.size(), mPassOrder.data());
        mGpuFrame = bgfx::frame();
        mPendingCommands = false;
        mPassCount = 0;
        mUsedPasses.fill(false);
    }
    void beginTarget(GpuTarget &target) {
        if (target.lastRender == mGpuFrame || target.lastReadback == mGpuFrame)
            flush();
        target.lastRender = mGpuFrame;
        mPendingCommands = true;
    }
    std::unordered_map<uint64_t, GpuTarget> mTargets;
    std::unordered_map<uint64_t, bgfx::TextureHandle> mTextures;
    std::vector<ReadbackSlot> mReadbacks;
    std::array<bool, 12> mViews{};
    FrameStats mStats;
    bgfx::VertexLayout mVertices, mUiVertices;
    std::vector<uint32_t> mDrawIndices;
    bgfx::UniformHandle mMaterial = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle mWhite = BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle mColorProgram = BGFX_INVALID_HANDLE, mDataProgram = BGFX_INVALID_HANDLE,
                        mUiProgram = BGFX_INVALID_HANDLE;
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
        for (auto texture : scene.textures)
            bgfx::destroy(texture);
        for (auto &mesh : scene.meshes) {
            if (bgfx::isValid(mesh.vertices))
                bgfx::destroy(mesh.vertices);
            if (bgfx::isValid(mesh.indices))
                bgfx::destroy(mesh.indices);
        }
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
        auto memory = bgfx::alloc(static_cast<uint32_t>(length) + 1);
        stream.read(reinterpret_cast<char *>(memory->data), length);
        memory->data[length] = 0;
        auto handle = bgfx::createShader(memory);
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
    void releaseTarget(GpuTarget &t) {
        if (bgfx::isValid(t.colorFb))
            bgfx::destroy(t.colorFb);
        if (bgfx::isValid(t.dataFb))
            bgfx::destroy(t.dataFb);
        t.colorFb = t.dataFb = BGFX_INVALID_HANDLE;
    }
    void allocateTarget(GpuTarget &t) {
        uint64_t flags = BGFX_TEXTURE_RT | sampler;
        uint64_t msaa = t.samples == 4 ? BGFX_TEXTURE_RT_MSAA_X4 : 0;
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
            for (size_t i = 0; i < 4; ++i)
                t.data[i] =
                    texture(i == 3 ? bgfx::TextureFormat::R32F : bgfx::TextureFormat::RGBA8, flags);
            auto dataDepth = texture(bgfx::TextureFormat::D32F, flags | BGFX_TEXTURE_RT_WRITE_ONLY);
            bgfx::TextureHandle attachments[] = {t.data[0], t.data[1], t.data[2], t.data[3],
                                                 dataDepth};
            t.dataFb = bgfx::createFrameBuffer(5, attachments, true);
            if (!bgfx::isValid(t.colorFb) || !bgfx::isValid(t.dataFb))
                throw std::runtime_error("Cannot create framebuffer");
            orphaned.clear();
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
        for (auto texture : slot.textures)
            if (bgfx::isValid(texture))
                bgfx::destroy(texture);
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
        init.profile = true;
        init.swapChain.maxFrameLatency = 2;
        init.swapChain.nwh = options.window.handle;
        init.swapChain.ndt = options.window.display;
        init.swapChain.width = options.window.handle ? options.window.size.width : 0;
        init.swapChain.height = options.window.handle ? options.window.size.height : 0;
        mSwapChain = init.swapChain;
        mWindowSize = options.window.size;
        mHasWindow = options.window.handle != nullptr;
        if (!bgfx::init(init)) {
            runtimeActive.clear();
            throw std::runtime_error("Cannot initialize native renderer");
        }
        mInitialized = true;
        const auto *caps = bgfx::getCaps();
        mCaps.backend = bgfx::getRendererName(caps->rendererType);
        mCaps.device = std::to_string(caps->vendorId) + ":" + std::to_string(caps->deviceId);
        mCaps.maxTextureSize = std::min<uint32_t>(caps->limits.maxTextureSize, UINT16_MAX);
        mCaps.readback = bgfx::isTextureValid(0, false, 1, bgfx::TextureFormat::RGBA8,
                                              BGFX_TEXTURE_READ_BACK | BGFX_TEXTURE_BLIT_DST);
        mCaps.instancing = caps->limits.maxInstanceData >= 5;
        mCaps.multipleScenes = true;
        mCaps.multipleWindows = (caps->supported & BGFX_CAPS_SWAP_CHAIN) != 0;
        auto rt = [&](bgfx::TextureFormat::Enum f) {
            return (caps->formats[f] & BGFX_CAPS_FORMAT_TEXTURE_FRAMEBUFFER) != 0;
        };
        mCaps.integerTarget = rt(bgfx::TextureFormat::R32U);
        mCaps.signedPairTarget = rt(bgfx::TextureFormat::RG32I);
        mCaps.floatTarget = rt(bgfx::TextureFormat::R32F);
        if (!mCaps.readback || !mCaps.instancing || !mCaps.floatTarget ||
            caps->limits.maxFBAttachments < 5)
            throw std::runtime_error("Device lacks a required probe capability");
        mVertices.begin()
            .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::Normal, 3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
            .end();
        mUiVertices.begin()
            .add(bgfx::Attrib::Position, 2, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
            .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
            .end();
        mColorProgram = program(options.shaderDirectory, "vs_scene", "fs_color");
        mDataProgram = program(options.shaderDirectory, "vs_scene", "fs_data");
        mUiProgram = program(options.shaderDirectory, "vs_ui", "fs_ui");
        mImageSampler = bgfx::createUniform("s_image", bgfx::UniformType::Sampler);
        mMaterial = bgfx::createUniform("u_material", bgfx::UniformType::Vec4);
        const uint32_t white = 0xffffffff;
        mWhite = bgfx::createTexture2D(1, 1, false, 1, bgfx::TextureFormat::RGBA8, 0,
                                       bgfx::copy(&white, 4));
        bgfx::setPaletteColor(0, uint32_t{0});
        bgfx::setPaletteColor(1, 0xffffffff);
    }
    ~BgfxRenderer() override {
        if (!mInitialized)
            return;
        // Pending readback destinations must survive until the render thread has stopped.
        for (auto &[id, t] : mTargets)
            releaseTarget(t);
        for (auto &[id, value] : mScenes)
            releaseScene(value);
        for (auto &r : mReadbacks)
            releaseSlot(r);
        for (auto &[id, t] : mTextures)
            bgfx::destroy(t);
        if (bgfx::isValid(mColorProgram))
            bgfx::destroy(mColorProgram);
        if (bgfx::isValid(mDataProgram))
            bgfx::destroy(mDataProgram);
        if (bgfx::isValid(mUiProgram))
            bgfx::destroy(mUiProgram);
        if (bgfx::isValid(mImageSampler))
            bgfx::destroy(mImageSampler);
        if (bgfx::isValid(mMaterial))
            bgfx::destroy(mMaterial);
        if (bgfx::isValid(mWhite))
            bgfx::destroy(mWhite);
        bgfx::shutdown();
        runtimeActive.clear();
    }
    const Capabilities &capabilities() const override {
        return mCaps;
    }
    void configure(Scene id, const SceneStyle &style) override {
        owner();
        for (float value : style.background)
            if (!std::isfinite(value) || value < 0 || value > 1)
                throw std::invalid_argument("Background must contain normalized finite channels");
        if (style.wireframe)
            throw std::invalid_argument("Wireframe is unavailable");
        scene(id).style = style;
    }
    GpuScene uploadScene(const SceneSource &source) {
        validateScene(source);
        GpuScene result;
        result.source = source;
        result.instances.resize(source.instances.size());
        result.meshes.resize(source.meshes.size());
        try {
            for (size_t i = 0; i < source.meshes.size(); ++i) {
                const auto &input = source.meshes[i];
                auto &mesh = result.meshes[i];
                std::vector<GpuVertex> vertices(input.vertices.size());
                for (size_t v = 0; v < vertices.size(); ++v)
                    vertices[v] = {input.vertices[v], input.texcoords.empty()
                                                          ? std::array<float, 2>{0, 0}
                                                          : input.texcoords[v]};
                mesh.vertices = bgfx::createDynamicVertexBuffer(
                    bgfx::copy(vertices.data(), vertices.size() * sizeof(GpuVertex)), mVertices);
                mesh.indices = bgfx::createIndexBuffer(
                    bgfx::copy(input.indices.data(), input.indices.size() * sizeof(uint32_t)),
                    BGFX_BUFFER_INDEX32);
                if (!bgfx::isValid(mesh.vertices) || !bgfx::isValid(mesh.indices))
                    throw std::runtime_error("Cannot allocate scene mesh");
                mesh.vertexCount = input.vertices.size();
            }
            for (const auto &texture : source.textures) {
                extent(texture.size);
                auto handle = bgfx::createTexture2D(
                    texture.size.width, texture.size.height, texture.mipmaps, 1,
                    bgfx::TextureFormat::RGBA8,
                    BGFX_SAMPLER_MIN_ANISOTROPIC | BGFX_SAMPLER_MAG_ANISOTROPIC,
                    bgfx::copy(texture.rgba.data(), texture.rgba.size()));
                if (!bgfx::isValid(handle))
                    throw std::runtime_error("Cannot allocate scene texture");
                result.textures.push_back(handle);
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
                std::copy(input.color.begin(), input.color.end(), data.begin() + 16);
                data[28] = data[29] = 1;
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
        if (mPendingCommands)
            flush();
        for (auto &[targetId, target] : mTargets) {
            if (target.scene == id) {
                cancel(Target{targetId});
                target.latest = {};
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
        for (size_t i = 0; i < current.instances.size(); ++i) {
            std::copy(frame.transforms[i].begin(), frame.transforms[i].end(),
                      current.instances[i].begin());
            if (!frame.colors.empty()) {
                current.source.instances[i].color = frame.colors[i];
                std::copy(frame.colors[i].begin(), frame.colors[i].end(),
                          current.instances[i].begin() + 16);
            }
            if (!frame.texcoords.empty())
                std::copy(frame.texcoords[i].begin(), frame.texcoords[i].end(),
                          current.instances[i].begin() + 28);
        }
    }
    void updateMesh(uint32_t index, std::span<const Vertex> vertices) override {
        updateMesh(Scene{}, index, vertices);
    }
    void updateMesh(Scene id, uint32_t index, std::span<const Vertex> vertices) override {
        owner();
        auto &current = scene(id);
        if (index >= current.meshes.size() || vertices.size() != current.meshes[index].vertexCount)
            throw std::invalid_argument("Dynamic mesh topology changed");
        if (mPendingCommands)
            flush();
        std::vector<GpuVertex> upload(vertices.size());
        const auto &uv = current.source.meshes[index].texcoords;
        for (size_t i = 0; i < vertices.size(); ++i)
            upload[i] = {vertices[i], uv.empty() ? std::array<float, 2>{0, 0} : uv[i]};
        bgfx::update(current.meshes[index].vertices, 0,
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
        if (samples != 1 && samples != 4)
            throw std::invalid_argument("Probe supports 1x or 4x MSAA");
        auto slot = std::find(mViews.begin(), mViews.end(), false);
        if (slot == mViews.end())
            throw std::runtime_error("Render target capacity reached");
        GpuTarget t;
        t.size = size;
        t.samples = samples;
        t.scene = sceneId;
        t.view = std::distance(mViews.begin(), slot) * 4;
        allocateTarget(t);
        *slot = true;
        Target id{allocateId()};
        mTargets.emplace(id.id, std::move(t));
        return id;
    }
    Target createSurface(NativeWindow window) override {
        owner();
        extent(window.size);
        if (!window.handle || !mCaps.multipleWindows)
            throw std::invalid_argument("Native surface unavailable");
        auto slot = std::find(mViews.begin(), mViews.end(), false);
        if (slot == mViews.end())
            throw std::runtime_error("Render target capacity reached");
        GpuTarget t;
        t.size = window.size;
        t.surface = true;
        t.view = std::distance(mViews.begin(), slot) * 4;
        auto &surface = t.swapChain;
        surface.nwh = window.handle;
        surface.ndt = window.display;
        surface.width = window.size.width;
        surface.height = window.size.height;
        surface.maxFrameLatency = 2;
        t.colorFb = bgfx::createFrameBuffer(surface);
        if (!bgfx::isValid(t.colorFb))
            throw std::runtime_error("Cannot create native surface");
        *slot = true;
        Target id{allocateId()};
        mTargets.emplace(id.id, std::move(t));
        return id;
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
        mViews[t.view / 4] = false;
        mTargets.erase(id.id);
    }
    FrameToken render(Target id, const CameraView &camera) override {
        owner();
        validateCamera(camera);
        auto &t = target(id);
        const auto &current = scene(t.scene);
        if (t.surface)
            throw std::invalid_argument(
                "Render scenes to an offscreen target, then present its texture");
        beginTarget(t);
        t.colorOnly = false;
        float clear_depth[4] = {camera.farPlane, 0, 0, 1};
        bgfx::setPaletteColor(2 + t.view / 4, clear_depth);
        auto view = columnMajor(camera.view), projection = camera.projection;
        if (!bgfx::getCaps()->homogeneousDepth)
            for (size_t c = 0; c < 4; ++c)
                projection[8 + c] = (projection[8 + c] + projection[12 + c]) * 0.5f;
        projection = columnMajor(projection);
        for (uint16_t pass = 0; pass < 2; ++pass) {
            orderPass(t.view + pass);
            bgfx::setViewRect(t.view + pass, 0, 0, t.size.width, t.size.height);
            bgfx::setViewFrameBuffer(t.view + pass, pass ? t.dataFb : t.colorFb);
            bgfx::setViewTransform(t.view + pass, view.data(), projection.data());
            if (pass)
                bgfx::setViewClear(t.view + pass, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 1.0f, 0, 0,
                                   1, 1, 2 + t.view / 4);
            else
                bgfx::setViewClear(t.view + pass, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH,
                                   (uint32_t(current.style.background[0] * 255) << 24) |
                                       (uint32_t(current.style.background[1] * 255) << 16) |
                                       (uint32_t(current.style.background[2] * 255) << 8) |
                                       uint32_t(current.style.background[3] * 255),
                                   1.0f);
            bgfx::touch(t.view + pass);
        }
        for (const auto &batch : current.batches) {
            const auto &mesh = current.meshes[batch.mesh];
            uint32_t count = batch.instances.size();
            if (!count)
                continue;
            constexpr uint16_t stride = 20 * sizeof(float);
            // The portable instance layout has five vec4 slots. Share the three
            // affine rows, then upload color or exact metadata for each product pass.
            for (uint16_t pass = 0; pass < 2; ++pass) {
                mDrawIndices.clear();
                for (auto index : batch.instances) {
                    float alpha = current.source.instances[index].color[3];
                    if (alpha > 0 && (!pass || current.style.transparentIds || alpha >= 1))
                        mDrawIndices.push_back(index);
                }
                count = mDrawIndices.size();
                if (!count)
                    continue;
                if (bgfx::getAvailInstanceDataBuffer(count, stride) != count)
                    throw std::runtime_error("Instance upload capacity exhausted");
                bgfx::InstanceDataBuffer buffer;
                bgfx::allocInstanceDataBuffer(&buffer, count, stride);
                for (size_t i = 0; i < count; ++i) {
                    auto *destination = buffer.data + i * stride;
                    const auto &source = current.instances[mDrawIndices[i]];
                    std::memcpy(destination, source.data(), 12 * sizeof(float));
                    if (pass)
                        std::memcpy(destination + 12 * sizeof(float), source.data() + 20,
                                    8 * sizeof(float));
                    else {
                        std::memcpy(destination + 12 * sizeof(float), source.data() + 16,
                                    4 * sizeof(float));
                        std::memcpy(destination + 16 * sizeof(float), source.data() + 28,
                                    4 * sizeof(float));
                    }
                }
                mStats.uploadBytes += count * stride;
                bgfx::setVertexBuffer(0, mesh.vertices);
                bgfx::setIndexBuffer(mesh.indices);
                bgfx::setInstanceDataBuffer(&buffer);
                uint64_t state = BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_WRITE_Z |
                                 BGFX_STATE_DEPTH_TEST_LESS;
                if (!pass && t.samples > 1)
                    state |= BGFX_STATE_MSAA;
                if (!pass) {
                    const auto material = current.source.materials.empty()
                                              ? Material{-1, 0, 0, 0}
                                              : current.source.materials[batch.material];
                    const float values[4] = {material.emission, material.specular,
                                             material.shininess,
                                             current.source.linearColors ? 1.0f : 0.0f};
                    bgfx::setUniform(mMaterial, values);
                    bgfx::setTexture(0, mImageSampler,
                                     current.style.textures && material.texture >= 0
                                         ? current.textures[material.texture]
                                         : mWhite);

                    state |= BGFX_STATE_BLEND_ALPHA;
                }
                bgfx::setState(state);
                bgfx::submit(t.view + pass, pass ? mDataProgram : mColorProgram);
                ++mStats.drawCalls;
                mStats.instances += count;
            }
        }
        t.latest = {
            id,           t.generation, current.source.revision, current.sequence, camera.revision,
            ++mSubmission};
        return t.latest;
    }
    ReadbackTicket readback(FrameToken frame, Product product, Region region) override {
        owner();
        auto &t = target(frame.target);
        if (t.surface ||
            (t.colorOnly && product != Product::Color && product != Product::ColorAlpha))
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
                     bgfx::isValid(r.textures[0]);
        if (!reuse) {
            releaseSlot(r);
            r.textures = {bgfx::TextureHandle{bgfx::kInvalidHandle},
                          bgfx::TextureHandle{bgfx::kInvalidHandle}};
        }
        r.ticket = allocateId();
        r.canceled = false;
        r.frame = frame;
        r.product = product;
        r.size = {region.width, region.height};
        r.count = product == Product::Segmentation ? 2 : 1;
        for (size_t i = 0; i < r.count; ++i) {
            auto format = product == Product::MetricDepth ? bgfx::TextureFormat::R32F
                                                          : bgfx::TextureFormat::RGBA8;
            if (!reuse)
                r.textures[i] =
                    bgfx::createTexture2D(r.size.width, r.size.height, false, 1, format,
                                          BGFX_TEXTURE_BLIT_DST | BGFX_TEXTURE_READ_BACK | sampler);
            r.bytes[i].resize(size_t(r.size.width) * r.size.height * 4);
            bgfx::TextureHandle source = t.color;
            if (product == Product::ObjectId)
                source = t.data[0];
            if (product == Product::Segmentation)
                source = t.data[i + 1];
            if (product == Product::MetricDepth)
                source = t.data[3];
            uint32_t y = bgfx::getCaps()->originBottomLeft
                             ? t.size.height - region.y - region.height
                             : region.y;
            bgfx::TextureRegion dst{}, src{};
            dst.init(r.textures[i]);
            src.init(source, region.x, y, region.width, region.height);
            orderPass(t.view + 2);
            bgfx::blit(t.view + 2, dst, src);
            r.ready = bgfx::read(dst, r.bytes[i].data());
        }
        t.lastReadback = mGpuFrame;
        mPendingCommands = true;
        return {r.ticket};
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
        if (!it->canceled) {
            auto &image = result.image;
            image.product = it->product;
            image.size = it->size;
            auto stride = pixelBytes(image.product);
            image.pixels.resize(size_t(image.size.width) * image.size.height * stride);
            bool flip = bgfx::getCaps()->originBottomLeft;
            for (size_t y = 0; y < image.size.height; ++y)
                for (size_t x = 0; x < image.size.width; ++x) {
                    size_t from =
                        ((flip ? image.size.height - 1 - y : y) * image.size.width + x) * 4;
                    auto *destination = image.pixels.data() + (y * image.size.width + x) * stride;
                    if (image.product == Product::Color)
                        std::memcpy(destination, it->bytes[0].data() + from, 3);
                    else if (image.product == Product::ColorAlpha)
                        std::memcpy(destination, it->bytes[0].data() + from, 4);
                    else if (image.product == Product::MetricDepth)
                        std::memcpy(destination, it->bytes[0].data() + from, 4);
                    else
                        for (size_t part = 0; part < it->count; ++part) {
                            uint32_t value = word(it->bytes[part].data() + from);
                            std::memcpy(destination + part * 4, &value, 4);
                        }
                }
        }
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
        auto texture =
            bgfx::createTexture2D(size.width, size.height, false, 1, bgfx::TextureFormat::RGBA8,
                                  sampler, bgfx::copy(rgba.data(), rgba.size()));
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
            bgfx::reset(BGFX_RESET_NONE, &mSwapChain);
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
            token = t.latest = {output, t.generation, 0, 0, 0, ++mSubmission};
        }
        if (!output.id) {
            if (mMainRender == mGpuFrame)
                flush();
            mMainRender = mGpuFrame;
        }
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
