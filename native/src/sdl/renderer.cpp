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
std::atomic<uint64_t> next_id{1};
constexpr uint64_t target_bit = uint64_t{1} << 63;
uint64_t allocate_id() {
    return next_id.fetch_add(1, std::memory_order_relaxed);
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
    SDL_GPUTexture *color = nullptr, *msaa = nullptr, *depth = nullptr, *data_depth = nullptr;
    std::array<SDL_GPUTexture *, 4> data{};
    FrameToken latest;
    bool color_only = false;
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
    SDL_GPUDevice *device_ = nullptr;
    bool video_ = false;
    SDL_GPUCommandBuffer *command_ = nullptr;
    std::thread::id owner_ = std::this_thread::get_id();
    Capabilities caps_;
    SceneSource scene_;
    uint64_t sequence_ = 0, submission_ = 0;
    std::string shader_source_;
    std::vector<MeshData> meshes_;
    std::vector<std::array<float, 28>> instances_;
    std::vector<float> packed_;
    Buffer instance_color_, instance_data_, ui_vertices_, ui_indices_;
    SDL_GPUTransferBuffer *upload_ = nullptr;
    uint32_t upload_capacity_ = 0;
    std::unordered_map<uint64_t, TargetData> targets_;
    std::unordered_map<uint64_t, SDL_GPUTexture *> textures_;
    std::array<Readback, 8> readbacks_;
    std::deque<Fence> in_flight_;
    std::unordered_map<uint64_t, SDL_GPUGraphicsPipeline *> pipelines_;
    SDL_GPUSampler *sampler_ = nullptr;
    FrameStats stats_;
    void owner() const {
        if (std::this_thread::get_id() != owner_)
            throw std::logic_error("Renderer called outside its owner thread");
    }
    SDL_GPUCommandBuffer *command() {
        if (!command_)
            command_ = checked(SDL_AcquireGPUCommandBuffer(device_));
        return command_;
    }
    void submit() {
        if (!command_)
            return;
        while (!in_flight_.empty() && SDL_QueryGPUFence(device_, in_flight_.front().get()))
            in_flight_.pop_front();
        // Bound outstanding GPU work even when there is no swapchain or readback.
        if (in_flight_.size() >= 2) {
            auto *first = in_flight_.front().get();
            checked(SDL_WaitForGPUFences(device_, true, &first, 1));
            in_flight_.pop_front();
        }
        auto *raw = checked(SDL_SubmitGPUCommandBufferAndAcquireFence(command_));
        command_ = nullptr;
        Fence fence(raw, [this](auto *f) { SDL_ReleaseGPUFence(device_, f); });
        for (auto &r : readbacks_)
            if (r.ticket && !r.fence)
                r.fence = fence;
        in_flight_.push_back(fence);
    }

    void idle() {
        submit();
        checked(SDL_WaitForGPUIdle(device_));
        in_flight_.clear();
    }
    void extent(Extent s) const {
        if (!s.width || !s.height || s.width > caps_.max_texture_size ||
            s.height > caps_.max_texture_size)
            throw std::invalid_argument("Unsupported render target size");
    }
    TargetData &target(Target id) {
        auto i = targets_.find(id.id);
        if (i == targets_.end())
            throw std::invalid_argument("Unknown render target");
        return i->second;
    }
    SDL_GPUTransferBuffer *transfer(uint32_t bytes, SDL_GPUTransferBufferUsage usage) {
        SDL_GPUTransferBufferCreateInfo info{};
        info.usage = usage;
        info.size = bytes;
        return checked(SDL_CreateGPUTransferBuffer(device_, &info));
    }
    void *map_upload(uint32_t size) {
        if (size > upload_capacity_) {
            auto *replacement = transfer(size, SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD);
            if (upload_)
                SDL_ReleaseGPUTransferBuffer(device_, upload_);
            upload_ = replacement;
            upload_capacity_ = size;
        }
        return checked(SDL_MapGPUTransferBuffer(device_, upload_, true));
    }
    void release(Buffer &b) {
        if (b.gpu)
            SDL_ReleaseGPUBuffer(device_, b.gpu);
        b = {};
    }
    void upload_buffer(Buffer &b, SDL_GPUBufferUsageFlags usage, const void *data, uint32_t size) {
        if (!size)
            return;
        if (size > b.size) {
            release(b);
            SDL_GPUBufferCreateInfo info{};
            info.size = size;
            info.usage = usage;
            b.gpu = checked(SDL_CreateGPUBuffer(device_, &info));
            b.size = size;
        }
        std::memcpy(map_upload(size), data, size);
        SDL_UnmapGPUTransferBuffer(device_, upload_);
        auto *pass = checked(SDL_BeginGPUCopyPass(command()));
        SDL_GPUTransferBufferLocation src{upload_, 0};
        SDL_GPUBufferRegion dst{b.gpu, 0, size};
        SDL_UploadToGPUBuffer(pass, &src, &dst, true);
        SDL_EndGPUCopyPass(pass);
        stats_.upload_bytes += size;
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
        if (!SDL_GPUTextureSupportsFormat(device_, format, info.type, usage) ||
            !SDL_GPUTextureSupportsSampleCount(device_, format, info.sample_count))
            throw std::runtime_error("Unsupported SDL texture format/sample count");
        return checked(SDL_CreateGPUTexture(device_, &info));
    }
    void release_target(TargetData &t, bool wait = true) {
        for (auto *p : {t.color, t.msaa, t.depth, t.data_depth})
            if (p)
                SDL_ReleaseGPUTexture(device_, p);
        for (auto *p : t.data)
            if (p)
                SDL_ReleaseGPUTexture(device_, p);
        t.color = t.msaa = t.depth = t.data_depth = nullptr;
        t.data = {};
        if (t.window) {
            if (wait)
                idle();
            SDL_ReleaseWindowFromGPUDevice(device_, t.window);
            SDL_DestroyWindow(t.window);
            t.window = nullptr;
        }
    }
    void allocate_target(TargetData &t) {
        constexpr auto color = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
        t.color = texture(t.size, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, color);
        if (t.samples == 4)
            t.msaa = texture(t.size, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
                             SDL_GPU_TEXTUREUSAGE_COLOR_TARGET, 4);
        t.depth = texture(t.size, SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
                          SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET, t.samples);
        t.data_depth = texture(t.size, SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
                               SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET);
        for (size_t i = 0; i < 4; ++i)
            t.data[i] = texture(t.size,
                                i == 3 ? SDL_GPU_TEXTUREFORMAT_R32_FLOAT
                                       : SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
                                color);
    }
    void cancel(Target id = {}) {
        for (auto &r : readbacks_)
            if (!id.id || r.frame.target == id)
                r.canceled = true;
    }
    SDL_GPUShader *shader(const char *entry, bool vertex, bool ui) {
        SDL_GPUShaderCreateInfo info{};
        info.code = reinterpret_cast<const Uint8 *>(shader_source_.c_str());
        info.code_size = shader_source_.size() + 1;
        info.entrypoint = entry;
        info.format = SDL_GPU_SHADERFORMAT_MSL;
        info.stage = vertex ? SDL_GPU_SHADERSTAGE_VERTEX : SDL_GPU_SHADERSTAGE_FRAGMENT;
        info.num_uniform_buffers = vertex ? 1 : 0;
        info.num_samplers = ui && !vertex ? 1 : 0;
        return checked(SDL_CreateGPUShader(device_, &info));
    }
    SDL_GPUGraphicsPipeline *
    pipeline(int kind, uint32_t samples,
             SDL_GPUTextureFormat format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM) {
        uint64_t key = uint64_t(format) * 100 + samples * 10 + kind;
        if (auto i = pipelines_.find(key); i != pipelines_.end())
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
            SDL_ReleaseGPUShader(device_, vs);
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
        auto *result = SDL_CreateGPUGraphicsPipeline(device_, &info);
        SDL_ReleaseGPUShader(device_, vs);
        SDL_ReleaseGPUShader(device_, fs);
        pipelines_[key] = checked(result);
        return result;
    }
    SDL_GPUTexture *texture_handle(Texture id) const {
        if (id.id & target_bit) {
            auto i = targets_.find(id.id & ~target_bit);
            if (i == targets_.end() || i->second.window)
                throw std::invalid_argument("Expired target texture");
            return i->second.color;
        }
        auto i = textures_.find(id.id);
        if (i == textures_.end())
            throw std::invalid_argument("Unknown UI texture");
        return i->second;
    }
    FrameToken token(Target id, TargetData &t, uint64_t camera) {
        t.latest = {id, t.generation, scene_.revision, sequence_, camera, ++submission_};
        return t.latest;
    }

  public:
    void initialize(const SdlOptions &options) {
        checked(SDL_InitSubSystem(SDL_INIT_VIDEO));
        video_ = true;
        device_ = checked(SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_MSL, false, "metal"));
        caps_ = {
            "SDL3 Metal", SDL_GetGPUDeviceDriver(device_), true, true, true, false, false, true,
            16384};
        caps_.integer_target =
            SDL_GPUTextureSupportsFormat(device_, SDL_GPU_TEXTUREFORMAT_R32_UINT,
                                         SDL_GPU_TEXTURETYPE_2D, SDL_GPU_TEXTUREUSAGE_COLOR_TARGET);
        caps_.signed_pair_target =
            SDL_GPUTextureSupportsFormat(device_, SDL_GPU_TEXTUREFORMAT_R32G32_INT,
                                         SDL_GPU_TEXTURETYPE_2D, SDL_GPU_TEXTUREUSAGE_COLOR_TARGET);
        caps_.device = SDL_GetStringProperty(SDL_GetGPUDeviceProperties(device_),
                                             SDL_PROP_GPU_DEVICE_NAME_STRING, "Unknown GPU");
        std::ifstream file(options.shader_directory + "/sdl.metal");
        if (!file)
            throw std::runtime_error("Cannot open SDL Metal shaders");
        shader_source_.assign(std::istreambuf_iterator<char>(file), {});
        SDL_GPUSamplerCreateInfo sampler{};
        sampler.min_filter = sampler.mag_filter = SDL_GPU_FILTER_LINEAR;
        sampler.address_mode_u = sampler.address_mode_v = sampler.address_mode_w =
            SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
        sampler_ = checked(SDL_CreateGPUSampler(device_, &sampler));
        pipeline(0, 1);
        pipeline(0, 4);
        pipeline(1, 1);
        pipeline(2, 1);
        if (options.window.handle) {
            auto id = create_surface(options.window);
            targets_[0] = targets_.at(id.id);
            targets_.erase(id.id);
        }
    }
    ~SdlRenderer() override {
        if (device_) {
            if (command_)
                SDL_CancelGPUCommandBuffer(command_);
            SDL_WaitForGPUIdle(device_);
            in_flight_.clear();
            for (auto &r : readbacks_) {
                r.fence.reset();
                if (r.buffer)
                    SDL_ReleaseGPUTransferBuffer(device_, r.buffer);
            }
            command_ = nullptr;
            for (auto &[id, t] : targets_)
                release_target(t, false);
            for (auto &[id, t] : textures_)
                SDL_ReleaseGPUTexture(device_, t);
            for (auto &[key, p] : pipelines_)
                SDL_ReleaseGPUGraphicsPipeline(device_, p);
            for (auto &m : meshes_) {
                release(m.vertices);
                release(m.indices);
            }
            release(instance_color_);
            release(instance_data_);
            release(ui_vertices_);
            release(ui_indices_);
            if (upload_)
                SDL_ReleaseGPUTransferBuffer(device_, upload_);
            if (sampler_)
                SDL_ReleaseGPUSampler(device_, sampler_);
            SDL_DestroyGPUDevice(device_);
        }
        if (video_)
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
    }
    const Capabilities &capabilities() const override {
        return caps_;
    }
    void set_scene(const SceneSource &scene) override {
        owner();
        validate_scene(scene);
        cancel();
        for (auto &m : meshes_) {
            release(m.vertices);
            release(m.indices);
        }
        meshes_.clear();
        scene_ = scene;
        instances_.resize(scene.instances.size());
        meshes_.resize(scene.meshes.size());
        for (size_t i = 0; i < meshes_.size(); ++i) {
            auto &m = meshes_[i];
            auto &s = scene.meshes[i];
            upload_buffer(m.vertices, SDL_GPU_BUFFERUSAGE_VERTEX, s.vertices.data(),
                          s.vertices.size() * sizeof(Vertex));
            upload_buffer(m.indices, SDL_GPU_BUFFERUSAGE_INDEX, s.indices.data(),
                          s.indices.size() * 4);
        }
        for (uint32_t i = 0; i < scene.instances.size(); ++i) {
            const auto &s = scene.instances[i];
            meshes_[s.mesh].instances.push_back(i);
            auto &out = instances_[i];
            out = {};
            auto mat = identity();
            std::copy_n(mat.begin(), 12, out.begin());
            std::copy(s.color.begin(), s.color.end(), out.begin() + 12);
            std::array<uint32_t, 3> ids{s.object_id, std::bit_cast<uint32_t>(s.segmentation[0]),
                                        std::bit_cast<uint32_t>(s.segmentation[1])};
            for (size_t j = 0; j < 3; ++j) {
                out[20 + j * 2] = float(ids[j] & 65535);
                out[21 + j * 2] = float(ids[j] >> 16);
            }
        }
        packed_.resize(instances_.size() * 20);
        sequence_ = 0;
        for (auto &[id, t] : targets_)
            t.latest = {};
    }
    void update(const SceneFrame &frame) override {
        owner();
        validate_frame(scene_, frame);
        sequence_ = frame.sequence;
        for (size_t i = 0; i < instances_.size(); ++i)
            std::copy_n(frame.transforms[i].begin(), 12, instances_[i].begin());
    }
    void update_mesh(uint32_t index, std::span<const Vertex> vertices) override {
        owner();
        if (index >= meshes_.size() || vertices.size() != scene_.meshes[index].vertices.size())
            throw std::invalid_argument("Invalid dynamic mesh update");
        upload_buffer(meshes_[index].vertices, SDL_GPU_BUFFERUSAGE_VERTEX, vertices.data(),
                      vertices.size_bytes());
    }
    Target create_target(Extent size, uint32_t samples) override {
        owner();
        extent(size);
        if (samples != 1 && samples != 4)
            throw std::invalid_argument("Unsupported MSAA count");
        TargetData t;
        t.size = size;
        t.samples = samples;
        try {
            allocate_target(t);
        } catch (...) {
            release_target(t);
            throw;
        }
        auto id = allocate_id();
        targets_.emplace(id, t);
        return {id};
    }
    Target create_surface(NativeWindow window) override {
        owner();
        extent(window.size);
        if (!window.handle)
            throw std::invalid_argument("Missing native window");
        auto props = checked(SDL_CreateProperties());
        checked(SDL_SetPointerProperty(props, SDL_PROP_WINDOW_CREATE_COCOA_WINDOW_POINTER,
                                       window.handle));
        checked(SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_METAL_BOOLEAN, true));
        auto *wrapped = SDL_CreateWindowWithProperties(props);
        SDL_DestroyProperties(props);
        checked(wrapped);
        if (!SDL_ClaimWindowForGPUDevice(device_, wrapped)) {
            SDL_DestroyWindow(wrapped);
            checked(false);
        }
        TargetData t;
        t.window = wrapped;
        t.size = window.size;
        t.color_only = true;
        auto id = allocate_id();
        targets_.emplace(id, t);
        return {id};
    }
    void resize(Target id, Extent size) override {
        owner();
        extent(size);
        auto &t = target(id);
        if (t.size == size)
            return;
        if (t.window)
            throw std::invalid_argument("Resize native surfaces through the platform window");
        cancel(id);
        release_target(t);
        t.size = size;
        ++t.generation;
        t.latest = {};
        allocate_target(t);
    }
    void destroy(Target id) override {
        owner();
        auto &t = target(id);
        cancel(id);
        release_target(t);
        targets_.erase(id.id);
    }
    FrameToken render(Target id, const CameraView &camera) override {
        owner();
        auto &t = target(id);
        if (t.window)
            throw std::invalid_argument("Render the scene to an offscreen target");
        t.color_only = false;
        std::array<float, 32> uniforms;
        std::copy(camera.view.begin(), camera.view.end(), uniforms.begin());
        std::copy(camera.projection.begin(), camera.projection.end(), uniforms.begin() + 16);
        for (int kind = 0; kind < 2; ++kind) {
            size_t at = 0;
            for (const auto &mesh : meshes_)
                for (auto index : mesh.instances) {
                    auto &instance = instances_[index];
                    std::copy_n(instance.begin(), 12, packed_.begin() + at);
                    std::copy_n(instance.begin() + (kind ? 20 : 12), 8, packed_.begin() + at + 12);
                    at += 20;
                }
            auto &buffer = kind ? instance_data_ : instance_color_;
            upload_buffer(buffer, SDL_GPU_BUFFERUSAGE_VERTEX, packed_.data(),
                          packed_.size() * sizeof(float));
            std::array<SDL_GPUColorTargetInfo, 4> colors{};
            for (size_t i = 0; i < (kind ? 4 : 1); ++i) {
                auto &c = colors[i];
                c.texture = kind ? t.data[i] : (t.msaa ? t.msaa : t.color);
                c.load_op = SDL_GPU_LOADOP_CLEAR;
                c.store_op = SDL_GPU_STOREOP_STORE;
                c.clear_color = kind ? (i == 0   ? SDL_FColor{0, 0, 0, 0}
                                        : i == 3 ? SDL_FColor{camera.far_plane, 0, 0, 0}
                                                 : SDL_FColor{1, 1, 1, 1})
                                     : SDL_FColor{32.0f / 255, 38.0f / 255, 47.0f / 255, 1};
                if (!kind && t.msaa) {
                    c.resolve_texture = t.color;
                    c.store_op = SDL_GPU_STOREOP_RESOLVE;
                }
            }
            SDL_GPUDepthStencilTargetInfo depth{};
            depth.texture = kind ? t.data_depth : t.depth;
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
            for (size_t i = 0; i < meshes_.size(); ++i) {
                auto &m = meshes_[i];
                if (m.instances.empty())
                    continue;
                SDL_GPUBufferBinding bindings[] = {{m.vertices.gpu, 0}, {buffer.gpu, offset}};
                SDL_BindGPUVertexBuffers(pass, 0, bindings, 2);
                SDL_GPUBufferBinding indices{m.indices.gpu, 0};
                SDL_BindGPUIndexBuffer(pass, &indices, SDL_GPU_INDEXELEMENTSIZE_32BIT);
                SDL_DrawGPUIndexedPrimitives(pass, scene_.meshes[i].indices.size(),
                                             m.instances.size(), 0, 0, 0);
                offset += m.instances.size() * 80;
                ++stats_.draw_calls;
            }
            SDL_EndGPURenderPass(pass);
        }
        stats_.instances += instances_.size() * 2;
        return token(id, t, camera.revision);
    }
    ReadbackTicket readback(FrameToken frame, Product product, Region region) override {
        owner();
        auto &t = target(frame.target);
        if (t.window || frame != t.latest || !frame.submission ||
            frame.scene_revision != scene_.revision)
            throw std::invalid_argument("Stale or unreadable frame");
        if (t.color_only && product != Product::Color)
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
        auto it = std::find_if(readbacks_.begin(), readbacks_.end(),
                               [](const auto &r) { return !r.ticket; });
        if (it == readbacks_.end())
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
                SDL_ReleaseGPUTransferBuffer(device_, r.buffer);
            r.buffer = replacement;
            r.capacity = bytes;
        }
        auto *pass = checked(SDL_BeginGPUCopyPass(command()));
        for (uint32_t i = 0; i < r.count; ++i) {
            auto *image = product == Product::Color         ? t.color
                          : product == Product::MetricDepth ? t.data[3]
                          : product == Product::ObjectId    ? t.data[0]
                                                            : t.data[1 + i];
            SDL_GPUTextureRegion src{image,         0, 0, region.x, region.y, 0, region.width,
                                     region.height, 1};
            SDL_GPUTextureTransferInfo dst{r.buffer, i * r.pitch * region.height, r.pitch / 4,
                                           region.height};
            SDL_DownloadFromGPUTexture(pass, &src, &dst);
        }
        SDL_EndGPUCopyPass(pass);
        r.ticket = allocate_id();
        return {r.ticket};
    }
    ReadbackResult poll(ReadbackTicket ticket) override {
        owner();
        auto it = std::find_if(readbacks_.begin(), readbacks_.end(),
                               [&](const auto &r) { return r.ticket == ticket.id && r.ticket; });
        if (it == readbacks_.end())
            throw std::invalid_argument("Unknown readback ticket");
        auto &r = *it;
        ReadbackResult result;
        result.frame = r.frame;
        if (!r.fence || !SDL_QueryGPUFence(device_, r.fence.get()))
            return result;
        result.state = r.canceled ? ReadbackState::Canceled : ReadbackState::Ready;
        if (!r.canceled) {
            result.image = {r.product, r.size, {}};
            auto stride = pixel_bytes(r.product);
            result.image.pixels.resize(size_t(r.size.width) * r.size.height * stride);
            auto *data = static_cast<const std::byte *>(
                checked(SDL_MapGPUTransferBuffer(device_, r.buffer, false)));
            for (uint32_t y = 0; y < r.size.height; ++y)
                for (uint32_t x = 0; x < r.size.width; ++x) {
                    auto *src = data + size_t(y) * r.pitch + x * 4;
                    auto *dst =
                        result.image.pixels.data() + (size_t(y) * r.size.width + x) * stride;
                    std::memcpy(dst, src, r.product == Product::Color ? 3 : 4);
                    if (r.count == 2)
                        std::memcpy(dst + 4, src + size_t(r.pitch) * r.size.height, 4);
                }
            SDL_UnmapGPUTransferBuffer(device_, r.buffer);
        }
        r.ticket = 0;
        r.fence.reset();
        return result;
    }
    FrameStats advance() override {
        owner();
        submit();
        auto stats = stats_;
        stats_ = {};
        return stats;
    }
    Texture target_texture(Target id) const override {
        owner();
        texture_handle({id.id | target_bit});
        return {id.id | target_bit};
    }
    Texture upload_texture(Extent size, std::span<const std::byte> rgba) override {
        owner();
        extent(size);
        if (rgba.size() != size_t(size.width) * size.height * 4)
            throw std::invalid_argument("Invalid RGBA texture bytes");
        auto *tex =
            texture(size, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, SDL_GPU_TEXTUREUSAGE_SAMPLER);
        std::memcpy(map_upload(rgba.size()), rgba.data(), rgba.size());
        SDL_UnmapGPUTransferBuffer(device_, upload_);
        auto *pass = checked(SDL_BeginGPUCopyPass(command()));
        SDL_GPUTextureTransferInfo src{upload_, 0, size.width, size.height};
        SDL_GPUTextureRegion dst{tex, 0, 0, 0, 0, 0, size.width, size.height, 1};
        SDL_UploadToGPUTexture(pass, &src, &dst, false);
        SDL_EndGPUCopyPass(pass);
        auto id = allocate_id();
        textures_[id] = tex;
        return {id};
    }
    void destroy(Texture id) override {
        owner();
        auto i = textures_.find(id.id);
        if (i == textures_.end())
            throw std::invalid_argument("Unknown UI texture");
        SDL_ReleaseGPUTexture(device_, i->second);
        textures_.erase(i);
    }
    FrameToken render_ui(const UiFrame &frame, Target output) override {
        owner();
        validate_ui(frame);
        auto &t = target(output);
        t.color_only = true;
        auto *image = t.color;
        auto format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
        if (t.window) {
            checked(SDL_WaitAndAcquireGPUSwapchainTexture(command(), t.window, &image,
                                                          &t.size.width, &t.size.height));
            if (!image)
                return {};
            format = SDL_GetGPUSwapchainTextureFormat(device_, t.window);
        } else if (frame.size != t.size)
            throw std::invalid_argument("UI frame extent mismatch");
        upload_buffer(ui_vertices_, SDL_GPU_BUFFERUSAGE_VERTEX, frame.vertices.data(),
                      frame.vertices.size_bytes());
        upload_buffer(ui_indices_, SDL_GPU_BUFFERUSAGE_INDEX, frame.indices.data(),
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
            SDL_GPUBufferBinding vb{ui_vertices_.gpu, 0}, ib{ui_indices_.gpu, 0};
            SDL_BindGPUVertexBuffers(pass, 0, &vb, 1);
            SDL_BindGPUIndexBuffer(pass, &ib, SDL_GPU_INDEXELEMENTSIZE_32BIT);
            for (auto &c : frame.commands) {
                int x = std::max(0, int(c.clip[0])), y = std::max(0, int(c.clip[1]));
                int right = std::min(int(frame.size.width), int(c.clip[2])),
                    bottom = std::min(int(frame.size.height), int(c.clip[3]));
                if (right <= x || bottom <= y || !c.index_count)
                    continue;
                SDL_Rect clip{x, y, right - x, bottom - y};
                SDL_SetGPUScissor(pass, &clip);
                SDL_GPUTextureSamplerBinding binding{texture_handle(c.texture), sampler_};
                SDL_BindGPUFragmentSamplers(pass, 0, &binding, 1);
                SDL_DrawGPUIndexedPrimitives(pass, c.index_count, 1, c.first_index, c.vertex_offset,
                                             0);
                ++stats_.draw_calls;
            }
        }
        SDL_EndGPURenderPass(pass);
        return token(output, t, 0);
    }
};
} // namespace
std::unique_ptr<Renderer> make_sdl_renderer(const SdlOptions &options) {
    auto renderer = std::make_unique<SdlRenderer>();
    renderer->initialize(options);
    return renderer;
}
} // namespace mojive
