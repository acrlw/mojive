#include <algorithm>
#include <atomic>
#include <bgfx/bgfx.h>
#include <bgfx/defines.h>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <mojive/backends/bgfx.hpp>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace mojive {
namespace {
std::atomic_flag runtime_active = ATOMIC_FLAG_INIT;
std::atomic<uint64_t> resource_sequence{1};
uint64_t allocate_id() {
    return resource_sequence.fetch_add(1, std::memory_order_relaxed);
}
constexpr uint64_t sampler =
    BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP | BGFX_SAMPLER_MIN_POINT | BGFX_SAMPLER_MAG_POINT;
constexpr uint64_t target_texture_bit = uint64_t{1} << 63;
Matrix column_major(const Matrix &input) {
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
    std::vector<uint32_t> instances;
    size_t vertex_count = 0;
};
struct GpuTarget {
    Extent size;
    uint32_t samples = 1;
    bool surface = false, color_only = false;
    uint16_t view = 0;
    uint64_t generation = 1;
    uint32_t last_render = UINT32_MAX, last_readback = UINT32_MAX;
    FrameToken latest;
    bgfx::FrameBufferHandle color_fb = BGFX_INVALID_HANDLE, data_fb = BGFX_INVALID_HANDLE;
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
    bool initialized_ = false;
    std::thread::id owner_ = std::this_thread::get_id();
    Capabilities caps_;
    SceneSource scene_;
    uint64_t sequence_ = 0, submission_ = 0;
    uint32_t gpu_frame_ = 0;
    bool pending_commands_ = false;
    void flush() {
        gpu_frame_ = bgfx::frame();
        pending_commands_ = false;
    }
    void begin_target(GpuTarget &target) {
        if (target.last_render == gpu_frame_ || target.last_readback == gpu_frame_)
            flush();
        target.last_render = gpu_frame_;
        pending_commands_ = true;
    }
    std::vector<GpuMesh> meshes_;
    std::vector<std::array<float, 28>> instances_;
    std::unordered_map<uint64_t, GpuTarget> targets_;
    std::unordered_map<uint64_t, bgfx::TextureHandle> textures_;
    std::vector<ReadbackSlot> readbacks_;
    std::array<bool, 12> views_{};
    FrameStats stats_;
    bgfx::VertexLayout vertices_, ui_vertices_;
    bgfx::ProgramHandle color_program_ = BGFX_INVALID_HANDLE, data_program_ = BGFX_INVALID_HANDLE,
                        ui_program_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle image_sampler_ = BGFX_INVALID_HANDLE;
    Extent window_size_;
    bool has_window_ = false;
    bgfx::SwapChain swap_chain_;

    void owner() const {
        if (std::this_thread::get_id() != owner_)
            throw std::logic_error("Renderer called outside its owner thread");
    }
    GpuTarget &target(Target id) {
        auto it = targets_.find(id.id);
        if (it == targets_.end())
            throw std::invalid_argument("Unknown render target");
        return it->second;
    }
    void extent(Extent size) const {
        if (!size.width || !size.height || size.width > caps_.max_texture_size ||
            size.height > caps_.max_texture_size)
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
    void release_target(GpuTarget &t) {
        if (bgfx::isValid(t.color_fb))
            bgfx::destroy(t.color_fb);
        if (bgfx::isValid(t.data_fb))
            bgfx::destroy(t.data_fb);
        t.color_fb = t.data_fb = BGFX_INVALID_HANDLE;
    }
    void allocate_target(GpuTarget &t) {
        uint64_t flags = BGFX_TEXTURE_RT | sampler;
        uint64_t msaa = t.samples == 4 ? BGFX_TEXTURE_RT_MSAA_X4 : 0;
        if (!bgfx::isTextureValid(0, false, 1, bgfx::TextureFormat::RGBA8, flags | msaa))
            throw std::runtime_error("Requested color/MSAA target unsupported");
        auto texture = [&](bgfx::TextureFormat::Enum format, uint64_t options) {
            auto h = bgfx::createTexture2D(t.size.width, t.size.height, false, 1, format, options);
            if (!bgfx::isValid(h))
                throw std::runtime_error("Cannot allocate render target texture");
            return h;
        };
        t.color = texture(bgfx::TextureFormat::RGBA8, flags | msaa);
        auto color_depth =
            texture(bgfx::TextureFormat::D32F, flags | msaa | BGFX_TEXTURE_RT_WRITE_ONLY);
        bgfx::TextureHandle color_attachments[] = {t.color, color_depth};
        t.color_fb = bgfx::createFrameBuffer(2, color_attachments, true);
        for (size_t i = 0; i < 4; ++i)
            t.data[i] =
                texture(i == 3 ? bgfx::TextureFormat::R32F : bgfx::TextureFormat::RGBA8, flags);
        auto data_depth = texture(bgfx::TextureFormat::D32F, flags | BGFX_TEXTURE_RT_WRITE_ONLY);
        bgfx::TextureHandle attachments[] = {t.data[0], t.data[1], t.data[2], t.data[3],
                                             data_depth};
        t.data_fb = bgfx::createFrameBuffer(5, attachments, true);
        if (!bgfx::isValid(t.color_fb) || !bgfx::isValid(t.data_fb))
            throw std::runtime_error("Cannot create framebuffer");
    }
    void cancel(Target id = {}) {
        for (auto &request : readbacks_)
            if (!id.id || request.frame.target == id)
                request.canceled = true;
    }
    void release_slot(ReadbackSlot &slot) {
        for (auto texture : slot.textures)
            if (bgfx::isValid(texture))
                bgfx::destroy(texture);
    }
    bgfx::TextureHandle texture_handle(Texture texture) const {
        if (texture.id & target_texture_bit) {
            auto it = targets_.find(texture.id & ~target_texture_bit);
            if (it == targets_.end())
                throw std::invalid_argument("Expired target texture");
            return it->second.color;
        }
        auto it = textures_.find(texture.id);
        if (it == textures_.end())
            throw std::invalid_argument("Unknown UI texture");
        return it->second;
    }

  public:
    void initialize(const BgfxOptions &options) {
        if (runtime_active.test_and_set())
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
        swap_chain_ = init.swapChain;
        window_size_ = options.window.size;
        has_window_ = options.window.handle != nullptr;
        if (!bgfx::init(init)) {
            runtime_active.clear();
            throw std::runtime_error("Cannot initialize native renderer");
        }
        initialized_ = true;
        const auto *caps = bgfx::getCaps();
        caps_.backend = bgfx::getRendererName(caps->rendererType);
        caps_.device = std::to_string(caps->vendorId) + ":" + std::to_string(caps->deviceId);
        caps_.max_texture_size = std::min<uint32_t>(caps->limits.maxTextureSize, UINT16_MAX);
        caps_.readback = bgfx::isTextureValid(0, false, 1, bgfx::TextureFormat::RGBA8,
                                              BGFX_TEXTURE_READ_BACK | BGFX_TEXTURE_BLIT_DST);
        caps_.instancing = caps->limits.maxInstanceData >= 5;
        caps_.multiple_windows = (caps->supported & BGFX_CAPS_SWAP_CHAIN) != 0;
        auto rt = [&](bgfx::TextureFormat::Enum f) {
            return (caps->formats[f] & BGFX_CAPS_FORMAT_TEXTURE_FRAMEBUFFER) != 0;
        };
        caps_.integer_target = rt(bgfx::TextureFormat::R32U);
        caps_.signed_pair_target = rt(bgfx::TextureFormat::RG32I);
        caps_.float_target = rt(bgfx::TextureFormat::R32F);
        if (!caps_.readback || !caps_.instancing || !caps_.float_target ||
            caps->limits.maxFBAttachments < 5)
            throw std::runtime_error("Device lacks a required probe capability");
        vertices_.begin()
            .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::Normal, 3, bgfx::AttribType::Float)
            .end();
        ui_vertices_.begin()
            .add(bgfx::Attrib::Position, 2, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
            .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
            .end();
        color_program_ = program(options.shader_directory, "vs_scene", "fs_color");
        data_program_ = program(options.shader_directory, "vs_scene", "fs_data");
        ui_program_ = program(options.shader_directory, "vs_ui", "fs_ui");
        image_sampler_ = bgfx::createUniform("s_image", bgfx::UniformType::Sampler);
        bgfx::setPaletteColor(0, uint32_t{0});
        bgfx::setPaletteColor(1, 0xffffffff);
    }
    ~BgfxRenderer() override {
        if (!initialized_)
            return;
        // Pending readback destinations must survive until the render thread has stopped.
        for (auto &[id, t] : targets_)
            release_target(t);
        for (auto &m : meshes_) {
            bgfx::destroy(m.vertices);
            bgfx::destroy(m.indices);
        }
        for (auto &r : readbacks_)
            release_slot(r);
        for (auto &[id, t] : textures_)
            bgfx::destroy(t);
        if (bgfx::isValid(color_program_))
            bgfx::destroy(color_program_);
        if (bgfx::isValid(data_program_))
            bgfx::destroy(data_program_);
        if (bgfx::isValid(ui_program_))
            bgfx::destroy(ui_program_);
        if (bgfx::isValid(image_sampler_))
            bgfx::destroy(image_sampler_);
        bgfx::shutdown();
        runtime_active.clear();
    }
    const Capabilities &capabilities() const override {
        return caps_;
    }
    void set_scene(const SceneSource &source) override {
        owner();
        validate_scene(source);
        if (pending_commands_)
            flush();
        cancel();
        for (auto &m : meshes_) {
            bgfx::destroy(m.vertices);
            bgfx::destroy(m.indices);
        }
        meshes_.clear();
        scene_ = source;
        instances_.resize(source.instances.size());
        for (const auto &mesh : source.meshes) {
            GpuMesh m;
            m.vertices = bgfx::createDynamicVertexBuffer(
                bgfx::copy(mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex)), vertices_);
            m.indices = bgfx::createIndexBuffer(
                bgfx::copy(mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t)),
                BGFX_BUFFER_INDEX32);
            m.vertex_count = mesh.vertices.size();
            meshes_.push_back(std::move(m));
        }
        for (uint32_t i = 0; i < source.instances.size(); ++i) {
            const auto &s = source.instances[i];
            meshes_[s.mesh].instances.push_back(i);
            auto &data = instances_[i];
            auto matrix = identity();
            std::copy(matrix.begin(), matrix.end(), data.begin());
            std::copy(s.color.begin(), s.color.end(), data.begin() + 16);
            auto encode = [&](size_t offset, uint32_t value) {
                data[offset] = float(value & 0xffff);
                data[offset + 1] = float(value >> 16);
            };
            encode(20, s.object_id);
            encode(22, std::bit_cast<uint32_t>(s.segmentation[0]));
            encode(24, std::bit_cast<uint32_t>(s.segmentation[1]));
        }
        for (auto &[id, t] : targets_)
            t.latest = {};
    }
    void update(const SceneFrame &frame) override {
        owner();
        validate_frame(scene_, frame);
        sequence_ = frame.sequence;
        for (size_t i = 0; i < instances_.size(); ++i) {
            auto matrix = frame.transforms[i];
            std::copy(matrix.begin(), matrix.end(), instances_[i].begin());
        }
    }
    void update_mesh(uint32_t index, std::span<const Vertex> vertices) override {
        owner();
        if (index >= meshes_.size() || vertices.size() != meshes_[index].vertex_count)
            throw std::invalid_argument("Dynamic mesh topology changed");
        if (pending_commands_)
            flush();
        bgfx::update(meshes_[index].vertices, 0,
                     bgfx::copy(vertices.data(), vertices.size_bytes()));
        stats_.upload_bytes += vertices.size_bytes();
    }
    Target create_target(Extent size, uint32_t samples) override {
        owner();
        extent(size);
        if (samples != 1 && samples != 4)
            throw std::invalid_argument("Probe supports 1x or 4x MSAA");
        auto slot = std::find(views_.begin(), views_.end(), false);
        if (slot == views_.end())
            throw std::runtime_error("Render target capacity reached");
        GpuTarget t;
        t.size = size;
        t.samples = samples;
        t.view = std::distance(views_.begin(), slot) * 4;
        allocate_target(t);
        *slot = true;
        Target id{allocate_id()};
        targets_.emplace(id.id, std::move(t));
        return id;
    }
    Target create_surface(NativeWindow window) override {
        owner();
        extent(window.size);
        if (!window.handle || !caps_.multiple_windows)
            throw std::invalid_argument("Native surface unavailable");
        auto slot = std::find(views_.begin(), views_.end(), false);
        if (slot == views_.end())
            throw std::runtime_error("Render target capacity reached");
        GpuTarget t;
        t.size = window.size;
        t.surface = true;
        t.view = std::distance(views_.begin(), slot) * 4;
        bgfx::SwapChain surface;
        surface.nwh = window.handle;
        surface.ndt = window.display;
        surface.width = window.size.width;
        surface.height = window.size.height;
        t.color_fb = bgfx::createFrameBuffer(surface);
        if (!bgfx::isValid(t.color_fb))
            throw std::runtime_error("Cannot create native surface");
        *slot = true;
        Target id{allocate_id()};
        targets_.emplace(id.id, std::move(t));
        return id;
    }
    void resize(Target id, Extent size) override {
        owner();
        extent(size);
        auto &t = target(id);
        if (t.size == size)
            return;
        if (t.surface)
            throw std::logic_error("Recreate secondary native surfaces when resizing in the probe");
        if (pending_commands_)
            flush();
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
        if (pending_commands_)
            flush();
        cancel(id);
        release_target(t);
        views_[t.view / 4] = false;
        targets_.erase(id.id);
    }
    FrameToken render(Target id, const CameraView &camera) override {
        owner();
        auto &t = target(id);
        if (t.surface)
            throw std::invalid_argument(
                "Render scenes to an offscreen target, then present its texture");
        begin_target(t);
        t.color_only = false;
        float clear_depth[4] = {camera.far_plane, 0, 0, 1};
        bgfx::setPaletteColor(2 + t.view / 4, clear_depth);
        auto view = column_major(camera.view), projection = camera.projection;
        if (!bgfx::getCaps()->homogeneousDepth)
            for (size_t c = 0; c < 4; ++c)
                projection[8 + c] = (projection[8 + c] + projection[12 + c]) * 0.5f;
        projection = column_major(projection);
        for (uint16_t pass = 0; pass < 2; ++pass) {
            bgfx::setViewRect(t.view + pass, 0, 0, t.size.width, t.size.height);
            bgfx::setViewFrameBuffer(t.view + pass, pass ? t.data_fb : t.color_fb);
            bgfx::setViewTransform(t.view + pass, view.data(), projection.data());
            if (pass)
                bgfx::setViewClear(t.view + pass, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 1.0f, 0, 0,
                                   1, 1, 2 + t.view / 4);
            else
                bgfx::setViewClear(t.view + pass, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x20262fff,
                                   1.0f);
            bgfx::touch(t.view + pass);
        }
        for (const auto &mesh : meshes_) {
            uint32_t count = mesh.instances.size();
            if (!count)
                continue;
            constexpr uint16_t stride = 20 * sizeof(float);
            // The portable instance layout has five vec4 slots. Share the three
            // affine rows, then upload color or exact metadata for each product pass.
            for (uint16_t pass = 0; pass < 2; ++pass) {
                if (bgfx::getAvailInstanceDataBuffer(count, stride) != count)
                    throw std::runtime_error("Instance upload capacity exhausted");
                bgfx::InstanceDataBuffer buffer;
                bgfx::allocInstanceDataBuffer(&buffer, count, stride);
                for (size_t i = 0; i < count; ++i) {
                    auto *destination = buffer.data + i * stride;
                    const auto &source = instances_[mesh.instances[i]];
                    std::memcpy(destination, source.data(), 12 * sizeof(float));
                    std::memcpy(destination + 12 * sizeof(float), source.data() + (pass ? 20 : 16),
                                8 * sizeof(float));
                }
                stats_.upload_bytes += count * stride;
                bgfx::setVertexBuffer(0, mesh.vertices);
                bgfx::setIndexBuffer(mesh.indices);
                bgfx::setInstanceDataBuffer(&buffer);
                uint64_t state = BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_WRITE_Z |
                                 BGFX_STATE_DEPTH_TEST_LESS;
                if (!pass && t.samples > 1)
                    state |= BGFX_STATE_MSAA;
                bgfx::setState(state);
                bgfx::submit(t.view + pass, pass ? data_program_ : color_program_);
                ++stats_.draw_calls;
                stats_.instances += count;
            }
        }
        t.latest = {id, t.generation, scene_.revision, sequence_, camera.revision, ++submission_};
        return t.latest;
    }
    ReadbackTicket readback(FrameToken frame, Product product, Region region) override {
        owner();
        auto &t = target(frame.target);
        if (t.surface || (t.color_only && product != Product::Color))
            throw std::invalid_argument("Requested output is unavailable");
        if (frame != t.latest || !frame.submission)
            throw std::invalid_argument("Frame is no longer available for readback");
        auto free = std::find_if(readbacks_.begin(), readbacks_.end(),
                                 [](const auto &r) { return r.ticket == 0; });
        if (free == readbacks_.end() && readbacks_.size() >= 8)
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
        size_t slot = std::distance(readbacks_.begin(), free);
        if (free == readbacks_.end())
            readbacks_.emplace_back();
        auto &r = readbacks_[slot];
        bool reuse = r.size == Extent{region.width, region.height} && r.product == product &&
                     bgfx::isValid(r.textures[0]);
        if (!reuse) {
            release_slot(r);
            r.textures = {bgfx::TextureHandle{bgfx::kInvalidHandle},
                          bgfx::TextureHandle{bgfx::kInvalidHandle}};
        }
        r.ticket = allocate_id();
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
            bgfx::blit(t.view + 2, dst, src);
            r.ready = bgfx::read(dst, r.bytes[i].data());
        }
        t.last_readback = gpu_frame_;
        pending_commands_ = true;
        return {r.ticket};
    }
    ReadbackResult poll(ReadbackTicket ticket) override {
        owner();
        if (!ticket.id)
            throw std::invalid_argument("Invalid readback ticket");
        auto it = std::find_if(readbacks_.begin(), readbacks_.end(),
                               [&](const auto &r) { return r.ticket == ticket.id; });
        if (it == readbacks_.end())
            throw std::invalid_argument("Unknown or consumed readback ticket");
        if (!complete(gpu_frame_, it->ready))
            return {ReadbackState::Pending, it->frame, {}};
        ReadbackResult result;
        result.frame = it->frame;
        result.state = it->canceled ? ReadbackState::Canceled : ReadbackState::Ready;
        if (!it->canceled) {
            auto &image = result.image;
            image.product = it->product;
            image.size = it->size;
            auto stride = pixel_bytes(image.product);
            image.pixels.resize(size_t(image.size.width) * image.size.height * stride);
            bool flip = bgfx::getCaps()->originBottomLeft;
            for (size_t y = 0; y < image.size.height; ++y)
                for (size_t x = 0; x < image.size.width; ++x) {
                    size_t from =
                        ((flip ? image.size.height - 1 - y : y) * image.size.width + x) * 4;
                    auto *destination = image.pixels.data() + (y * image.size.width + x) * stride;
                    if (image.product == Product::Color)
                        std::memcpy(destination, it->bytes[0].data() + from, 3);
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
            stats_.gpu_ms =
                1000.0 * double(stats->gpuTimeEnd - stats->gpuTimeBegin) / stats->gpuTimerFreq;
        auto result = stats_;
        stats_ = {};
        return result;
    }
    Texture target_texture(Target id) const override {
        owner();
        if (!targets_.contains(id.id))
            throw std::invalid_argument("Unknown target texture");
        return {id.id | target_texture_bit};
    }
    Texture upload_texture(Extent size, std::span<const std::byte> rgba) override {
        owner();
        extent(size);
        if (rgba.size() != size_t(size.width) * size.height * 4)
            throw std::invalid_argument("Invalid RGBA texture bytes");
        auto texture =
            bgfx::createTexture2D(size.width, size.height, false, 1, bgfx::TextureFormat::RGBA8,
                                  sampler, bgfx::copy(rgba.data(), rgba.size()));
        Texture id{allocate_id()};
        textures_.emplace(id.id, texture);
        return id;
    }
    void destroy(Texture id) override {
        owner();
        auto it = textures_.find(id.id);
        if (it == textures_.end())
            throw std::invalid_argument("Unknown owned texture");
        bgfx::destroy(it->second);
        textures_.erase(it);
    }
    FrameToken render_ui(const UiFrame &ui, Target output) override {
        owner();
        extent(ui.size);
        validate_ui(ui);
        if (!has_window_ && !output.id)
            throw std::logic_error("UI rendering requires a window");
        if (!output.id && ui.size != window_size_) {
            swap_chain_.width = ui.size.width;
            swap_chain_.height = ui.size.height;
            bgfx::reset(BGFX_RESET_NONE, &swap_chain_);
            window_size_ = ui.size;
        }
        auto projection = identity();
        projection[0] = 2.0f / ui.size.width;
        projection[3] = -1;
        projection[5] = -2.0f / ui.size.height;
        projection[7] = 1;
        projection = column_major(projection);
        bgfx::ViewId view = 250;
        bgfx::FrameBufferHandle framebuffer = BGFX_INVALID_HANDLE;
        FrameToken token;
        if (output.id) {
            auto &t = target(output);
            if (t.size != ui.size)
                throw std::invalid_argument("UI extent must match its target");
            begin_target(t);
            view = t.view + 1;
            framebuffer = t.color_fb;
            t.color_only = true;
            token = t.latest = {output, t.generation, scene_.revision, sequence_, 0, ++submission_};
        }
        bgfx::setViewMode(view, bgfx::ViewMode::Sequential);
        bgfx::setViewRect(view, 0, 0, ui.size.width, ui.size.height);
        bgfx::setViewFrameBuffer(view, framebuffer);
        bgfx::setViewTransform(view, nullptr, projection.data());
        bgfx::setViewClear(view, BGFX_CLEAR_COLOR, 0x14191eff);
        bgfx::touch(view);
        pending_commands_ = true;
        if (ui.vertices.empty() || ui.indices.empty())
            return token;
        bgfx::TransientVertexBuffer vertices;
        bgfx::TransientIndexBuffer indices;
        if (bgfx::getAvailTransientVertexBuffer(ui.vertices.size(), ui_vertices_) !=
                ui.vertices.size() ||
            bgfx::getAvailTransientIndexBuffer(ui.indices.size(), true) != ui.indices.size())
            throw std::runtime_error("UI upload capacity exhausted");
        bgfx::allocTransientVertexBuffer(&vertices, ui.vertices.size(), ui_vertices_);
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
            bgfx::setVertexBuffer(0, &vertices, command.vertex_offset,
                                  ui.vertices.size() - command.vertex_offset);
            bgfx::setIndexBuffer(&indices, command.first_index, command.index_count);
            bgfx::setTexture(0, image_sampler_, texture_handle(command.texture));
            bgfx::setState(
                BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A |
                BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA));
            bgfx::submit(view, ui_program_);
        }
        return token;
    }
};
} // namespace
std::unique_ptr<Renderer> make_bgfx_renderer(const BgfxOptions &options) {
    auto renderer = std::make_unique<BgfxRenderer>();
    renderer->initialize(options);
    return renderer;
}
} // namespace mojive
