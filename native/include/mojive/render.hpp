#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mojive {

// Experimental native contracts. Row-major matrices, Z-up world coordinates.
// These headers deliberately require only the C++ standard library.
using Matrix = std::array<float, 16>;
constexpr Matrix identity() {
    return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}
struct Extent {
    uint32_t width = 1, height = 1;
    bool operator==(const Extent &) const = default;
};
struct Region {
    uint32_t x = 0, y = 0, width = 0, height = 0;
};
struct Vertex {
    std::array<float, 3> position, normal;
};
struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};
struct Instance {
    uint32_t mesh = 0, object_id = 0;
    std::array<int32_t, 2> segmentation = {-1, -1};
    std::array<float, 4> color = {1, 1, 1, 1};
};
struct SceneSource {
    uint64_t revision = 1;
    std::vector<Mesh> meshes;
    std::vector<Instance> instances;
};
struct SceneFrame {
    uint64_t source_revision = 1, sequence = 0;
    std::span<const Matrix> transforms;
};
struct CameraView {
    Matrix view = identity();
    // Canonical OpenGL NDC depth [-1,1]. Conversion belongs to the backend.
    Matrix projection = identity();
    uint64_t revision = 0;
    float far_plane = 200;
};
enum class Product { Color, ObjectId, Segmentation, MetricDepth };
constexpr size_t pixel_bytes(Product product) {
    switch (product) {
    case Product::Color:
        return 3; // RGB uint8
    case Product::Segmentation:
        return 8; // two signed int32 values
    case Product::ObjectId:
    case Product::MetricDepth:
        return 4;
    }
    return 0;
}
struct Image {
    Product product = Product::Color;
    Extent size;
    // Tightly packed, top-left origin. Integer/float values use host byte order.
    std::vector<std::byte> pixels;
};
struct Target {
    uint64_t id = 0;
    bool operator==(const Target &) const = default;
};
struct FrameToken {
    Target target;
    uint64_t generation = 0, scene_revision = 0, sequence = 0, camera_revision = 0;
    uint64_t submission = 0;
    bool operator==(const FrameToken &) const = default;
};
struct ReadbackTicket {
    uint64_t id = 0;
};
enum class ReadbackState { Pending, Ready, Canceled };
struct ReadbackResult {
    ReadbackState state = ReadbackState::Pending;
    FrameToken frame;
    Image image;
};
struct Capabilities {
    std::string backend, device;
    bool readback = false, instancing = false, multiple_windows = false;
    bool integer_target = false, signed_pair_target = false, float_target = false;
    uint32_t max_texture_size = 0;
};
struct FrameStats {
    uint64_t draw_calls = 0, instances = 0, upload_bytes = 0;
    double gpu_ms = -1;
};
struct Texture {
    uint64_t id = 0;
};
struct UiVertex {
    float x, y, u, v;
    uint32_t rgba;
};
struct UiCommand {
    uint32_t first_index = 0, index_count = 0, vertex_offset = 0;
    std::array<float, 4> clip; // framebuffer pixels: left, top, right, bottom
    Texture texture;
};
struct UiFrame {
    Extent size;
    std::span<const UiVertex> vertices;
    std::span<const uint32_t> indices;
    std::span<const UiCommand> commands;
};

struct NativeWindow {
    void *handle = nullptr;
    void *display = nullptr;
    Extent size;
};

// One owner thread drives a renderer. Scene, CPU output, and UI packet types do
// not expose a graphics API, its handles, views, encoders, or completion counters.
class Renderer {
  public:
    virtual ~Renderer() = default;
    virtual const Capabilities &capabilities() const = 0;
    virtual void set_scene(const SceneSource &) = 0;
    virtual void update(const SceneFrame &) = 0;
    virtual void update_mesh(uint32_t mesh, std::span<const Vertex>) = 0;
    virtual Target create_target(Extent, uint32_t samples = 1) = 0;
    virtual Target create_surface(NativeWindow) = 0;
    virtual void resize(Target, Extent) = 0;
    virtual void destroy(Target) = 0;
    virtual FrameToken render(Target, const CameraView &) = 0;
    // Region coordinates refer to the displayed image, not backend texture origin.
    // Submission copies the source before its target can be overwritten. Results
    // retain provenance; resize/reload/destroy cancel outstanding results safely.
    virtual ReadbackTicket readback(FrameToken, Product, Region = {}) = 0;
    virtual ReadbackResult poll(ReadbackTicket) = 0;
    virtual FrameStats advance() = 0;
    virtual Texture target_texture(Target) const = 0;
    virtual Texture upload_texture(Extent, std::span<const std::byte> rgba) = 0;
    virtual void destroy(Texture) = 0;
    virtual FrameToken render_ui(const UiFrame &, Target output = {}) = 0;
};

// The platform layer supplies native OS handles; a backend decides how to use them.
// Offscreen consumers leave handles empty and do not initialize an editor.

void validate_ui(const UiFrame &);
void validate_scene(const SceneSource &);
void validate_frame(const SceneSource &, const SceneFrame &);
Matrix look_at(std::array<float, 3> eye, std::array<float, 3> target, std::array<float, 3> up);
Matrix perspective(float fov_y_radians, float aspect, float near_plane, float far_plane);
Matrix orthographic(float width, float height, float near_plane, float far_plane);

} // namespace mojive
