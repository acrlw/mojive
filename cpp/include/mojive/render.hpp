#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
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
    std::vector<std::array<float, 2>> texcoords;
};
struct Instance {
    uint32_t mesh = 0, objectId = 0;
    std::array<int32_t, 2> segmentation = {-1, -1};
    std::array<float, 4> color = {1, 1, 1, 1};
};
struct Material {
    int32_t texture = -1;
    float emission = 0, specular = 0.5f, shininess = 0.5f;
};
struct TextureSource {
    Extent size;
    bool mipmaps = false;
    std::vector<std::byte> rgba;
};
struct SceneStyle {
    std::array<float, 4> background = {0.125f, 0.15f, 0.18f, 1};
    bool textures = true, wireframe = false, transparentIds = true;
};
struct SceneSource {
    uint64_t revision = 1;
    std::vector<Mesh> meshes;
    std::vector<Instance> instances;
    std::vector<Material> materials;
    std::vector<TextureSource> textures;
    std::vector<uint32_t> materialIndices;
    bool linearColors = false;
};
struct SceneFrame {
    uint64_t sourceRevision = 1, sequence = 0;
    std::span<const Matrix> transforms;
    std::span<const std::array<float, 4>> texcoords;
    std::span<const std::array<float, 4>> colors;
};
struct CameraView {
    Matrix view = identity();
    // Canonical OpenGL NDC depth [-1,1]. Conversion belongs to the backend.
    Matrix projection = identity();
    uint64_t revision = 0;
    float farPlane = 200;
};
enum class Product { Color, ObjectId, Segmentation, MetricDepth, ColorAlpha };
constexpr size_t pixelBytes(Product product) {
    switch (product) {
    case Product::Color:
        return 3; // RGB uint8
    case Product::Segmentation:
        return 8; // two signed int32 values
    case Product::ColorAlpha:
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
struct Scene {
    uint64_t id = 0;
    bool operator==(const Scene &) const = default;
};
struct Target {
    uint64_t id = 0;
    bool operator==(const Target &) const = default;
};
struct FrameToken {
    Target target;
    uint64_t generation = 0, sceneRevision = 0, sequence = 0, cameraRevision = 0;
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
    bool readback = false, instancing = false, multipleWindows = false;
    bool integerTarget = false, signedPairTarget = false, floatTarget = false;
    uint32_t maxTextureSize = 0;
    bool multipleScenes = false;
};
struct FrameStats {
    uint64_t drawCalls = 0, instances = 0, uploadBytes = 0;
    double gpuMs = -1;
};
struct Texture {
    uint64_t id = 0;
};
struct UiVertex {
    float x, y, u, v;
    uint32_t rgba;
};
struct UiCommand {
    uint32_t firstIndex = 0, indexCount = 0, vertexOffset = 0;
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
    virtual void setScene(const SceneSource &) = 0;
    virtual void configure(Scene, const SceneStyle &) {
        throw std::logic_error("Scene styling is unavailable");
    }
    virtual Scene createScene(const SceneSource &) {
        throw std::logic_error("Independent scenes are unavailable");
    }
    virtual void setScene(Scene, const SceneSource &) {
        throw std::logic_error("Independent scenes are unavailable");
    }
    virtual void update(Scene, const SceneFrame &) {
        throw std::logic_error("Independent scenes are unavailable");
    }
    virtual void updateMesh(Scene, uint32_t, std::span<const Vertex>) {
        throw std::logic_error("Independent scenes are unavailable");
    }
    virtual Target createTarget(Scene, Extent, uint32_t = 1) {
        throw std::logic_error("Independent scenes are unavailable");
    }
    // Destroying a scene invalidates its attached targets and readback tickets.
    virtual void destroy(Scene) {
        throw std::logic_error("Independent scenes are unavailable");
    }
    virtual void update(const SceneFrame &) = 0;
    virtual void updateMesh(uint32_t mesh, std::span<const Vertex>) = 0;
    virtual Target createTarget(Extent, uint32_t samples = 1) = 0;
    virtual Target createSurface(NativeWindow) = 0;
    virtual void resize(Target, Extent) = 0;
    virtual void destroy(Target) = 0;
    virtual FrameToken render(Target, const CameraView &) = 0;
    // Region coordinates refer to the displayed image, not backend texture origin.
    // Submission copies the source before its target can be overwritten. Results
    // retain provenance; resize/reload/destroy cancel outstanding results safely.
    virtual ReadbackTicket readback(FrameToken, Product, Region = {}) = 0;
    virtual ReadbackResult poll(ReadbackTicket) = 0;
    virtual FrameStats advance() = 0;
    virtual Texture targetTexture(Target) const = 0;
    virtual Texture uploadTexture(Extent, std::span<const std::byte> rgba) = 0;
    virtual void destroy(Texture) = 0;
    virtual FrameToken renderUi(const UiFrame &, Target output = {}) = 0;
};

// The platform layer supplies native OS handles; a backend decides how to use them.
// Offscreen consumers leave handles empty and do not initialize an editor.

void validateCamera(const CameraView &);
void validateUi(const UiFrame &);
void validateScene(const SceneSource &);
void validateFrame(const SceneSource &, const SceneFrame &);
Matrix lookAt(std::array<float, 3> eye, std::array<float, 3> target, std::array<float, 3> up);
Matrix perspective(float fovYRadians, float aspect, float nearPlane, float farPlane);
Matrix orthographic(float width, float height, float nearPlane, float farPlane);

} // namespace mojive
