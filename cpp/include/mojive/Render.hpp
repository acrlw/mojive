#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
struct Light {
    bool operator==(const Light &) const = default;
    std::array<float, 3> position = {0, 0, 3}, direction = {0, 0, -1};
    std::array<float, 3> diffuse = {.7f, .7f, .7f}, specular = {.3f, .3f, .3f};
    std::array<float, 3> attenuation = {1, 0, 0};
    int type = 0;
    float cutoff = 45, exponent = 10, range = 0, radius = 0;
    bool castShadow = true;
};
struct Lighting {
    bool operator==(const Lighting &) const = default;
    bool enabled = false;
    std::vector<Light> lights;
    std::array<float, 3> ambient = {.2f, .2f, .2f};
    std::array<float, 4> headlightDiffuse = {0, 0, 0, 0};
    std::array<float, 3> headlightSpecular = {0, 0, 0};
    std::array<float, 4> fog = {0, 0, 0, 0};
    std::array<float, 3> fogColor = {0, 0, 0}, hazeColor = {1, 1, 1};
    int imageTexture = -1, skyboxTexture = -1;
    float imageIntensity = 0;
    bool horizonHaze = false;
    int hazeSlices = 64;
    float hazeDensity = 0;
};
struct TextureSource {
    Extent size;
    bool mipmaps = false, srgb = false, cube = false;
    // Immutable shared upload storage can outlive the caller until GPU submission.
    std::shared_ptr<const std::vector<std::byte>> rgba;
};
struct SceneStyle {
    bool operator==(const SceneStyle &) const = default;
    std::array<float, 4> background = {0.13f, 0.14f, 0.16f, 1};
    bool textures = true, wireframe = false, transparentIds = true;
    bool cullFace = false, transparent = true, additive = false, tonemap = true;
    bool fog = false, haze = true, msaa = true;
    bool shadows = true, skybox = true, reflections = true;
    int shadowQuality = 1;
    int debugView = 0;
    uint32_t selectedId = 0;
    bool selectionFill = true, outline = true, selectionOutline = true, selectionXray = false;
};
struct SceneSource {
    uint64_t revision = 1;
    // Shared immutable geometry survives source replacement without repacking.
    std::vector<std::shared_ptr<const Mesh>> meshes;
    std::vector<Instance> instances;
    std::vector<Material> materials;
    std::vector<TextureSource> textures;
    std::vector<uint32_t> materialIndices;
    std::vector<uint32_t> infinitePlanes;
    // 0: no planar surface, 1: local XY plane, 2: local +Z box face.
    std::vector<uint8_t> planarKinds;
    bool linearColors = false;
    std::vector<std::array<float, 4>> visualMaterials, cubeCoords;
    float extent = 1, shadowClip = 1;
    std::array<float, 3> center = {0, 0, 0};
};
struct SceneFrame {
    uint64_t sourceRevision = 1, sequence = 0;
    std::span<const Matrix> transforms;
    std::span<const std::array<float, 4>> texcoords;
    std::span<const std::array<float, 4>> colors;
    std::span<const std::array<float, 4>> materials, cubeCoords;
};
struct CameraView {
    Matrix view = identity();
    // Canonical OpenGL NDC depth [-1,1]. Conversion belongs to the backend.
    Matrix projection = identity();
    uint64_t revision = 0;
    float farPlane = 200;
    float nearPlane = .01f;
    std::array<float, 3> focus = {0, 0, 0};
};
struct Texture {
    uint64_t id = 0;
};
enum class DebugPath {
    Segment,
    Arrow,
    Point,
    Stroke,
    Solid,
    Sector,
    DragLink,
    ScreenTriangle,
    Text
};
enum class Occlusion { Depth, Always, Ghost };
constexpr std::array<uint32_t, 9> debugRecordFloats = {13, 13, 8, 14, 20, 14, 18, 13, 17};
struct DebugBatch {
    DebugPath path = DebugPath::Segment;
    Occlusion occlusion = Occlusion::Depth;
    uint32_t start = 0, count = 0, mesh = 0;
};
struct OverlayDraw {
    uint32_t mesh = 0;
    Matrix transform = identity();
    std::array<float, 4> color = {1, 1, 1, 1};
    float maskRadius = 0;
    bool depthTest = true, depthWrite = true, cullFace = true;
};
struct SurfaceBatch {
    uint32_t mesh = 0, start = 0, count = 0;
    int texture = -1;
    bool transparent = false;
};
struct OverlayFrame {
    // Lit color-only surfaces: affine pose, color, UV, material, cube mapping, reserved.
    std::vector<std::array<float, 32>> surfaces;
    std::vector<SurfaceBatch> surfaceBatches;
    std::vector<OverlayDraw> gizmos;
    std::array<std::vector<float>, 9> streams;
    std::vector<DebugBatch> debug;
    Texture glyphAtlas;
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
struct ImageView {
    Product product = Product::Color;
    Extent size;
    // Writable, tightly packed storage with the same layout as Image.
    std::span<std::byte> pixels;
};
struct Scene {
    uint64_t id = 0;
    bool operator==(const Scene &) const = default;
};
struct Target {
    uint64_t id = 0;
    bool operator==(const Target &) const = default;
};
struct RenderRequest {
    bool color = true, sceneData = true;
    std::optional<Product> dataProduct;
};
enum class RenderPass {
    Shadow,
    Reflection,
    Color,
    SceneData,
    Identity,
    Outline,
    Debug,
    Gizmo,
    Ui,
    Count
};
constexpr std::array<const char *, size_t(RenderPass::Count)> renderPassNames = {
    "shadow", "reflection", "color", "scene data", "identity", "outline", "debug", "gizmo", "ui"};
struct PassTimings {
    bool operator==(const PassTimings &) const = default;
    std::array<double, size_t(RenderPass::Count)> cpuMs{}, gpuMs{};
    uint32_t cpuMask = 0, gpuMask = 0;
    // GPU samples arrive later; retain their original target submission identity.
    uint64_t cpuSubmission = 0, gpuSubmission = 0;
};
struct ResourceStats {
    uint64_t meshUploads = 0, textureUploads = 0, uploadBytes = 0;
};
struct FrameStats {
    bool operator==(const FrameStats &) const = default;
    uint64_t drawCalls = 0, instances = 0, uploadBytes = 0;
    double gpuMs = -1;
    bool reflectionRendered = false, reflectionReused = false;
    bool shadowRendered = false, shadowReused = false;
    uint64_t shadowInstances = 0, culledShadowInstances = 0;
    uint64_t culledInstances = 0;
    PassTimings passes;
};
struct FrameToken {
    Target target;
    uint64_t generation = 0, sceneRevision = 0, sequence = 0, cameraRevision = 0;
    uint64_t submission = 0;
    FrameStats statistics;
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

enum class WindowSystem { Native, Wayland };
struct NativeWindow {
    void *handle = nullptr;
    void *display = nullptr;
    Extent size;
    WindowSystem system = WindowSystem::Native;
};

// One owner thread drives a renderer. Scene, CPU output, and UI packet types do
// not expose a graphics API, its handles, views, encoders, or completion counters.
class Renderer {
  public:
    virtual ~Renderer() = default;
    virtual const Capabilities &capabilities() const = 0;
    virtual void setScene(const SceneSource &) = 0;
    virtual void setOverlays(Scene, OverlayFrame) {
        throw std::logic_error("Overlay rendering is unsupported");
    }
    virtual void setLighting(Scene, const Lighting &) {
        throw std::logic_error("Scene lighting is unavailable");
    }
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
    virtual void setVsync(Target, bool) {
        throw std::logic_error("Surface presentation control is unavailable");
    }
    virtual void resize(Target, Extent) = 0;
    virtual void destroy(Target) = 0;
    virtual FrameToken render(Target, const CameraView &) = 0;
    virtual FrameToken renderRequested(Target target, const CameraView &camera,
                                       RenderRequest request) {
        if (!request.color && !request.sceneData)
            throw std::invalid_argument("Empty render request");
        return render(target, camera);
    }
    // Region coordinates refer to the displayed image, not backend texture origin.
    // Submission copies the source before its target can be overwritten. Results
    // retain provenance; resize/reload/destroy cancel outstanding results safely.
    virtual ReadbackTicket readback(FrameToken, Product, Region = {}) = 0;
    virtual ReadbackResult poll(ReadbackTicket) = 0;
    // Synchronous delivery into caller-owned storage; no reference survives return.
    virtual ReadbackState readInto(FrameToken, ImageView, Region = {});
    virtual FrameStats advance() = 0;
    virtual void reloadShaders() {
        throw std::logic_error("Shader reload is unavailable");
    }
    virtual Texture targetTexture(Target) const = 0;
    virtual ResourceStats resourceStats() const {
        return {};
    }
    virtual Texture uploadTexture(Extent, std::span<const std::byte> rgba) = 0;
    virtual void destroy(Texture) = 0;
    virtual FrameToken renderUi(const UiFrame &, Target output = {}) = 0;
};

// The platform layer supplies native OS handles; a backend decides how to use them.
// Offscreen consumers leave handles empty and do not initialize an editor.

void validateCamera(const CameraView &);
void validateUi(const UiFrame &);
void validateMesh(const Mesh &);
void validateScene(const SceneSource &, bool validateGeometry = true);
void validateFrame(const SceneSource &, const SceneFrame &);
Matrix lookAt(std::array<float, 3> eye, std::array<float, 3> target, std::array<float, 3> up);
Matrix perspective(float fovYRadians, float aspect, float nearPlane, float farPlane);
Matrix orthographic(float width, float height, float nearPlane, float farPlane);

} // namespace mojive
