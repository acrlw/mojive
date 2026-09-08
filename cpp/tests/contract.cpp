#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <mojive/readback.hpp>
#include <mojive/render.hpp>
#include <stdexcept>

using namespace mojive;
static void check(bool value, const char *reason) {
    if (!value)
        throw std::runtime_error(reason);
}
template <class F> static void rejects(F f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception &) {
        rejected = true;
    }
    check(rejected, "Invalid input accepted");
}
// A test double compiled without any graphics dependency. It exercises the same
// consumer and completion protocol used by the GPU conformance executable.
class RecordingRenderer final : public Renderer {
    Capabilities mCaps{"recording", "CPU"};
    SceneSource mSource;
    uint64_t mSequence = 0;
    FrameToken mFrame;

  public:
    bool finish = true, canceled = false;
    int ticks = 0;
    const Capabilities &capabilities() const override {
        return mCaps;
    }
    void setScene(const SceneSource &source) override {
        validateScene(source);
        mSource = source;
    }
    void update(const SceneFrame &frame) override {
        validateFrame(mSource, frame);
        mSequence = frame.sequence;
    }
    void updateMesh(uint32_t, std::span<const Vertex>) override {
        throw std::logic_error("Not supported by test double");
    }
    Target createTarget(Extent, uint32_t) override {
        return {17};
    }
    Target createSurface(NativeWindow) override {
        throw std::logic_error("No native surfaces");
    }
    void resize(Target, Extent) override {
        canceled = true;
    }
    void destroy(Target) override {
        canceled = true;
    }
    FrameToken render(Target target, const CameraView &camera) override {
        return mFrame = {target, 3, mSource.revision, mSequence, camera.revision, 1};
    }
    ReadbackTicket readback(FrameToken frame, Product product, Region) override {
        check(frame == mFrame && product == Product::ObjectId,
              "Consumer changed output/provenance");
        return {91};
    }
    ReadbackResult poll(ReadbackTicket ticket) override {
        check(ticket.id == 91, "Consumer changed ticket");
        if (!finish || ticks < 2)
            return {ReadbackState::Pending, mFrame, {}};
        if (canceled)
            return {ReadbackState::Canceled, mFrame, {}};
        Image image{Product::ObjectId, {1, 1}, std::vector<std::byte>(4)};
        std::memcpy(image.pixels.data(), &mSource.instances[0].objectId, 4);
        return {ReadbackState::Ready, mFrame, std::move(image)};
    }
    FrameStats advance() override {
        ++ticks;
        return {};
    }
    Texture targetTexture(Target) const override {
        throw std::logic_error("No textures");
    }
    Texture uploadTexture(Extent, std::span<const std::byte>) override {
        throw std::logic_error("No textures");
    }
    void destroy(Texture) override {
        throw std::logic_error("No textures");
    }
    FrameToken renderUi(const UiFrame &, Target) override {
        throw std::logic_error("No UI");
    }
};
int main() {
    try {
        SceneSource scene;
        scene.meshes = {
            {{{{0, 0, 0}, {0, 0, 1}}, {{1, 0, 0}, {0, 0, 1}}, {{0, 1, 0}, {0, 0, 1}}}, {0, 1, 2}}};
        scene.instances = {{0, 0xfedcba98, {INT32_MIN, INT32_MAX}, {1, 1, 1, 1}}};
        validateScene(scene);
        auto matrix = identity();
        std::array<Matrix, 1> transforms = {matrix};
        validateFrame(scene, {1, 8, transforms});
        rejects([&] { validateFrame(scene, {2, 8, transforms}); });
        transforms[0][3] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { validateFrame(scene, {1, 8, transforms}); });
        transforms[0] = matrix;
        scene.meshes[0].indices[0] = 3;
        rejects([&] { validateScene(scene); });
        scene.meshes[0].indices[0] = 0;
        transforms[0][15] = 0;
        rejects([&] { validateFrame(scene, {1, 8, transforms}); });
        transforms[0] = matrix;
        auto camera = lookAt({0, 0, 5}, {0, 0, 0}, {0, 1, 0});
        check(camera[11] == -5, "Camera is not row-major");
        rejects([] { lookAt({0, 0, 0}, {0, 0, 0}, {0, 1, 0}); });
        rejects([] { perspective(1, 0, .1f, 10); });
        auto projection = perspective(1, 1, .1f, 10);
        check(projection[14] == -1, "Projection convention changed");
        const auto nan = std::numeric_limits<float>::quiet_NaN();
        const auto infinity = std::numeric_limits<float>::infinity();
        rejects([&] { perspective(nan, 1, .1f, 10); });
        rejects([&] { orthographic(1, infinity, .1f, 10); });
        rejects([&] { lookAt({0, 0, nan}, {0, 0, 0}, {0, 1, 0}); });
        rejects([] { lookAt({0, 0, 1}, {0, 0, 0}, {0, 0, 1}); });
        rejects([] { lookAt({0, 0, 1}, {0, 0, 0}, {1e38f, 1e38f, 0}); });
        CameraView invalidCamera;
        invalidCamera.view[3] = nan;
        rejects([&] { validateCamera(invalidCamera); });
        invalidCamera.view = identity();
        invalidCamera.farPlane = 0;
        rejects([&] { validateCamera(invalidCamera); });
        auto ndcDepth = [](const Matrix &p, float z) {
            return (p[10] * z + p[11]) / (p[14] * z + p[15]);
        };
        auto ortho = orthographic(4, 2, .1f, 10);
        for (const auto &p : {projection, ortho}) {
            check(std::abs(ndcDepth(p, -.1f) + 1) < 1e-5f, "Near plane convention changed");
            check(std::abs(ndcDepth(p, -10) - 1) < 1e-5f, "Far plane convention changed");
        }
        auto angled = lookAt({3, -4, 5}, {1, 2, 0}, {0, 0, 1});
        for (size_t row = 0; row < 3; ++row)
            check(std::abs(3 * angled[row * 4] - 4 * angled[row * 4 + 1] + 5 * angled[row * 4 + 2] +
                           angled[row * 4 + 3]) < 1e-5f,
                  "View transform does not map the eye to the origin");
        std::array<UiVertex, 3> uiVertices{};
        std::array<uint32_t, 3> uiIndices = {0, 1, 2};
        std::array<UiCommand, 1> commands = {{{0, 3, 0, {0, 0, 32, 32}, {1}}}};
        UiFrame ui{{32, 32}, uiVertices, uiIndices, commands};
        validateUi(ui);
        commands[0].firstIndex = UINT32_MAX;
        rejects([&] { validateUi(ui); });
        commands[0].firstIndex = 0;
        commands[0].vertexOffset = 1;
        rejects([&] { validateUi(ui); });
        commands[0].vertexOffset = 0;
        commands[0].clip[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { validateUi(ui); });
        auto mock = std::make_unique<RecordingRenderer>();
        auto *recording = mock.get();
        std::unique_ptr<Renderer> renderer = std::move(mock);
        renderer->setScene(scene);
        renderer->update({1, 8, transforms});
        auto target = renderer->createTarget({1, 1});
        CameraView view;
        view.revision = 23;
        auto token = renderer->render(target, view);
        auto ticket = renderer->readback(token, Product::ObjectId);
        auto result = waitForReadback(*renderer, ticket);
        uint32_t id;
        std::memcpy(&id, result.image.pixels.data(), 4);
        check(id == 0xfedcba98 && result.frame.sequence == 8 && result.frame.cameraRevision == 23,
              "Common consumer lost identity");
        renderer->resize(target, {2, 2});
        check(waitForReadback(*renderer, ticket).state == ReadbackState::Canceled,
              "Cancellation hidden by consumer");
        recording->finish = false;
        rejects([&] { waitForReadback(*renderer, ticket, std::chrono::milliseconds(0)); });
        std::cout << "Backend-independent contracts and consumer passed" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
