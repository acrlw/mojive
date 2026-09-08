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
    Capabilities caps_{"recording", "CPU"};
    SceneSource source_;
    uint64_t sequence_ = 0;
    FrameToken frame_;

  public:
    bool finish = true, canceled = false;
    int ticks = 0;
    const Capabilities &capabilities() const override {
        return caps_;
    }
    void set_scene(const SceneSource &source) override {
        validate_scene(source);
        source_ = source;
    }
    void update(const SceneFrame &frame) override {
        validate_frame(source_, frame);
        sequence_ = frame.sequence;
    }
    void update_mesh(uint32_t, std::span<const Vertex>) override {
        throw std::logic_error("Not supported by test double");
    }
    Target create_target(Extent, uint32_t) override {
        return {17};
    }
    Target create_surface(NativeWindow) override {
        throw std::logic_error("No native surfaces");
    }
    void resize(Target, Extent) override {
        canceled = true;
    }
    void destroy(Target) override {
        canceled = true;
    }
    FrameToken render(Target target, const CameraView &camera) override {
        return frame_ = {target, 3, source_.revision, sequence_, camera.revision, 1};
    }
    ReadbackTicket readback(FrameToken frame, Product product, Region) override {
        check(frame == frame_ && product == Product::ObjectId,
              "Consumer changed output/provenance");
        return {91};
    }
    ReadbackResult poll(ReadbackTicket ticket) override {
        check(ticket.id == 91, "Consumer changed ticket");
        if (!finish || ticks < 2)
            return {ReadbackState::Pending, frame_, {}};
        if (canceled)
            return {ReadbackState::Canceled, frame_, {}};
        Image image{Product::ObjectId, {1, 1}, std::vector<std::byte>(4)};
        std::memcpy(image.pixels.data(), &source_.instances[0].object_id, 4);
        return {ReadbackState::Ready, frame_, std::move(image)};
    }
    FrameStats advance() override {
        ++ticks;
        return {};
    }
    Texture target_texture(Target) const override {
        throw std::logic_error("No textures");
    }
    Texture upload_texture(Extent, std::span<const std::byte>) override {
        throw std::logic_error("No textures");
    }
    void destroy(Texture) override {
        throw std::logic_error("No textures");
    }
    FrameToken render_ui(const UiFrame &, Target) override {
        throw std::logic_error("No UI");
    }
};
int main() {
    try {
        SceneSource scene;
        scene.meshes = {
            {{{{0, 0, 0}, {0, 0, 1}}, {{1, 0, 0}, {0, 0, 1}}, {{0, 1, 0}, {0, 0, 1}}}, {0, 1, 2}}};
        scene.instances = {{0, 0xfedcba98, {INT32_MIN, INT32_MAX}, {1, 1, 1, 1}}};
        validate_scene(scene);
        auto matrix = identity();
        std::array<Matrix, 1> transforms = {matrix};
        validate_frame(scene, {1, 8, transforms});
        rejects([&] { validate_frame(scene, {2, 8, transforms}); });
        transforms[0][3] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { validate_frame(scene, {1, 8, transforms}); });
        transforms[0] = matrix;
        scene.meshes[0].indices[0] = 3;
        rejects([&] { validate_scene(scene); });
        scene.meshes[0].indices[0] = 0;
        transforms[0][15] = 0;
        rejects([&] { validate_frame(scene, {1, 8, transforms}); });
        transforms[0] = matrix;
        auto camera = look_at({0, 0, 5}, {0, 0, 0}, {0, 1, 0});
        check(camera[11] == -5, "Camera is not row-major");
        rejects([] { look_at({0, 0, 0}, {0, 0, 0}, {0, 1, 0}); });
        rejects([] { perspective(1, 0, .1f, 10); });
        auto projection = perspective(1, 1, .1f, 10);
        check(projection[14] == -1, "Projection convention changed");
        std::array<UiVertex, 3> ui_vertices{};
        std::array<uint32_t, 3> ui_indices = {0, 1, 2};
        std::array<UiCommand, 1> commands = {{{0, 3, 0, {0, 0, 32, 32}, {1}}}};
        UiFrame ui{{32, 32}, ui_vertices, ui_indices, commands};
        validate_ui(ui);
        commands[0].first_index = UINT32_MAX;
        rejects([&] { validate_ui(ui); });
        commands[0].first_index = 0;
        commands[0].vertex_offset = 1;
        rejects([&] { validate_ui(ui); });
        commands[0].vertex_offset = 0;
        commands[0].clip[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { validate_ui(ui); });
        auto mock = std::make_unique<RecordingRenderer>();
        auto *recording = mock.get();
        std::unique_ptr<Renderer> renderer = std::move(mock);
        renderer->set_scene(scene);
        renderer->update({1, 8, transforms});
        auto target = renderer->create_target({1, 1});
        CameraView view;
        view.revision = 23;
        auto token = renderer->render(target, view);
        auto ticket = renderer->readback(token, Product::ObjectId);
        auto result = wait_for_readback(*renderer, ticket);
        uint32_t id;
        std::memcpy(&id, result.image.pixels.data(), 4);
        check(id == 0xfedcba98 && result.frame.sequence == 8 && result.frame.camera_revision == 23,
              "Common consumer lost identity");
        renderer->resize(target, {2, 2});
        check(wait_for_readback(*renderer, ticket).state == ReadbackState::Canceled,
              "Cancellation hidden by consumer");
        recording->finish = false;
        rejects([&] { wait_for_readback(*renderer, ticket, std::chrono::milliseconds(0)); });
        std::cout << "Backend-independent contracts and consumer passed" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
