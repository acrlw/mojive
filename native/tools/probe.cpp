#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mojive/backends/bgfx.hpp>
#include <mojive/readback.hpp>
#include <stdexcept>
#include <thread>

using namespace mojive;
using Clock = std::chrono::steady_clock;
static void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class F> static void rejects(F function, const char *message) {
    bool caught = false;
    try {
        function();
    } catch (const std::exception &) {
        caught = true;
    }
    require(caught, message);
}
static ReadbackResult wait(Renderer &r, ReadbackTicket t) {
    return wait_for_readback(r, t);
}
template <class T> static T pixel(const Image &image, uint32_t x, uint32_t y, size_t part = 0) {
    T result;
    std::memcpy(&result,
                image.pixels.data() +
                    (size_t(y) * image.size.width + x) * pixel_bytes(image.product) +
                    part * sizeof(T),
                sizeof(T));
    return result;
}
static void save(const std::filesystem::path &root, const char *name, const Image &image) {
    std::ofstream raw(root / (std::string(name) + ".bin"), std::ios::binary);
    raw.write(reinterpret_cast<const char *>(image.pixels.data()), image.pixels.size());
    if (image.product == Product::Color) {
        std::ofstream ppm(root / (std::string(name) + ".ppm"), std::ios::binary);
        ppm << "P6\n" << image.size.width << " " << image.size.height << "\n255\n";
        ppm.write(reinterpret_cast<const char *>(image.pixels.data()), image.pixels.size());
    }
}
static SceneSource fixture() {
    SceneSource source;
    Mesh mesh;
    mesh.vertices = {{{-0.8f, -0.8f, 0}, {0, 0, 1}},
                     {{0.8f, -0.8f, 0}, {0, 0, 1}},
                     {{0.8f, 0.8f, 0}, {0, 0, 1}},
                     {{-0.8f, 0.8f, 0}, {0, 0, 1}}};
    mesh.indices = {0, 1, 2, 0, 2, 3};
    source.meshes.push_back(mesh);
    source.instances = {{0, 0x01000001, {-1, 0x1000001}, {1, 0.2f, 0.1f, 1}},
                        {0, 0xfedcba98, {INT32_MIN, INT32_MAX}, {0.1f, 1, 0.2f, 1}},
                        {0, UINT32_MAX, {-23456789, 42}, {0.2f, 0.3f, 1, 1}}};
    return source;
}
static std::vector<Matrix> poses() {
    std::vector<Matrix> result(3, identity());
    result[0][3] = -1.0f;
    result[0][7] = 0.7f;
    result[1][3] = 1.0f;
    result[1][7] = 0.7f;
    result[2][3] = 0;
    result[2][7] = -1.0f;
    return result;
}
static void conformance(Renderer &renderer, const std::filesystem::path &output) {
    auto source = fixture();
    auto transforms = poses();
    renderer.set_scene(source);
    renderer.update({source.revision, 7, transforms});
    CameraView camera;
    camera.view = look_at({0, 0, 5}, {0, 0, 0}, {0, 1, 0});
    camera.projection = orthographic(4, 4, .1f, 20);
    camera.far_plane = 20;
    camera.revision = 9;
    auto target = renderer.create_target({256, 256}, 4);
    auto frame = renderer.render(target, camera);
    auto color = renderer.readback(frame, Product::Color),
         ids = renderer.readback(frame, Product::ObjectId);
    auto segments = renderer.readback(frame, Product::Segmentation),
         depth = renderer.readback(frame, Product::MetricDepth);
    auto c = wait(renderer, color), i = wait(renderer, ids), s = wait(renderer, segments),
         d = wait(renderer, depth);
    require(c.state == ReadbackState::Ready && i.state == ReadbackState::Ready &&
                s.state == ReadbackState::Ready && d.state == ReadbackState::Ready,
            "Output failed");
    save(output, "color", c.image);
    save(output, "object-id", i.image);
    save(output, "segmentation", s.image);
    save(output, "depth", d.image);
    std::array<std::array<uint32_t, 2>, 3> positions = {{{64, 83}, {192, 83}, {128, 192}}};
    for (size_t n = 0; n < 3; ++n) {
        auto [x, y] = positions[n];
        std::cout << "pixel " << x << "," << y << " id=" << pixel<uint32_t>(i.image, x, y)
                  << " depth=" << pixel<float>(d.image, x, y) << std::endl;
        require(pixel<uint32_t>(i.image, x, y) == source.instances[n].object_id,
                "Object ID lost precision or image orientation is wrong");
        for (size_t part = 0; part < 2; ++part)
            require(pixel<int32_t>(s.image, x, y, part) == source.instances[n].segmentation[part],
                    "Signed segmentation changed");
        require(std::abs(pixel<float>(d.image, x, y) - 5) < 1e-4f, "Metric depth is incorrect");
    }
    require(pixel<uint32_t>(i.image, 0, 0) == 0, "Background ID must be zero");
    require(pixel<int32_t>(s.image, 0, 0) == -1 && pixel<int32_t>(s.image, 0, 0, 1) == -1,
            "Background segmentation must be negative");
    require(std::abs(pixel<float>(d.image, 0, 0) - 20) < 1e-4f,
            "Background depth must equal camera far");
    // Interior pixels must all retain the exact ID, including interpolated triangles.
    for (uint32_t y = 50; y < 110; ++y)
        for (uint32_t x = 35; x < 92; ++x)
            require(pixel<uint32_t>(i.image, x, y) == source.instances[0].object_id,
                    "ID interpolation corrupted an interior pixel");
    for (uint32_t y = 0; y < 256; ++y)
        for (uint32_t x = 0; x < 256; ++x) {
            auto value = pixel<uint32_t>(i.image, x, y);
            require(value == 0 || value == 0x01000001 || value == 0xfedcba98 || value == UINT32_MAX,
                    "MSAA edge corrupted ID");
        }
    auto pick = wait(renderer, renderer.readback(frame, Product::ObjectId, {192, 83, 1, 1}));
    require(pixel<uint32_t>(pick.image, 0, 0) == 0xfedcba98 && pick.frame == frame,
            "One-pixel picking lost frame provenance");
    rejects([&] { renderer.readback(frame, Product::Color, {256, 0, 1, 1}); },
            "Out-of-bounds readback accepted");
    auto earlier = renderer.readback(frame, Product::ObjectId, {192, 83, 1, 1});
    auto moved = transforms;
    moved[1][3] = 10;
    renderer.update({source.revision, 8, moved});
    auto newer_frame = renderer.render(target, camera);
    auto newer = renderer.readback(newer_frame, Product::ObjectId, {192, 83, 1, 1});
    require(pixel<uint32_t>(wait(renderer, earlier).image, 0, 0) == 0xfedcba98,
            "A later submission overwrote the picked frame");
    require(pixel<uint32_t>(wait(renderer, newer).image, 0, 0) == 0,
            "New frame contains stale transforms");
    renderer.update({source.revision, 9, transforms});
    frame = renderer.render(target, camera);
    auto canceled = renderer.readback(frame, Product::Color);
    renderer.resize(target, {193, 137});
    require(wait(renderer, canceled).state == ReadbackState::Canceled,
            "Resize returned an obsolete readback");
    rejects([&] { renderer.readback(frame, Product::ObjectId); },
            "Stale frame accepted after resize");
    for (auto size : {Extent{1, 1}, Extent{511, 257}, Extent{257, 511}, Extent{256, 256}}) {
        renderer.resize(target, size);
        auto token = renderer.render(target, camera);
        auto result = wait(renderer, renderer.readback(token, Product::ObjectId));
        require(result.image.size == size &&
                    result.image.pixels.size() == size_t(size.width) * size.height * 4,
                "Resize extent mismatch");
    }
    auto peer = renderer.create_target({128, 96});
    auto peer_frame = renderer.render(peer, camera);
    auto pending = renderer.readback(peer_frame, Product::ObjectId);
    renderer.destroy(peer);
    require(wait(renderer, pending).state == ReadbackState::Canceled,
            "Destroyed peer returned a live result");
    frame = renderer.render(target, camera);
    require(wait(renderer, renderer.readback(frame, Product::ObjectId)).state ==
                ReadbackState::Ready,
            "Closing peer stopped shared runtime");
    pending = renderer.readback(frame, Product::ObjectId);
    ++source.revision;
    renderer.set_scene(source);
    require(wait(renderer, pending).state == ReadbackState::Canceled,
            "Scene reload returned obsolete selection");
    rejects([&] { renderer.update({1, 0, transforms}); }, "Old source revision accepted");
    bool wrong_thread = false;
    std::thread other([&] {
        try {
            renderer.advance();
        } catch (const std::logic_error &) {
            wrong_thread = true;
        }
    });
    other.join();
    require(wrong_thread, "Owner-thread rule not enforced");
    renderer.update({source.revision, 8, transforms});
    frame = renderer.render(target, camera);
    std::vector<ReadbackTicket> tickets;
    for (int n = 0; n < 8; ++n)
        tickets.push_back(renderer.readback(frame, Product::ObjectId, {64, 83, 1, 1}));
    rejects([&] { renderer.readback(frame, Product::ObjectId); },
            "Readback backpressure not enforced");
    for (auto ticket : tickets)
        require(wait(renderer, ticket).state == ReadbackState::Ready, "Readback queue failed");
    auto changed = source.meshes[0].vertices;
    for (auto &v : changed)
        v.position[2] += 0.5f;
    renderer.update_mesh(0, changed);
    frame = renderer.render(target, camera);
    auto updated = wait(renderer, renderer.readback(frame, Product::MetricDepth));
    require(std::abs(pixel<float>(updated.image, 64, 83) - 4.5f) < 1e-4f,
            "Dynamic mesh update ignored");
    renderer.destroy(target);
    renderer.advance();
    std::ofstream report(output / "conformance.json");
    const auto &caps = renderer.capabilities();
    report << "{\n  \"passed\": true,\n  \"backend\": \"" << caps.backend
           << "\",\n  \"integer_target_capability\": " << caps.integer_target
           << ",\n  \"signed_pair_target_capability\": " << caps.signed_pair_target
           << ",\n  \"id_encoding\": \"lossless_rgba8\",\n  \"color_msaa\": 4,\n  "
              "\"multiple_windows_capability\": "
           << caps.multiple_windows << "\n}\n";
}
int main(int argc, char **argv) {
    try {
        std::string shaders = argc > 1 ? argv[1] : "output/native-build/shaders";
        std::filesystem::path output = argc > 2 ? argv[2] : "output/native-probe";
        std::filesystem::create_directories(output);
        BgfxOptions options;
        options.shader_directory = shaders;
        auto renderer = make_bgfx_renderer(options);
        std::cout << "Renderer: " << renderer->capabilities().backend << std::endl;
        rejects([&] { make_bgfx_renderer(options); }, "Second process runtime accepted");
        conformance(*renderer, output);
        renderer.reset();
        auto reopened = make_bgfx_renderer(options);
        conformance(*reopened, output);
        std::cout << "Native conformance passed, including runtime restart" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Native probe failed: " << error.what() << std::endl;
        return 1;
    }
}
