#include "RendererFactory.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mojive/Readback.hpp>
#include <stdexcept>
using namespace mojive;
struct Quad {
    std::array<UiVertex, 4> vertices;
    std::array<uint32_t, 6> indices{0, 1, 2, 0, 2, 3};
    std::array<UiCommand, 1> commands;
    Extent size;
    Quad(Texture texture, Extent extent, uint32_t tint = 0xffffffff) : size(extent) {
        float w = extent.width, h = extent.height;
        vertices = {
            {{0, 0, 0, 0, tint}, {w, 0, 1, 0, tint}, {w, h, 1, 1, tint}, {0, h, 0, 1, tint}}};
        commands = {{{0, 6, 0, {0, 0, w, h}, texture}}};
    }
    UiFrame frame() const {
        return {size, vertices, indices, commands};
    }
};
static void expect(const Image &image, uint32_t x, uint32_t y, std::array<int, 3> color) {
    auto offset = (size_t(y) * image.size.width + x) * 3;
    for (size_t i = 0; i < 3; ++i)
        if (std::abs(int(image.pixels[offset + i]) - color[i]) > 2)
            throw std::runtime_error("Composition pixel mismatch at " + std::to_string(x) + "," +
                                     std::to_string(y) + ": " +
                                     std::to_string(int(image.pixels[offset + i])) +
                                     " != " + std::to_string(color[i]));
}
int main(int argc, char **argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument("Usage: composition shaders output backend");
        std::filesystem::path output = argv[2];
        std::filesystem::create_directories(output);
        auto renderer = probe::makeRenderer({{}, argv[1]}, argv[3]);
        std::array<std::byte, 8 * 8 * 4> pixels;
        for (size_t y = 0; y < 8; ++y)
            for (size_t x = 0; x < 8; ++x) {
                auto p = (y * 8 + x) * 4;
                pixels[p] = std::byte(y < 4 && x < 4 ? 255 : 0);
                pixels[p + 1] = std::byte(y < 4 && x >= 4 ? 255 : 0);
                pixels[p + 2] = std::byte(y >= 4 ? 255 : 0);
                pixels[p + 3] = std::byte(255);
            }
        auto texture = renderer->uploadTexture({8, 8}, pixels);
        // Allocate the consumer first: resource allocation order must not dictate
        // render-pass execution order or introduce a frame of stale texture data.
        auto destination = renderer->createTarget({128, 128}, 1);
        auto source = renderer->createTarget({128, 128}, 1);
        auto mask = renderer->createTarget({128, 128}, 1);
        Quad original(texture, {128, 128});
        renderer->renderUi(original.frame(), source);
        Quad copy(renderer->targetTexture(source), {128, 128});
        auto copied = renderer->renderUi(copy.frame(), destination);
        auto result = waitForReadback(*renderer, renderer->readback(copied, Product::Color));
        expect(result.image, 20, 20, {255, 0, 0});
        expect(result.image, 100, 20, {0, 255, 0});
        expect(result.image, 20, 100, {0, 0, 255});
        Quad blend(texture, {128, 128}, 0x80ffffff);
        blend.commands[0].clip = {8, 8, 120, 120};
        auto blended = renderer->renderUi(blend.frame(), mask);
        result = waitForReadback(*renderer, renderer->readback(blended, Product::Color));
        expect(result.image, 20, 20, {138, 12, 15});
        expect(result.image, 2, 2, {20, 25, 30});
        std::ofstream image(output / "composition.ppm", std::ios::binary);
        image << "P6\n128 128\n255\n";
        image.write(reinterpret_cast<const char *>(result.image.pixels.data()),
                    result.image.pixels.size());
        renderer->destroy(mask);
        renderer->destroy(source);
        renderer->destroy(destination);
        renderer->destroy(texture);
        // Reuse view slots, staging storage, pipelines and targets across reloads.
        for (int n = 0; n < 120; ++n) {
            Extent size{uint32_t(129 + n % 5 * 31), uint32_t(101 + n % 3 * 29)};
            auto tex = renderer->uploadTexture({8, 8}, pixels);
            auto b = renderer->createTarget(size, 1), a = renderer->createTarget(size, 1);
            Quad q(tex, size);
            renderer->renderUi(q.frame(), a);
            Quad dependent(renderer->targetTexture(a), size);
            auto frame = renderer->renderUi(dependent.frame(), b);
            auto ticket = renderer->readback(frame, Product::Color, {12, 12, 1, 1});
            if (n % 3 == 0) {
                renderer->resize(b, {64, 64});
                if (waitForReadback(*renderer, ticket).state != ReadbackState::Canceled)
                    throw std::runtime_error("Resize did not cancel composition readback");
            } else
                expect(waitForReadback(*renderer, ticket).image, 0, 0, {255, 0, 0});
            renderer->destroy(a);
            renderer->destroy(b);
            renderer->destroy(tex);
            renderer->advance();
        }
        std::ofstream report(output / "composition.json");
        report << "{\"passed\":true,\"reverse_allocation_pass_dependency\":true,\"alpha_blend\":"
                  "true,\"scissor\":true,\"texture_orientation\":true,\"resource_cycles\":120}\n";
        std::cout << argv[3] << " composition and 120 resource cycles passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
