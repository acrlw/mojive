#include "rendererFactory.hpp"
#include "sceneStream.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mojive/readback.hpp>
int main(int argc, char **argv) {
    using namespace mojive;
    try {
        if (argc != 5)
            throw std::invalid_argument("Usage: scene_capture shaders trajectory output backend");
        auto scene = probe::load(argv[2]);
        auto renderer = probe::makeRenderer({{}, argv[1]}, argv[4]);
        renderer->setScene(scene.source);
        renderer->update({1, 0, scene.frames[0]});
        auto target = renderer->createTarget({1920, 1080}, 4);
        auto token = renderer->render(target, scene.camera);
        std::array<Product, 4> products{Product::Color, Product::ObjectId, Product::MetricDepth,
                                        Product::Segmentation};
        std::array<const char *, 4> names{"color", "object-id", "depth", "segmentation"};
        std::array<ReadbackTicket, 4> tickets;
        for (size_t i = 0; i < 4; ++i)
            tickets[i] = renderer->readback(token, products[i]);
        auto output = std::filesystem::path(argv[3]);
        std::filesystem::create_directories(output);
        for (size_t i = 0; i < 4; ++i) {
            auto result = waitForReadback(*renderer, tickets[i]);
            if (result.state != ReadbackState::Ready || result.frame != token)
                throw std::runtime_error("Scene capture provenance failed");
            const auto &image = result.image;
            std::ofstream raw(output / (std::string(names[i]) + ".bin"), std::ios::binary);
            raw.write(reinterpret_cast<const char *>(image.pixels.data()), image.pixels.size());
            if (i == 0) {
                std::ofstream ppm(output / "color.ppm", std::ios::binary);
                ppm << "P6\n1920 1080\n255\n";
                ppm.write(reinterpret_cast<const char *>(image.pixels.data()), image.pixels.size());
            }
        }
        std::cout << argv[4] << " full scene products captured\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
