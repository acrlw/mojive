#include <array>
#include <iostream>
#include <mojive/Texture.hpp>
#include <stdexcept>

using namespace mojive;
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
int main() {
    try {
        const std::array<uint8_t, 8> pair{0, 0, 0, 0, 255, 255, 255, 255};
        for (bool srgb : {false, true}) {
            auto texture = prepareTexture({2, 1}, pair, 4, false, srgb);
            check(texture.rgba->size() == 12, "Incomplete mip chain");
            for (int c = 0; c < 3; ++c)
                check(int((*texture.rgba)[8 + c]) == (srgb ? 188 : 128), "Wrong RGB color space");
            check(int((*texture.rgba)[11]) == 128, "Alpha must remain linear and independent");
            for (size_t i = 0; i < pair.size(); ++i)
                check(int((*texture.rgba)[i]) == pair[i], "Base texture changed");
        }
        std::vector<uint8_t> odd(3 * 5);
        for (size_t y = 0; y < 3; ++y)
            for (size_t x = 0; x < 5; ++x)
                odd[y * 5 + x] = x == 4 ? 240 : y == 2 ? 120 : 0;
        auto texture = prepareTexture({5, 3}, odd, 1, false, false);
        check(texture.rgba->size() == (15 + 2 + 1) * 4, "Wrong NPOT extents");
        check(int((*texture.rgba)[60]) == 40 && int((*texture.rgba)[64]) == 120 &&
                  int((*texture.rgba)[68]) == 80,
              "Odd source edges lost in area filtering");
        check(int((*texture.rgba)[61]) == 0 && int((*texture.rgba)[63]) == 255,
              "Single-channel expansion changed");
        std::vector<uint8_t> cube(6 * 3 * 3 * 3);
        for (size_t face = 0; face < 6; ++face)
            std::fill(cube.begin() + face * 27, cube.begin() + (face + 1) * 27, face * 30);
        texture = prepareTexture({3, 3}, cube, 3, true, true);
        check(texture.rgba->size() == 6 * 40, "Incomplete cube mip chain");
        for (size_t face = 0; face < 6; ++face)
            for (size_t pixel = 0; pixel < 10; ++pixel) {
                check(int((*texture.rgba)[face * 40 + pixel * 4]) == face * 30,
                      "Cube face order changed");
                check(int((*texture.rgba)[face * 40 + pixel * 4 + 3]) == 255,
                      "RGB textures must be opaque");
            }
        bool rejected = false;
        try {
            prepareTexture({3, 2}, pair, 4, true, false);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "Malformed texture accepted");
        std::cout << "Texture color, alpha, NPOT, and cube contracts passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
