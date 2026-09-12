#include "../src/Rgb.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        for (size_t count : {0, 1, 3, 4, 5, 15, 16, 17, 193, 1920}) {
            for (size_t alignment = 0; alignment < 16; ++alignment) {
                std::vector<std::byte> source(count * 4 + alignment);
                std::vector<std::byte> destination(count * 3 + alignment + 16, std::byte{0xa5});
                for (size_t i = 0; i < source.size(); ++i)
                    source[i] = std::byte((i * 37 + 11) % 256);
                const auto original = source;
                mojive::detail::copyRgb(destination.data() + alignment, source.data() + alignment,
                                        count);
                for (size_t i = 0; i < destination.size(); ++i) {
                    auto expected = std::byte{0xa5};
                    if (i >= alignment && i < alignment + count * 3) {
                        auto channel = i - alignment;
                        expected = source[alignment + channel / 3 * 4 + channel % 3];
                    }
                    if (destination[i] != expected)
                        throw std::runtime_error("RGB channel or row boundary corrupted");
                }
                if (source != original)
                    throw std::runtime_error("RGB conversion changed its source");
            }
        }
        std::cout << "RGB channel, tail, alignment, and ownership checks passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
