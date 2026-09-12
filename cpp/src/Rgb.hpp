#pragma once
#include <cstddef>
#include <cstring>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <tmmintrin.h>
#endif

namespace mojive {
namespace detail {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("ssse3"))) inline void copyRgbSimd(std::byte *destination,
                                                         const std::byte *source, size_t count) {
    const auto mask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    size_t x = 0;
    for (; x + 4 <= count; x += 4) {
        auto rgba = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + x * 4));
        auto rgb = _mm_shuffle_epi8(rgba, mask);
        // Store exactly twelve bytes, including at the end of an unaligned row.
        _mm_storel_epi64(reinterpret_cast<__m128i *>(destination + x * 3), rgb);
        const auto tail = _mm_cvtsi128_si32(_mm_srli_si128(rgb, 8));
        std::memcpy(destination + x * 3 + 8, &tail, sizeof(tail));
    }
    for (; x < count; ++x)
        std::memcpy(destination + x * 3, source + x * 4, 3);
}
#endif
inline void copyRgb(std::byte *destination, const std::byte *source, size_t count) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool hasShuffle = __builtin_cpu_supports("ssse3");
    if (hasShuffle) {
        copyRgbSimd(destination, source, count);
        return;
    }
#endif
    for (size_t x = 0; x < count; ++x)
        std::memcpy(destination + x * 3, source + x * 4, 3);
}
} // namespace detail
} // namespace mojive
