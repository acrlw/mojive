#pragma once

#include <tesselator.h>

#include <algorithm>
#include <csetjmp>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace mojive::geometry2d {

// Some upstream allocation sites dereference null without checking. Abort only
// through the C-only tessellation boundary, and reclaim every outstanding block.
// No C++ object with a nontrivial destructor may be created inside that boundary.
class TessAllocator {
    struct alignas(std::max_align_t) Block {
        Block *previous;
        Block *next;
        size_t size;
    };
    Block *mHead = nullptr;
    size_t mBytes = 0;
    size_t mLimit;
    size_t mPeak = 0;

    void *allocate(size_t size) noexcept {
        if (size > mLimit || sizeof(Block) > mLimit - size ||
            mBytes > mLimit - size - sizeof(Block))
            std::longjmp(failure, 1);
        auto *block = static_cast<Block *>(std::malloc(sizeof(Block) + size));
        if (!block)
            std::longjmp(failure, 2);
        *block = {nullptr, mHead, size};
        if (mHead)
            mHead->previous = block;
        mHead = block;
        mBytes += sizeof(Block) + size;
        mPeak = std::max(mPeak, mBytes);
        return block + 1;
    }

    void free(void *pointer) noexcept {
        if (!pointer)
            return;
        auto *block = static_cast<Block *>(pointer) - 1;
        if (block->previous)
            block->previous->next = block->next;
        else
            mHead = block->next;
        if (block->next)
            block->next->previous = block->previous;
        mBytes -= sizeof(Block) + block->size;
        std::free(block);
    }

  public:
    std::jmp_buf failure;

    explicit TessAllocator(size_t limit) : mLimit(limit) {}
    TessAllocator(const TessAllocator &) = delete;
    TessAllocator &operator=(const TessAllocator &) = delete;
    ~TessAllocator() {
        while (mHead)
            free(mHead + 1);
    }

    size_t peakBytes() const {
        return mPeak;
    }

    TESSalloc callbacks() {
        TESSalloc result{};
        result.userData = this;
        result.memalloc = [](void *user, unsigned size) {
            return static_cast<TessAllocator *>(user)->allocate(size);
        };
        result.memfree = [](void *user, void *pointer) {
            static_cast<TessAllocator *>(user)->free(pointer);
        };
        result.memrealloc = [](void *user, void *pointer, unsigned size) {
            auto &arena = *static_cast<TessAllocator *>(user);
            auto *replacement = arena.allocate(size);
            if (pointer) {
                auto *block = static_cast<Block *>(pointer) - 1;
                std::memcpy(replacement, pointer, std::min(size_t(size), block->size));
                arena.free(pointer);
            }
            return replacement;
        };
        result.meshEdgeBucketSize = 128;
        result.meshVertexBucketSize = 128;
        result.meshFaceBucketSize = 64;
        result.dictNodeBucketSize = 128;
        result.regionBucketSize = 64;
        return result;
    }
};
} // namespace mojive::geometry2d
