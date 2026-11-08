#pragma once
#include "imgui.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mojive::imguiBridge {
// Immutable local geometry, independent of draw lists, contexts, fonts and colors.
// Python conversion and index validation happen once, not on every submitted frame.
class DrawMesh {
public:
    DrawMesh(std::vector<ImVec2> positions, std::vector<unsigned int> indices,
             std::vector<bool> opaque)
        : mPositions(std::move(positions)), mIndices(std::move(indices)),
          mOpaque(std::move(opaque)) {
        if (mPositions.size() != mOpaque.size() || mIndices.size() % 3)
            throw std::invalid_argument("mesh coverage must match vertices; indices must form triangles");
        if (mPositions.size() > std::numeric_limits<ImDrawIdx>::max() ||
            mIndices.size() > std::numeric_limits<int>::max())
            throw std::invalid_argument("mesh exceeds a single native draw reservation");
        for (const auto &point : mPositions)
            if (!std::isfinite(point.x) || !std::isfinite(point.y))
                throw std::invalid_argument("mesh positions must be finite");
        for (unsigned int index : mIndices)
            if (index >= mPositions.size())
                throw std::invalid_argument("triangle index outside vertex buffer");
    }

    void draw(ImDrawList &drawList, const ImVec2 &origin, ImU32 color) const {
        if (mIndices.empty() || !(color & IM_COL32_A_MASK))
            return;
        drawList.PrimReserve(static_cast<int>(mIndices.size()), static_cast<int>(mPositions.size()));
        const unsigned int base = drawList._VtxCurrentIdx;
        const ImVec2 uv = ImGui::GetIO().Fonts->TexUvWhitePixel;
        for (size_t i = 0; i < mPositions.size(); ++i) {
            const auto &point = mPositions[i];
            drawList.PrimWriteVtx(ImVec2(point.x + origin.x, point.y + origin.y), uv,
                                 mOpaque[i] ? color : color & ~IM_COL32_A_MASK);
        }
        for (unsigned int index : mIndices)
            drawList.PrimWriteIdx(static_cast<ImDrawIdx>(base + index));
    }

    void drawBehind(ImDrawList &drawList, const ImVec2 &origin, ImU32 color) {
        if (mIndices.empty() || !(color & IM_COL32_A_MASK))
            return;
        // Begin() has already emitted the native window decorations. Insert
        // this underlay without repainting their antialiased edge a second time.
        mUnderlay.Split(&drawList, 2);
        drawList.CmdBuffer.swap(mUnderlay._Channels[1]._CmdBuffer);
        drawList.IdxBuffer.swap(mUnderlay._Channels[1]._IdxBuffer);
        drawList.AddDrawCmd();
        draw(drawList, origin, color);
        mUnderlay.Merge(&drawList);
    }

private:
    std::vector<ImVec2> mPositions;
    std::vector<unsigned int> mIndices;
    std::vector<bool> mOpaque;
    ImDrawListSplitter mUnderlay;
};
} // namespace mojive::imguiBridge
