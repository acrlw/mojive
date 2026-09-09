#pragma once
#include <functional>
#include <mojive/Log.hpp>
#include <mojive/Render.hpp>

namespace mojive {
// Native runtime: serialize backend work on one native owner thread. Accepted
// calls finish before close returns. No job contains a Python object or callback.
class RenderRuntime final : public Renderer {
  public:
    using Factory = std::function<std::unique_ptr<Renderer>()>;
    explicit RenderRuntime(Factory, const LogOptions & = {});
    ~RenderRuntime() override;
    RenderRuntime(const RenderRuntime &) = delete;
    RenderRuntime &operator=(const RenderRuntime &) = delete;
    const Capabilities &capabilities() const override;
    void setScene(const SceneSource &) override;
    void configure(Scene, const SceneStyle &) override;
    void setLighting(Scene, const Lighting &) override;
    void setOverlays(Scene, OverlayFrame) override;
    Scene createScene(const SceneSource &) override;
    void setScene(Scene, const SceneSource &) override;
    void update(Scene, const SceneFrame &) override;
    void updateMesh(Scene, uint32_t, std::span<const Vertex>) override;
    Target createTarget(Scene, Extent, uint32_t = 1) override;
    void destroy(Scene) override;
    void update(const SceneFrame &) override;
    void updateMesh(uint32_t, std::span<const Vertex>) override;
    Target createTarget(Extent, uint32_t samples = 1) override;
    Target createSurface(NativeWindow) override;
    void setVsync(Target, bool) override;
    void resize(Target, Extent) override;
    void destroy(Target) override;
    FrameToken render(Target, const CameraView &) override;
    FrameToken renderRequested(Target, const CameraView &, RenderRequest) override;
    ReadbackTicket readback(FrameToken, Product, Region = {}) override;
    ReadbackResult poll(ReadbackTicket) override;
    FrameStats advance() override;
    Texture targetTexture(Target) const override;
    Texture uploadTexture(Extent, std::span<const std::byte>) override;
    void destroy(Texture) override;
    FrameToken renderUi(const UiFrame &, Target output = {}) override;
    ReadbackResult read(FrameToken, Product, Region = {});
    ReadbackResult wait(ReadbackTicket);
    Log &log();
    bool closed() const;
    void close();

  private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};
} // namespace mojive
