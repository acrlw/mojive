#include <cstring>
#include <mojive/RenderRuntime.hpp>
#include <mojive/Texture.hpp>
#include <mojive/backends/Bgfx.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace mojive;
namespace {
template <class T, int... Shape>
using Array = nb::ndarray<nb::numpy, const T, nb::shape<Shape...>, nb::c_contig, nb::device::cpu>;
template <class T> nb::object ownedArray(const void *bytes, std::initializer_list<size_t> shape) {
    size_t count = 1;
    for (auto value : shape)
        count *= value;
    auto storage = std::make_unique<std::vector<T>>(count);
    if (count)
        std::memcpy(storage->data(), bytes, count * sizeof(T));
    nb::capsule owner(storage.get(), [](void *pointer) noexcept {
        delete static_cast<std::vector<T> *>(pointer);
    });
    auto *data = storage.release();
    return nb::cast(nb::ndarray<nb::numpy, T>(data->data(), shape, owner));
}
template <class T>
nb::object imageView(ReadbackResult &result, std::initializer_list<size_t> shape) {
    // The completed result owns these bytes independently of the GPU/runtime.
    // Let NumPy retain that result instead of copying every exported image.
    return nb::cast(nb::ndarray<nb::numpy, T>(reinterpret_cast<T *>(result.image.pixels.data()),
                                              shape, nb::find(result)));
}
nb::object imageArray(ReadbackResult &result) {
    if (result.state != ReadbackState::Ready)
        return nb::none();
    const auto &image = result.image;
    const size_t h = image.size.height, w = image.size.width;
    switch (image.product) {
    case Product::Color:
        return imageView<uint8_t>(result, {h, w, 3});
    case Product::ColorAlpha:
        return imageView<uint8_t>(result, {h, w, 4});
    case Product::ObjectId:
        return imageView<uint32_t>(result, {h, w});
    case Product::Segmentation:
        return imageView<int32_t>(result, {h, w, 2});
    case Product::MetricDepth:
        return imageView<float>(result, {h, w});
    }
    throw std::invalid_argument("Unknown image product");
}
Matrix matrix(Array<float, 4, 4> value) {
    Matrix result;
    std::memcpy(result.data(), value.data(), sizeof(result));
    return result;
}
std::vector<Vertex> vertices(Array<float, -1, 3> positions, Array<float, -1, 3> normals) {
    if (positions.shape(0) != normals.shape(0))
        throw std::invalid_argument("Position and normal counts differ");
    std::vector<Vertex> result(positions.shape(0));
    for (size_t i = 0; i < result.size(); ++i) {
        std::memcpy(result[i].position.data(), positions.data() + 3 * i, 3 * sizeof(float));
        std::memcpy(result[i].normal.data(), normals.data() + 3 * i, 3 * sizeof(float));
    }
    return result;
}
} // namespace
void bindRender(nb::module_ &module) {
    nb::enum_<Product>(module, "Product")
        .value("COLOR", Product::Color)
        .value("RGBA", Product::ColorAlpha)
        .value("OBJECT_ID", Product::ObjectId)
        .value("SEGMENTATION", Product::Segmentation)
        .value("METRIC_DEPTH", Product::MetricDepth);
    nb::enum_<ReadbackState>(module, "ReadbackState")
        .value("PENDING", ReadbackState::Pending)
        .value("READY", ReadbackState::Ready)
        .value("CANCELED", ReadbackState::Canceled);
    nb::class_<Extent>(module, "Extent")
        .def(nb::init<>())
        .def_rw("width", &Extent::width)
        .def_rw("height", &Extent::height);
    nb::class_<Region>(module, "Region")
        .def(nb::init<>())
        .def_rw("x", &Region::x)
        .def_rw("y", &Region::y)
        .def_rw("width", &Region::width)
        .def_rw("height", &Region::height);
    nb::class_<Scene>(module, "Scene").def(nb::init<>()).def_ro("id", &Scene::id);
    nb::class_<Texture>(module, "Texture").def_ro("id", &Texture::id);
    nb::class_<UiCommand>(module, "UiCommand")
        .def(nb::init<>())
        .def_rw("first_index", &UiCommand::firstIndex)
        .def_rw("index_count", &UiCommand::indexCount)
        .def_rw("vertex_offset", &UiCommand::vertexOffset)
        .def_rw("clip", &UiCommand::clip)
        .def_rw("texture", &UiCommand::texture);
    nb::class_<Target>(module, "Target").def_ro("id", &Target::id);
    nb::class_<FrameToken>(module, "FrameToken")
        .def_ro("target", &FrameToken::target)
        .def_ro("generation", &FrameToken::generation)
        .def_ro("scene_revision", &FrameToken::sceneRevision)
        .def_ro("sequence", &FrameToken::sequence)
        .def_ro("camera_revision", &FrameToken::cameraRevision)
        .def_ro("submission", &FrameToken::submission)
        .def_ro("statistics", &FrameToken::statistics);
    nb::class_<ReadbackTicket>(module, "ReadbackTicket").def_ro("id", &ReadbackTicket::id);
    nb::class_<ReadbackResult>(module, "ReadbackResult")
        .def_ro("state", &ReadbackResult::state)
        .def_ro("frame", &ReadbackResult::frame)
        .def_prop_ro("image", &imageArray);
    nb::class_<ResourceStats>(module, "ResourceStats")
        .def_ro("mesh_uploads", &ResourceStats::meshUploads)
        .def_ro("texture_uploads", &ResourceStats::textureUploads)
        .def_ro("upload_bytes", &ResourceStats::uploadBytes);
    nb::class_<FrameStats>(module, "FrameStats")
        .def_prop_ro("cpu_ms",
                     [](const FrameStats &s) {
                         nb::dict result;
                         for (size_t i = 0; i < renderPassNames.size(); ++i)
                             if (s.passes.cpuMask & (1u << i))
                                 result[renderPassNames[i]] = s.passes.cpuMs[i];
                         return result;
                     })
        .def_prop_ro("gpu_pass_ms",
                     [](const FrameStats &s) {
                         nb::dict result;
                         for (size_t i = 0; i < renderPassNames.size(); ++i)
                             if (s.passes.gpuMask & (1u << i))
                                 result[renderPassNames[i]] = s.passes.gpuMs[i];
                         return result;
                     })
        .def_prop_ro("cpu_submission", [](const FrameStats &s) { return s.passes.cpuSubmission; })
        .def_prop_ro("gpu_submission", [](const FrameStats &s) { return s.passes.gpuSubmission; })
        .def_ro("draw_calls", &FrameStats::drawCalls)
        .def_ro("instances", &FrameStats::instances)
        .def_ro("upload_bytes", &FrameStats::uploadBytes)
        .def_ro("gpu_ms", &FrameStats::gpuMs)
        .def_ro("reflection_rendered", &FrameStats::reflectionRendered)
        .def_ro("reflection_reused", &FrameStats::reflectionReused)
        .def_ro("shadow_rendered", &FrameStats::shadowRendered)
        .def_ro("shadow_reused", &FrameStats::shadowReused)
        .def_ro("shadow_instances", &FrameStats::shadowInstances)
        .def_ro("culled_shadow_instances", &FrameStats::culledShadowInstances)
        .def_ro("culled_instances", &FrameStats::culledInstances);
    nb::class_<Capabilities>(module, "Capabilities")
        .def_ro("backend", &Capabilities::backend)
        .def_ro("device", &Capabilities::device)
        .def_ro("readback", &Capabilities::readback)
        .def_ro("instancing", &Capabilities::instancing)
        .def_ro("max_texture_size", &Capabilities::maxTextureSize)
        .def_ro("multiple_scenes", &Capabilities::multipleScenes);
    nb::class_<CameraView>(module, "CameraView")
        .def(nb::init<>())
        .def_rw("revision", &CameraView::revision)
        .def_rw("near_plane", &CameraView::nearPlane)
        .def_rw("focus", &CameraView::focus)
        .def_rw("far_plane", &CameraView::farPlane)
        .def_prop_rw(
            "view", [](const CameraView &c) { return ownedArray<float>(c.view.data(), {4, 4}); },
            [](CameraView &c, Array<float, 4, 4> value) { c.view = matrix(value); })
        .def_prop_rw(
            "projection",
            [](const CameraView &c) { return ownedArray<float>(c.projection.data(), {4, 4}); },
            [](CameraView &c, Array<float, 4, 4> value) { c.projection = matrix(value); });
    nb::class_<OverlayDraw>(module, "OverlayDraw")
        .def(nb::init<>())
        .def_rw("mesh", &OverlayDraw::mesh)
        .def_rw("color", &OverlayDraw::color)
        .def_rw("mask_radius", &OverlayDraw::maskRadius)
        .def_rw("depth_test", &OverlayDraw::depthTest)
        .def_rw("depth_write", &OverlayDraw::depthWrite)
        .def_rw("cull_face", &OverlayDraw::cullFace)
        .def_prop_rw(
            "transform",
            [](const OverlayDraw &d) { return ownedArray<float>(d.transform.data(), {4, 4}); },
            [](OverlayDraw &d, Array<float, 4, 4> v) { d.transform = matrix(v); });
    nb::class_<DebugBatch>(module, "DebugBatch")
        .def(nb::init<>())
        .def_rw("start", &DebugBatch::start)
        .def_rw("count", &DebugBatch::count)
        .def_rw("mesh", &DebugBatch::mesh)
        .def_prop_rw(
            "path", [](const DebugBatch &b) { return int(b.path); },
            [](DebugBatch &b, int p) { b.path = DebugPath(p); })
        .def_prop_rw(
            "occlusion", [](const DebugBatch &b) { return int(b.occlusion); },
            [](DebugBatch &b, int p) { b.occlusion = Occlusion(p); });
    nb::class_<SurfaceBatch>(module, "SurfaceBatch")
        .def(nb::init<>())
        .def_rw("mesh", &SurfaceBatch::mesh)
        .def_rw("start", &SurfaceBatch::start)
        .def_rw("count", &SurfaceBatch::count)
        .def_rw("texture", &SurfaceBatch::texture)
        .def_rw("transparent", &SurfaceBatch::transparent);
    nb::class_<OverlayFrame>(module, "OverlayFrame")
        .def(nb::init<>())
        .def_rw("gizmos", &OverlayFrame::gizmos)
        .def_rw("surface_batches", &OverlayFrame::surfaceBatches)
        .def("set_surfaces",
             [](OverlayFrame &frame, Array<float, -1, 32> values) {
                 frame.surfaces.resize(values.shape(0));
                 std::memcpy(frame.surfaces.data(), values.data(), values.size() * sizeof(float));
             })
        .def_rw("debug", &OverlayFrame::debug)
        .def_rw("glyph_atlas", &OverlayFrame::glyphAtlas)
        .def("set_stream", [](OverlayFrame &frame, size_t path, Array<float, -1, -1> data) {
            if (path >= frame.streams.size() || data.shape(1) != debugRecordFloats[path])
                throw std::invalid_argument("Invalid debug record layout");
            auto &out = frame.streams[path];
            out.resize(data.size());
            if (data.size())
                std::memcpy(out.data(), data.data(), data.size() * sizeof(float));
        });
    nb::class_<Light>(module, "Light")
        .def(nb::init<>())
        .def_rw("position", &Light::position)
        .def_rw("direction", &Light::direction)
        .def_rw("diffuse", &Light::diffuse)
        .def_rw("specular", &Light::specular)
        .def_rw("attenuation", &Light::attenuation)
        .def_rw("type", &Light::type)
        .def_rw("cutoff", &Light::cutoff)
        .def_rw("exponent", &Light::exponent)
        .def_rw("range", &Light::range)
        .def_rw("radius", &Light::radius)
        .def_rw("cast_shadow", &Light::castShadow);
    nb::class_<Lighting>(module, "Lighting")
        .def(nb::init<>())
        .def_rw("enabled", &Lighting::enabled)
        .def_rw("lights", &Lighting::lights)
        .def_rw("ambient", &Lighting::ambient)
        .def_rw("headlight_diffuse", &Lighting::headlightDiffuse)
        .def_rw("headlight_specular", &Lighting::headlightSpecular)
        .def_rw("fog", &Lighting::fog)
        .def_rw("fog_color", &Lighting::fogColor)
        .def_rw("haze_color", &Lighting::hazeColor)
        .def_rw("image_texture", &Lighting::imageTexture)
        .def_rw("skybox_texture", &Lighting::skyboxTexture)
        .def_rw("horizon_haze", &Lighting::horizonHaze)
        .def_rw("haze_slices", &Lighting::hazeSlices)
        .def_rw("haze_density", &Lighting::hazeDensity)
        .def_rw("image_intensity", &Lighting::imageIntensity);
    nb::class_<SceneStyle>(module, "SceneStyle")
        .def(nb::init<>())
        .def_rw("background", &SceneStyle::background)
        .def_rw("textures", &SceneStyle::textures)
        .def_rw("wireframe", &SceneStyle::wireframe)
        .def_rw("transparent_ids", &SceneStyle::transparentIds)
        .def_rw("cull_face", &SceneStyle::cullFace)
        .def_rw("transparent", &SceneStyle::transparent)
        .def_rw("additive", &SceneStyle::additive)
        .def_rw("tonemap", &SceneStyle::tonemap)
        .def_rw("fog", &SceneStyle::fog)
        .def_rw("haze", &SceneStyle::haze)
        .def_rw("msaa", &SceneStyle::msaa)
        .def_rw("debug_view", &SceneStyle::debugView)
        .def_rw("selected_id", &SceneStyle::selectedId)
        .def_rw("outline", &SceneStyle::outline)
        .def_rw("selection_outline", &SceneStyle::selectionOutline)
        .def_rw("selection_xray", &SceneStyle::selectionXray)
        .def_rw("selection_fill", &SceneStyle::selectionFill)
        .def_rw("reflections", &SceneStyle::reflections)
        .def_rw("skybox", &SceneStyle::skybox)
        .def_rw("shadows", &SceneStyle::shadows)
        .def_rw("shadow_quality", &SceneStyle::shadowQuality);
    nb::class_<Material>(module, "Material")
        .def(nb::init<>())
        .def_rw("texture", &Material::texture)
        .def_rw("emission", &Material::emission)
        .def_rw("specular", &Material::specular)
        .def_rw("shininess", &Material::shininess);
    nb::class_<Mesh>(module, "PreparedMesh");
    nb::class_<TextureSource>(module, "PreparedTexture");
    module.def("prepare_mesh", [](Array<float, -1, 3> positions, Array<float, -1, 3> normals,
                                  Array<uint32_t, -1> indices, Array<float, -1, 2> uv) {
        nb::gil_scoped_release release;
        auto mesh = std::make_shared<Mesh>();
        mesh->vertices = vertices(positions, normals);
        mesh->indices.assign(indices.data(), indices.data() + indices.size());
        mesh->texcoords.resize(uv.shape(0));
        if (uv.size())
            std::memcpy(mesh->texcoords.data(), uv.data(), uv.size() * sizeof(float));
        validateMesh(*mesh);
        return std::shared_ptr<const Mesh>(std::move(mesh));
    });
    module.def("prepare_texture", [](Array<uint8_t, -1, -1, -1, -1> pixels, bool cube, bool srgb) {
        if (pixels.shape(0) != (cube ? 6 : 1))
            throw std::invalid_argument("Invalid texture face count");
        nb::gil_scoped_release release;
        return prepareTexture({uint32_t(pixels.shape(2)), uint32_t(pixels.shape(1))},
                              {pixels.data(), pixels.size()}, pixels.shape(3), cube, srgb);
    });
    nb::class_<SceneSource>(module, "SceneSource")
        .def(nb::init<>())
        .def("add_prepared_mesh",
             [](SceneSource &s, std::shared_ptr<const Mesh> mesh) {
                 s.meshes.push_back(std::move(mesh));
                 return s.meshes.size() - 1;
             })
        .def("add_prepared_texture",
             [](SceneSource &s, const TextureSource &texture) {
                 s.textures.push_back(texture);
                 return s.textures.size() - 1;
             })
        .def_rw("revision", &SceneSource::revision)
        .def_rw("materials", &SceneSource::materials)
        .def_rw("planar_kinds", &SceneSource::planarKinds)
        .def_rw("infinite_planes", &SceneSource::infinitePlanes)
        .def_rw("linear_colors", &SceneSource::linearColors)
        .def_rw("extent", &SceneSource::extent)
        .def_rw("shadow_clip", &SceneSource::shadowClip)
        .def_rw("center", &SceneSource::center)
        .def("set_visuals",
             [](SceneSource &s, Array<float, -1, 4> material, Array<float, -1, 4> cube) {
                 s.visualMaterials.resize(material.shape(0));
                 s.cubeCoords.resize(cube.shape(0));
                 if (material.size())
                     std::memcpy(s.visualMaterials.data(), material.data(),
                                 material.size() * sizeof(float));
                 if (cube.size())
                     std::memcpy(s.cubeCoords.data(), cube.data(), cube.size() * sizeof(float));
             })
        .def("add_texture_pixels",
             [](SceneSource &s, Array<uint8_t, -1, -1, -1, -1> pixels, bool cube, bool srgb) {
                 if (pixels.shape(0) != (cube ? 6 : 1))
                     throw std::invalid_argument("Invalid texture face count");
                 TextureSource texture;
                 {
                     nb::gil_scoped_release release;
                     texture = prepareTexture(
                         {uint32_t(pixels.shape(2)), uint32_t(pixels.shape(1))},
                         {pixels.data(), pixels.size()}, pixels.shape(3), cube, srgb);
                 }
                 s.textures.push_back(std::move(texture));
                 return s.textures.size() - 1;
             })
        .def("add_texture_data",
             [](SceneSource &s, uint32_t width, uint32_t height, Array<uint8_t, -1> bytes,
                bool cube, bool srgb) {
                 TextureSource t;
                 t.size = {width, height};
                 t.mipmaps = true;
                 t.cube = cube;
                 t.srgb = srgb;
                 auto rgba = std::make_shared<std::vector<std::byte>>(bytes.size());
                 t.rgba = rgba;
                 if (bytes.size())
                     std::memcpy(rgba->data(), bytes.data(), bytes.size());
                 s.textures.push_back(std::move(t));
                 return s.textures.size() - 1;
             })
        .def("add_texture_mips",
             [](SceneSource &s, uint32_t width, uint32_t height, Array<uint8_t, -1> bytes) {
                 TextureSource texture;
                 texture.size = {width, height};
                 texture.mipmaps = true;
                 auto rgba = std::make_shared<std::vector<std::byte>>(bytes.size());
                 texture.rgba = rgba;
                 if (!rgba->empty())
                     std::memcpy(rgba->data(), bytes.data(), bytes.size());
                 s.textures.push_back(std::move(texture));
                 return s.textures.size() - 1;
             })
        .def(
            "set_material_indices",
            [](SceneSource &s, Array<uint32_t, -1> indices) {
                s.materialIndices.assign(indices.data(), indices.data() + indices.size());
            },
            nb::arg("indices").noconvert())
        .def("set_mesh_texcoords",
             [](SceneSource &s, size_t mesh, Array<float, -1, 2> uv) {
                 auto replacement = std::make_shared<Mesh>(*s.meshes.at(mesh));
                 s.meshes.at(mesh) = replacement;
                 auto &out = replacement->texcoords;
                 out.resize(uv.shape(0));
                 if (!out.empty())
                     std::memcpy(out.data(), uv.data(), uv.size() * sizeof(float));
             })
        .def("add_texture",
             [](SceneSource &s, Array<uint8_t, -1, -1, 4> pixels) {
                 TextureSource texture;
                 texture.size = {uint32_t(pixels.shape(1)), uint32_t(pixels.shape(0))};
                 auto rgba = std::make_shared<std::vector<std::byte>>(pixels.size());
                 texture.rgba = rgba;
                 if (!rgba->empty())
                     std::memcpy(rgba->data(), pixels.data(), pixels.size());
                 s.textures.push_back(std::move(texture));
                 return s.textures.size() - 1;
             })
        .def_prop_ro("mesh_count", [](const SceneSource &s) { return s.meshes.size(); })
        .def_prop_ro("instance_count", [](const SceneSource &s) { return s.instances.size(); })
        .def(
            "add_mesh",
            [](SceneSource &s, Array<float, -1, 3> positions, Array<float, -1, 3> normals,
               Array<uint32_t, -1> indices) {
                Mesh mesh;
                mesh.vertices = vertices(positions, normals);
                if (indices.size())
                    mesh.indices.assign(indices.data(), indices.data() + indices.size());
                auto index = s.meshes.size();
                s.meshes.push_back(std::make_shared<Mesh>(std::move(mesh)));
                return index;
            },
            nb::arg("positions"), nb::arg("normals"), nb::arg("indices").noconvert())
        .def(
            "set_instances",
            [](SceneSource &s, Array<uint32_t, -1> meshes, Array<uint32_t, -1> ids,
               Array<int32_t, -1, 2> segments, Array<float, -1, 4> colors) {
                const auto count = meshes.shape(0);
                if (ids.shape(0) != count || segments.shape(0) != count || colors.shape(0) != count)
                    throw std::invalid_argument("Instance array counts differ");
                std::vector<Instance> result(count);
                for (size_t i = 0; i < count; ++i) {
                    result[i].mesh = meshes.data()[i];
                    result[i].objectId = ids.data()[i];
                    std::memcpy(result[i].segmentation.data(), segments.data() + 2 * i,
                                2 * sizeof(int32_t));
                    std::memcpy(result[i].color.data(), colors.data() + 4 * i, 4 * sizeof(float));
                }
                s.instances = std::move(result);
            },
            nb::arg("meshes").noconvert(), nb::arg("object_ids").noconvert(),
            nb::arg("segmentation").noconvert(), nb::arg("colors"));
    nb::class_<RenderRuntime>(module, "RenderRuntime")
        .def("resource_stats", &RenderRuntime::resourceStats,
             nb::call_guard<nb::gil_scoped_release>())
        .def(
            "__init__",
            [](RenderRuntime *self, std::string shaders, LogOptions options, bool wayland) {
                nb::gil_scoped_release release;
                new (self) RenderRuntime(
                    [shaders = std::move(shaders), wayland] {
                        BgfxOptions render;
                        render.shaderDirectory = shaders;
                        render.window.system =
                            wayland ? WindowSystem::Wayland : WindowSystem::Native;
                        return makeBgfxRenderer(render);
                    },
                    options);
            },
            nb::arg("shader_directory"), nb::arg("log_options") = LogOptions{},
            nb::arg("wayland") = false)
        .def_prop_ro("capabilities", [](const RenderRuntime &r) { return r.capabilities(); })
        .def_prop_ro("log", &RenderRuntime::log, nb::rv_policy::reference_internal)
        .def_prop_ro("closed", &RenderRuntime::closed)
        .def("close", &RenderRuntime::close, nb::call_guard<nb::gil_scoped_release>())
        .def("set_overlays",
             [](RenderRuntime &r, Scene scene, OverlayFrame frame) {
                 nb::gil_scoped_release release;
                 r.setOverlays(scene, std::move(frame));
             })
        .def("set_lighting",
             [](RenderRuntime &r, Scene scene, Lighting light) {
                 nb::gil_scoped_release release;
                 r.setLighting(scene, light);
             })
        .def("update_visuals",
             [](RenderRuntime &r, Scene scene, Array<float, -1, 4, 4> pose, Array<float, -1, 4> uv,
                Array<float, -1, 4> color, Array<float, -1, 4> material, Array<float, -1, 4> cube,
                uint64_t revision, uint64_t sequence) {
                 nb::gil_scoped_release release;
                 thread_local std::vector<Matrix> transforms;
                 thread_local std::vector<std::array<float, 4>> coords, colors, materials, cubes;
                 transforms.resize(pose.shape(0));
                 if (pose.size())
                     std::memcpy(transforms.data(), pose.data(), pose.size() * sizeof(float));
                 auto copy = [](auto &out, const auto &in) {
                     out.resize(in.shape(0));
                     if (in.size())
                         std::memcpy(out.data(), in.data(), in.size() * sizeof(float));
                 };
                 copy(coords, uv);
                 copy(colors, color);
                 copy(materials, material);
                 copy(cubes, cube);
                 r.update(scene,
                          {revision, sequence, transforms, coords, colors, materials, cubes});
             })
        .def("configure",
             [](RenderRuntime &r, Scene scene, SceneStyle style) {
                 nb::gil_scoped_release release;
                 r.configure(scene, style);
             })
        .def("update_textured",
             [](RenderRuntime &r, Scene scene, Array<float, -1, 4, 4> array, Array<float, -1, 4> uv,
                Array<float, -1, 4> colors, uint64_t revision, uint64_t sequence) {
                 thread_local std::vector<Matrix> transforms;
                 thread_local std::vector<std::array<float, 4>> coords, rgba;
                 transforms.resize(array.shape(0));
                 coords.resize(uv.shape(0));
                 if (!transforms.empty())
                     std::memcpy(transforms.data(), array.data(), array.size() * sizeof(float));
                 if (!coords.empty())
                     std::memcpy(coords.data(), uv.data(), uv.size() * sizeof(float));
                 rgba.resize(colors.shape(0));
                 if (!rgba.empty())
                     std::memcpy(rgba.data(), colors.data(), colors.size() * sizeof(float));
                 nb::gil_scoped_release release;
                 r.update(scene, {revision, sequence, transforms, coords, rgba});
             })
        .def("create_scene",
             [](RenderRuntime &r, SceneSource source) {
                 nb::gil_scoped_release release;
                 return r.createScene(source);
             })
        .def("destroy_scene",
             [](RenderRuntime &r, Scene scene) {
                 nb::gil_scoped_release release;
                 r.destroy(scene);
             })
        .def(
            "set_scene",
            [](RenderRuntime &r, SceneSource source, Scene scene) {
                nb::gil_scoped_release release;
                r.setScene(scene, source);
            },
            nb::arg("source"), nb::arg("scene") = Scene{})
        .def(
            "update",
            [](RenderRuntime &r, Array<float, -1, 4, 4> array, uint64_t revision, uint64_t sequence,
               Scene scene) {
                // Synchronous dispatch never calls Python. Reuse this calling thread's
                // owned snapshot only after the backend has consumed it.
                thread_local std::vector<Matrix> transforms;
                transforms.resize(array.shape(0));
                if (!transforms.empty())
                    std::memcpy(transforms.data(), array.data(),
                                transforms.size() * sizeof(Matrix));
                nb::gil_scoped_release release;
                r.update(scene, {revision, sequence, transforms});
            },
            nb::arg("transforms"), nb::arg("source_revision") = 1, nb::arg("sequence") = 0,
            nb::arg("scene") = Scene{})
        .def(
            "update_mesh",
            [](RenderRuntime &r, uint32_t mesh, Array<float, -1, 3> positions,
               Array<float, -1, 3> normals, Scene scene) {
                auto data = vertices(positions, normals);
                nb::gil_scoped_release release;
                r.updateMesh(scene, mesh, data);
            },
            nb::arg("mesh"), nb::arg("positions"), nb::arg("normals"), nb::arg("scene") = Scene{})
        .def(
            "create_target",
            [](RenderRuntime &r, uint32_t width, uint32_t height, uint32_t samples, Scene scene) {
                nb::gil_scoped_release release;
                return r.createTarget(scene, {width, height}, samples);
            },
            nb::arg("width"), nb::arg("height"), nb::arg("samples") = 1, nb::arg("scene") = Scene{})
        .def("resize",
             [](RenderRuntime &r, Target target, uint32_t width, uint32_t height) {
                 nb::gil_scoped_release release;
                 r.resize(target, {width, height});
             })
        .def("destroy",
             [](RenderRuntime &r, Target target) {
                 nb::gil_scoped_release release;
                 r.destroy(target);
             })
        .def(
            "render",
            [](RenderRuntime &r, Target target, CameraView camera, bool color, bool sceneData,
               std::optional<Product> dataProduct) {
                nb::gil_scoped_release release;
                return r.renderRequested(target, camera, {color, sceneData, dataProduct});
            },
            nb::arg("target"), nb::arg("camera"), nb::arg("color") = true,
            nb::arg("scene_data") = true, nb::arg("data_product") = nb::none())
        .def(
            "readback",
            [](RenderRuntime &r, FrameToken frame, Product product, Region region) {
                nb::gil_scoped_release release;
                return r.readback(frame, product, region);
            },
            nb::arg("frame"), nb::arg("product"), nb::arg("region") = Region{})
        .def("read", &RenderRuntime::read, nb::arg("frame"), nb::arg("product"),
             nb::arg("region") = Region{}, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "read_into",
            [](RenderRuntime &r, FrameToken frame, Product product,
               nb::ndarray<nb::numpy, nb::c_contig, nb::device::cpu> output, Region region) {
                const auto type = product == Product::MetricDepth    ? nb::dtype<float>()
                                  : product == Product::ObjectId     ? nb::dtype<uint32_t>()
                                  : product == Product::Segmentation ? nb::dtype<int32_t>()
                                                                     : nb::dtype<uint8_t>();
                const size_t channels = product == Product::Color          ? 3
                                        : product == Product::ColorAlpha   ? 4
                                        : product == Product::Segmentation ? 2
                                                                           : 1;
                if (output.dtype() != type || output.ndim() != (channels == 1 ? 2 : 3) ||
                    (channels != 1 && output.shape(2) != channels) ||
                    output.shape(0) > UINT32_MAX || output.shape(1) > UINT32_MAX)
                    throw std::invalid_argument(
                        "Readback destination has the wrong shape or dtype");
                ImageView destination{product,
                                      {uint32_t(output.shape(1)), uint32_t(output.shape(0))},
                                      {static_cast<std::byte *>(output.data()), output.nbytes()}};
                nb::gil_scoped_release release;
                return r.readInto(frame, destination, region);
            },
            nb::arg("frame"), nb::arg("product"), nb::arg("out").noconvert(),
            nb::arg("region") = Region{})
        .def("poll", &RenderRuntime::poll, nb::call_guard<nb::gil_scoped_release>())
        .def("wait", &RenderRuntime::wait, nb::call_guard<nb::gil_scoped_release>())
        .def("advance", &RenderRuntime::advance, nb::call_guard<nb::gil_scoped_release>())
        .def("reload_shaders", &RenderRuntime::reloadShaders,
             nb::call_guard<nb::gil_scoped_release>())
        .def(
            "create_surface",
            [](RenderRuntime &r, uintptr_t handle, uint32_t width, uint32_t height,
               uintptr_t display, bool wayland) {
                nb::gil_scoped_release release;
                return r.createSurface({reinterpret_cast<void *>(handle),
                                        reinterpret_cast<void *>(display),
                                        {width, height},
                                        wayland ? WindowSystem::Wayland : WindowSystem::Native});
            },
            nb::arg("handle"), nb::arg("width"), nb::arg("height"), nb::arg("display") = 0,
            nb::arg("wayland") = false)
        .def("set_vsync", &RenderRuntime::setVsync, nb::call_guard<nb::gil_scoped_release>())
        .def("target_texture", &RenderRuntime::targetTexture,
             nb::call_guard<nb::gil_scoped_release>())
        .def("upload_texture",
             [](RenderRuntime &r, Array<uint8_t, -1, -1, 4> image) {
                 Extent size{static_cast<uint32_t>(image.shape(1)),
                             static_cast<uint32_t>(image.shape(0))};
                 std::vector<std::byte> data(image.size());
                 if (!data.empty())
                     std::memcpy(data.data(), image.data(), data.size());
                 nb::gil_scoped_release release;
                 return r.uploadTexture(size, data);
             })
        .def("destroy_texture",
             [](RenderRuntime &r, Texture texture) {
                 nb::gil_scoped_release release;
                 r.destroy(texture);
             })
        .def("render_ui",
             [](RenderRuntime &r, uint32_t width, uint32_t height, Array<uint8_t, -1> bytes,
                Array<uint32_t, -1> indices, std::vector<UiCommand> commands, Target target) {
                 static_assert(sizeof(UiVertex) == 20);
                 if (bytes.size() % sizeof(UiVertex))
                     throw std::invalid_argument("Invalid UI vertex byte count");
                 thread_local std::vector<UiVertex> vertices;
                 thread_local std::vector<uint32_t> indexData;
                 vertices.resize(bytes.size() / sizeof(UiVertex));
                 indexData.resize(indices.size());
                 if (!vertices.empty())
                     std::memcpy(vertices.data(), bytes.data(), bytes.size());
                 if (!indexData.empty())
                     std::memcpy(indexData.data(), indices.data(),
                                 indices.size() * sizeof(uint32_t));
                 nb::gil_scoped_release release;
                 return r.renderUi({{width, height}, vertices, indexData, commands}, target);
             })
        .def(
            "__enter__",
            [](RenderRuntime &r) -> RenderRuntime & {
                if (r.closed())
                    throw std::runtime_error("Render runtime is closed");
                return r;
            },
            nb::rv_policy::reference_internal)
        .def(
            "__exit__",
            [](RenderRuntime &r, nb::handle, nb::handle, nb::handle) {
                nb::gil_scoped_release release;
                r.close();
                return false;
            },
            nb::arg("exc_type").none(), nb::arg("exc_value").none(), nb::arg("traceback").none());
}
