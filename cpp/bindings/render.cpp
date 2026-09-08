#include <cstring>
#include <mojive/backends/bgfx.hpp>
#include <mojive/renderRuntime.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
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
nb::object imageArray(const ReadbackResult &result) {
    if (result.state != ReadbackState::Ready)
        return nb::none();
    const auto &image = result.image;
    const size_t h = image.size.height, w = image.size.width;
    const auto *data = image.pixels.data();
    switch (image.product) {
    case Product::Color:
        return ownedArray<uint8_t>(data, {h, w, 3});
    case Product::ColorAlpha:
        return ownedArray<uint8_t>(data, {h, w, 4});
    case Product::ObjectId:
        return ownedArray<uint32_t>(data, {h, w});
    case Product::Segmentation:
        return ownedArray<int32_t>(data, {h, w, 2});
    case Product::MetricDepth:
        return ownedArray<float>(data, {h, w});
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
        .def_ro("submission", &FrameToken::submission);
    nb::class_<ReadbackTicket>(module, "ReadbackTicket").def_ro("id", &ReadbackTicket::id);
    nb::class_<ReadbackResult>(module, "ReadbackResult")
        .def_ro("state", &ReadbackResult::state)
        .def_ro("frame", &ReadbackResult::frame)
        .def_prop_ro("image", &imageArray);
    nb::class_<FrameStats>(module, "FrameStats")
        .def_ro("draw_calls", &FrameStats::drawCalls)
        .def_ro("instances", &FrameStats::instances)
        .def_ro("upload_bytes", &FrameStats::uploadBytes)
        .def_ro("gpu_ms", &FrameStats::gpuMs);
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
        .def_rw("far_plane", &CameraView::farPlane)
        .def_prop_rw(
            "view", [](const CameraView &c) { return ownedArray<float>(c.view.data(), {4, 4}); },
            [](CameraView &c, Array<float, 4, 4> value) { c.view = matrix(value); })
        .def_prop_rw(
            "projection",
            [](const CameraView &c) { return ownedArray<float>(c.projection.data(), {4, 4}); },
            [](CameraView &c, Array<float, 4, 4> value) { c.projection = matrix(value); });
    nb::class_<SceneStyle>(module, "SceneStyle")
        .def(nb::init<>())
        .def_rw("background", &SceneStyle::background)
        .def_rw("textures", &SceneStyle::textures)
        .def_rw("wireframe", &SceneStyle::wireframe)
        .def_rw("transparent_ids", &SceneStyle::transparentIds);
    nb::class_<Material>(module, "Material")
        .def(nb::init<>())
        .def_rw("texture", &Material::texture)
        .def_rw("emission", &Material::emission)
        .def_rw("specular", &Material::specular)
        .def_rw("shininess", &Material::shininess);
    nb::class_<SceneSource>(module, "SceneSource")
        .def(nb::init<>())
        .def_rw("revision", &SceneSource::revision)
        .def_rw("materials", &SceneSource::materials)
        .def_rw("linear_colors", &SceneSource::linearColors)
        .def("add_texture_mips",
             [](SceneSource &s, uint32_t width, uint32_t height, Array<uint8_t, -1> bytes) {
                 TextureSource texture;
                 texture.size = {width, height};
                 texture.mipmaps = true;
                 texture.rgba.resize(bytes.size());
                 if (!texture.rgba.empty())
                     std::memcpy(texture.rgba.data(), bytes.data(), bytes.size());
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
                 auto &out = s.meshes.at(mesh).texcoords;
                 out.resize(uv.shape(0));
                 if (!out.empty())
                     std::memcpy(out.data(), uv.data(), uv.size() * sizeof(float));
             })
        .def("add_texture",
             [](SceneSource &s, Array<uint8_t, -1, -1, 4> pixels) {
                 TextureSource texture;
                 texture.size = {uint32_t(pixels.shape(1)), uint32_t(pixels.shape(0))};
                 texture.rgba.resize(pixels.size());
                 if (!texture.rgba.empty())
                     std::memcpy(texture.rgba.data(), pixels.data(), pixels.size());
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
                s.meshes.push_back(std::move(mesh));
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
        .def(
            "__init__",
            [](RenderRuntime *self, std::string shaders, LogOptions options) {
                nb::gil_scoped_release release;
                new (self) RenderRuntime(
                    [shaders = std::move(shaders)] { return makeBgfxRenderer({{}, shaders}); },
                    options);
            },
            nb::arg("shader_directory"), nb::arg("log_options") = LogOptions{})
        .def_prop_ro("capabilities", [](const RenderRuntime &r) { return r.capabilities(); })
        .def_prop_ro("log", &RenderRuntime::log, nb::rv_policy::reference_internal)
        .def_prop_ro("closed", &RenderRuntime::closed)
        .def("close", &RenderRuntime::close, nb::call_guard<nb::gil_scoped_release>())
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
        .def("render",
             [](RenderRuntime &r, Target target, CameraView camera) {
                 nb::gil_scoped_release release;
                 return r.render(target, camera);
             })
        .def(
            "readback",
            [](RenderRuntime &r, FrameToken frame, Product product, Region region) {
                nb::gil_scoped_release release;
                return r.readback(frame, product, region);
            },
            nb::arg("frame"), nb::arg("product"), nb::arg("region") = Region{})
        .def("poll", &RenderRuntime::poll, nb::call_guard<nb::gil_scoped_release>())
        .def("wait", &RenderRuntime::wait, nb::call_guard<nb::gil_scoped_release>())
        .def("advance", &RenderRuntime::advance, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "create_surface",
            [](RenderRuntime &r, uintptr_t handle, uint32_t width, uint32_t height,
               uintptr_t display) {
                nb::gil_scoped_release release;
                return r.createSurface({reinterpret_cast<void *>(handle),
                                        reinterpret_cast<void *>(display),
                                        {width, height}});
            },
            nb::arg("handle"), nb::arg("width"), nb::arg("height"), nb::arg("display") = 0)
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
