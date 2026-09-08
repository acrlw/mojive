#include <mojive/MeshProcessing.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <memory>

namespace nb = nanobind;
namespace {
template <size_t Columns>
using Rows = nb::ndarray<const float, nb::c_contig, nb::shape<-1, Columns>>;
using Indices = nb::ndarray<const uint32_t, nb::c_contig, nb::ndim<1>>;
template <size_t Columns> auto span(Rows<Columns> value) {
    return std::span(reinterpret_cast<const std::array<float, Columns> *>(value.data()),
                     value.shape(0));
}
} // namespace
void bindMeshProcessing(nb::module_ &module) {
    module.def(
        "simplify_mesh_indices",
        [](Rows<3> positions, Rows<3> normals, Rows<2> uvs, Indices indices, float ratio,
           float maxError) {
            auto result = std::make_unique<mojive::SimplifiedIndices>();
            {
                nb::gil_scoped_release release;
                *result = mojive::simplifyMesh(span<3>(positions), span<3>(normals), span<2>(uvs),
                                               {indices.data(), indices.size()}, ratio, maxError);
            }
            const float error = result->relativeError;
            nb::capsule owner(result.get(), [](void *data) noexcept {
                delete static_cast<mojive::SimplifiedIndices *>(data);
            });
            auto *data = result.release();
            auto array = nb::ndarray<nb::numpy, uint32_t>(data->indices.data(),
                                                          {data->indices.size()}, owner);
            return nb::make_tuple(array, error);
        },
        nb::arg("positions").noconvert(), nb::arg("normals").noconvert(),
        nb::arg("uvs").noconvert(), nb::arg("indices").noconvert(), nb::arg("ratio"),
        nb::arg("max_error"));
}
