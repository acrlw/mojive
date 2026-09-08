#include "workload.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
namespace nb = nanobind;
using namespace mojive::binding_probe;
using Array = nb::ndarray<nb::numpy, const double, nb::c_contig, nb::device::cpu>;
using Matrices = nb::ndarray<nb::numpy, const float, nb::c_contig, nb::device::cpu>;
NB_MODULE(_mojive_nanobind_probe, m) {
    m.def("ping", &ping);
    m.def(
        "address", [](Array a) { return reinterpret_cast<uintptr_t>(a.data()); },
        nb::arg("array").noconvert());
    nb::class_<OwnedState>(m, "OwnedState")
        .def(
            "__init__",
            [](OwnedState *self, Array a) {
                if (a.ndim() != 1)
                    throw std::invalid_argument("Expected a vector");
                new (self) OwnedState({a.data(), a.size()});
            },
            nb::arg("array").noconvert())
        .def("work", &OwnedState::work, nb::call_guard<nb::gil_scoped_release>())
        .def("view", [](nb::object self) {
            auto &state = nb::cast<OwnedState &>(self);
            auto values = state.values();
            return nb::ndarray<nb::numpy, const double>(values.data(), {values.size()}, self);
        });
    nb::class_<FrameBatch>(m, "FrameBatch")
        .def(nb::init<size_t>())
        .def(
            "pack",
            [](FrameBatch &batch, Matrices a) {
                if (a.ndim() != 3 || a.shape(1) != 4 || a.shape(2) != 4)
                    throw std::invalid_argument("Expected Nx4x4 matrices");
                nb::gil_scoped_release release;
                return batch.pack({a.data(), a.size()});
            },
            nb::arg("matrices").noconvert())
        .def("bytes", &FrameBatch::bytes);
    m.def("state_size", [](const OwnedState &state) { return state.values().size(); });
}
