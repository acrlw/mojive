#include "workload.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace mojive::bindingProbe;
using Array = py::array_t<double, py::array::c_style>;
using Matrices = py::array_t<float, py::array::c_style>;
PYBIND11_MODULE(_mojive_pybind_probe, m) {
    m.def("ping", &ping);
    m.def(
        "address", [](Array a) { return reinterpret_cast<uintptr_t>(a.data()); },
        py::arg("array").noconvert());
    py::class_<OwnedState>(m, "OwnedState")
        .def(py::init([](Array a) {
                 if (a.ndim() != 1)
                     throw std::invalid_argument("Expected a vector");
                 return OwnedState({a.data(), size_t(a.size())});
             }),
             py::arg("array").noconvert())
        .def("work", &OwnedState::work, py::call_guard<py::gil_scoped_release>())
        .def("view", [](py::object self) {
            auto &state = self.cast<OwnedState &>();
            auto values = state.values();
            py::array out(py::dtype::of<double>(), {values.size()}, {sizeof(double)}, values.data(),
                          self);
            out.attr("setflags")(false);
            return out;
        });
    py::class_<FrameBatch>(m, "FrameBatch")
        .def(py::init<size_t>())
        .def(
            "pack",
            [](FrameBatch &batch, Matrices a) {
                if (a.ndim() != 3 || a.shape(1) != 4 || a.shape(2) != 4)
                    throw std::invalid_argument("Expected Nx4x4 matrices");
                py::gil_scoped_release release;
                return batch.pack({a.data(), size_t(a.size())});
            },
            py::arg("matrices").noconvert())
        .def("bytes", &FrameBatch::bytes);
    m.def("state_size", [](const OwnedState &state) { return state.values().size(); });
}
