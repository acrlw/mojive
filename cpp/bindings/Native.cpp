#include <mojive/Log.hpp>
#include <mojive/Render.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace mojive;
namespace {
auto matrixArray(Matrix value) {
    auto data = std::make_unique<Matrix>(value);
    nb::capsule owner(data.get(),
                      [](void *pointer) noexcept { delete static_cast<Matrix *>(pointer); });
    auto *pointer = data.release();
    return nb::ndarray<nb::numpy, float>(pointer->data(), {4, 4}, owner);
}
} // namespace
#if MOJIVE_HAS_BGFX
void bindRender(nb::module_ &);
#endif
void bindMeshProcessing(nb::module_ &);
NB_MODULE(_native, module) {
    bindMeshProcessing(module);
    module.doc() = "Private Mojive runtime infrastructure; use the compatible Python facade.";
    module.attr("contract_version") = 1;
#if MOJIVE_HAS_BGFX
    module.attr("has_renderer") = true;
#else
    module.attr("has_renderer") = false;
#endif
    nb::enum_<LogLevel>(module, "LogLevel")
        .value("TRACE", LogLevel::Trace)
        .value("DEBUG", LogLevel::Debug)
        .value("INFO", LogLevel::Info)
        .value("WARNING", LogLevel::Warning)
        .value("ERROR", LogLevel::Error)
        .value("CRITICAL", LogLevel::Critical)
        .value("OFF", LogLevel::Off);
    nb::enum_<LogOrigin>(module, "LogOrigin")
        .value("NATIVE", LogOrigin::Native)
        .value("PYTHON", LogOrigin::Python);
    nb::class_<LogOptions>(module, "LogOptions")
        .def(nb::init<>())
        .def_rw("capacity", &LogOptions::capacity)
        .def_rw("byte_capacity", &LogOptions::byteCapacity)
        .def_rw("message_bytes", &LogOptions::messageBytes)
        .def_rw("level", &LogOptions::level)
        .def_rw("file", &LogOptions::file)
        .def_rw("file_bytes", &LogOptions::fileBytes)
        .def_rw("file_count", &LogOptions::fileCount)
        .def_rw("output_queue", &LogOptions::outputQueue)
        .def_rw("stderr_output", &LogOptions::stderrOutput);
    nb::class_<LogRecord>(module, "LogRecord")
        .def_ro("sequence", &LogRecord::sequence)
        .def_ro("runtime_id", &LogRecord::runtimeId)
        .def_ro("thread_id", &LogRecord::threadId)
        .def_ro("timestamp_ns", &LogRecord::timestampNs)
        .def_ro("level", &LogRecord::level)
        .def_ro("origin", &LogRecord::origin)
        .def_ro("component", &LogRecord::component)
        .def_ro("message", &LogRecord::message);
    nb::class_<LogBatch>(module, "LogBatch")
        .def_ro("records", &LogBatch::records)
        .def_ro("next", &LogBatch::next)
        .def_ro("missed", &LogBatch::missed);
    nb::class_<LogStats>(module, "LogStats")
        .def_ro("published", &LogStats::published)
        .def_ro("overwritten", &LogStats::overwritten)
        .def_ro("truncated", &LogStats::truncated)
        .def_ro("output_dropped", &LogStats::outputDropped)
        .def_ro("output_errors", &LogStats::outputErrors)
        .def_ro("retained", &LogStats::retained)
        .def_ro("retained_bytes", &LogStats::retainedBytes)
        .def_ro("closed", &LogStats::closed);
    nb::class_<Log>(module, "Log")
        .def(nb::init<const LogOptions &>(), nb::arg("options") = LogOptions{})
        .def("publish", &Log::publish, nb::arg("level"), nb::arg("component"), nb::arg("message"),
             nb::arg("origin") = LogOrigin::Python, nb::arg("timestamp_ns") = 0,
             nb::call_guard<nb::gil_scoped_release>())
        .def("read", &Log::read, nb::arg("after") = 0, nb::arg("limit") = 256,
             nb::call_guard<nb::gil_scoped_release>())
        .def("stats", &Log::stats, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("runtime_id", &Log::runtimeId)
        .def("close", &Log::close, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "__enter__",
            [](Log &log) -> Log & {
                if (log.stats().closed)
                    throw std::runtime_error("Native log is closed");
                return log;
            },
            nb::rv_policy::reference_internal)
        .def(
            "__exit__",
            [](Log &log, nb::handle, nb::handle, nb::handle) {
                nb::gil_scoped_release release;
                log.close();
                return false;
            },
            nb::arg("exc_type").none(), nb::arg("exc_value").none(), nb::arg("traceback").none());
    module.def("look_at",
               [](std::array<float, 3> eye, std::array<float, 3> target, std::array<float, 3> up) {
                   return matrixArray(lookAt(eye, target, up));
               });
    module.def("perspective", [](float fov, float aspect, float near, float far) {
        return matrixArray(perspective(fov, aspect, near, far));
    });
    module.def("orthographic", [](float width, float height, float near, float far) {
        return matrixArray(orthographic(width, height, near, far));
    });
#if MOJIVE_HAS_BGFX
    bindRender(module);
#endif
}
