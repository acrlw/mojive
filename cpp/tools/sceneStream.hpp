#pragma once
#include <bit>
#include <filesystem>
#include <fstream>
#include <mojive/render.hpp>
#include <stdexcept>

namespace mojive::probe {
struct Trajectory {
    SceneSource source;
    CameraView camera;
    std::vector<std::vector<Matrix>> frames;
};
inline Trajectory load(const std::filesystem::path &path) {
    static_assert(std::endian::native == std::endian::little);
    static_assert(sizeof(Vertex) == 24 && sizeof(Instance) == 32 && sizeof(Matrix) == 64);
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("Cannot open probe trajectory");
    auto read = [&](void *output, size_t bytes) {
        stream.read(static_cast<char *>(output), bytes);
        if (!stream)
            throw std::runtime_error("Truncated probe trajectory");
    };
    auto number = [&] {
        uint32_t value;
        read(&value, 4);
        return value;
    };
    char magic[8];
    read(magic, 8);
    if (std::string(magic, 8) != "MJVPROB1")
        throw std::runtime_error("Unsupported probe trajectory version");
    auto meshes = number(), instances = number(), frames = number();
    if (meshes > 10000 || instances > 100000 || !frames || frames > 1000 ||
        uint64_t(instances) * frames > 10000000)
        throw std::runtime_error("Probe trajectory exceeds resource limits");
    Trajectory result;
    read(result.camera.view.data(), 64);
    read(result.camera.projection.data(), 64);
    read(&result.camera.farPlane, 4);
    for (uint32_t m = 0; m < meshes; ++m) {
        auto vertices = number(), indices = number();
        if (vertices > 10000000 || indices > 30000000)
            throw std::runtime_error("Probe mesh exceeds resource limits");
        Mesh mesh;
        mesh.vertices.resize(vertices);
        mesh.indices.resize(indices);
        read(mesh.vertices.data(), vertices * sizeof(Vertex));
        read(mesh.indices.data(), indices * 4);
        result.source.meshes.push_back(std::move(mesh));
    }
    result.source.instances.resize(instances);
    read(result.source.instances.data(), instances * sizeof(Instance));
    result.frames.resize(frames);
    for (auto &frame : result.frames) {
        frame.resize(instances);
        read(frame.data(), instances * sizeof(Matrix));
    }
    validateScene(result.source);
    for (auto &frame : result.frames)
        validateFrame(result.source, {1, 0, frame});
    if (stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Trailing probe trajectory bytes");
    return result;
}
} // namespace mojive::probe
