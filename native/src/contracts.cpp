#include <algorithm>
#include <cmath>
#include <mojive/render.hpp>
#include <stdexcept>

namespace mojive {
void validate_ui(const UiFrame &ui) {
    if (!ui.size.width || !ui.size.height)
        throw std::invalid_argument("UI extent must be positive");
    for (const auto &command : ui.commands) {
        if (command.first_index > ui.indices.size() ||
            command.index_count > ui.indices.size() - command.first_index ||
            command.vertex_offset > ui.vertices.size())
            throw std::invalid_argument("UI command exceeds uploaded buffers");
        for (float value : command.clip)
            if (!std::isfinite(value))
                throw std::invalid_argument("Non-finite UI clip rectangle");
        for (auto index : ui.indices.subspan(command.first_index, command.index_count))
            if (index >= ui.vertices.size() - command.vertex_offset)
                throw std::invalid_argument("UI index exceeds uploaded vertices");
    }
}

void validate_scene(const SceneSource &scene) {
    for (const auto &mesh : scene.meshes) {
        if (mesh.vertices.empty() || mesh.indices.empty() || mesh.indices.size() % 3)
            throw std::invalid_argument("Meshes require indexed triangles");
        for (auto i : mesh.indices)
            if (i >= mesh.vertices.size())
                throw std::invalid_argument("Invalid vertex index");
        for (const auto &vertex : mesh.vertices)
            for (float p : vertex.position)
                if (!std::isfinite(p))
                    throw std::invalid_argument("Non-finite vertex");
    }
    for (const auto &instance : scene.instances) {
        if (instance.mesh >= scene.meshes.size())
            throw std::invalid_argument("Invalid mesh index");
        for (float c : instance.color)
            if (!std::isfinite(c))
                throw std::invalid_argument("Non-finite color");
    }
}
void validate_frame(const SceneSource &scene, const SceneFrame &frame) {
    if (frame.source_revision != scene.revision ||
        frame.transforms.size() != scene.instances.size())
        throw std::invalid_argument("Frame does not match the scene source");
    for (const auto &matrix : frame.transforms) {
        for (float p : matrix)
            if (!std::isfinite(p))
                throw std::invalid_argument("Non-finite transform");
        if (matrix[12] != 0 || matrix[13] != 0 || matrix[14] != 0 || matrix[15] != 1)
            throw std::invalid_argument("Scene transforms must be affine");
    }
}
using V = std::array<float, 3>;
static V sub(V a, V b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
static V cross(V a, V b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
static float dot(V a, V b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static V normalized(V a) {
    float length = std::sqrt(dot(a, a));
    if (length < 1e-8f)
        throw std::invalid_argument("Degenerate camera basis");
    return {a[0] / length, a[1] / length, a[2] / length};
}
Matrix look_at(V eye, V target, V up) {
    V z = normalized(sub(eye, target)), x = normalized(cross(up, z)), y = cross(z, x);
    return {x[0], x[1], x[2], -dot(x, eye), y[0], y[1], y[2], -dot(y, eye),
            z[0], z[1], z[2], -dot(z, eye), 0,    0,    0,    1};
}
Matrix perspective(float fov, float aspect, float near, float far) {
    if (fov <= 0 || fov >= 3.14159f || aspect <= 0 || near <= 0 || far <= near)
        throw std::invalid_argument("Invalid perspective camera");
    float f = 1 / std::tan(fov / 2);
    return {f / aspect,
            0,
            0,
            0,
            0,
            f,
            0,
            0,
            0,
            0,
            (far + near) / (near - far),
            2 * far * near / (near - far),
            0,
            0,
            -1,
            0};
}
Matrix orthographic(float width, float height, float near, float far) {
    if (width <= 0 || height <= 0 || far <= near)
        throw std::invalid_argument("Invalid orthographic camera");
    return {2 / width,
            0,
            0,
            0,
            0,
            2 / height,
            0,
            0,
            0,
            0,
            -2 / (far - near),
            -(far + near) / (far - near),
            0,
            0,
            0,
            1};
}
} // namespace mojive
