#include <algorithm>
#include <cmath>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <mojive/render.hpp>
#include <numbers>
#include <stdexcept>

namespace mojive {
void validateUi(const UiFrame &ui) {
    if (!ui.size.width || !ui.size.height)
        throw std::invalid_argument("UI extent must be positive");
    for (const auto &command : ui.commands) {
        if (command.firstIndex > ui.indices.size() ||
            command.indexCount > ui.indices.size() - command.firstIndex ||
            command.vertexOffset > ui.vertices.size())
            throw std::invalid_argument("UI command exceeds uploaded buffers");
        for (float value : command.clip)
            if (!std::isfinite(value))
                throw std::invalid_argument("Non-finite UI clip rectangle");
        for (auto index : ui.indices.subspan(command.firstIndex, command.indexCount))
            if (index >= ui.vertices.size() - command.vertexOffset)
                throw std::invalid_argument("UI index exceeds uploaded vertices");
    }
}

void validateScene(const SceneSource &scene) {
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
void validateCamera(const CameraView &camera) {
    if (!std::isfinite(camera.farPlane) || camera.farPlane <= 0)
        throw std::invalid_argument("Invalid camera far plane");
    for (const auto *matrix : {&camera.view, &camera.projection})
        for (float value : *matrix)
            if (!std::isfinite(value))
                throw std::invalid_argument("Non-finite camera matrix");
}
void validateFrame(const SceneSource &scene, const SceneFrame &frame) {
    if (frame.sourceRevision != scene.revision || frame.transforms.size() != scene.instances.size())
        throw std::invalid_argument("Frame does not match the scene source");
    for (const auto &matrix : frame.transforms) {
        for (float p : matrix)
            if (!std::isfinite(p))
                throw std::invalid_argument("Non-finite transform");
        if (matrix[12] != 0 || matrix[13] != 0 || matrix[14] != 0 || matrix[15] != 1)
            throw std::invalid_argument("Scene transforms must be affine");
    }
}
namespace {
Matrix rowMajor(const glm::mat4 &matrix) {
    Matrix result;
    for (size_t row = 0; row < 4; ++row)
        for (size_t column = 0; column < 4; ++column)
            result[4 * row + column] = matrix[column][row];
    return result;
}
bool finite(std::initializer_list<float> values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}
} // namespace
Matrix lookAt(std::array<float, 3> eye, std::array<float, 3> target, std::array<float, 3> up) {
    if (!finite({eye[0], eye[1], eye[2], target[0], target[1], target[2], up[0], up[1], up[2]}))
        throw std::invalid_argument("Non-finite camera basis");
    glm::vec3 e(eye[0], eye[1], eye[2]), t(target[0], target[1], target[2]), u(up[0], up[1], up[2]);
    const auto direction = t - e;
    const auto length = glm::length(direction);
    if (!std::isfinite(length) || length < 1e-8f)
        throw std::invalid_argument("Degenerate camera basis");
    const auto sideLength = glm::length(glm::cross(u, direction / length));
    if (!std::isfinite(sideLength) || sideLength < 1e-8f)
        throw std::invalid_argument("Degenerate camera basis");
    return rowMajor(glm::lookAtRH(e, t, u));
}
Matrix perspective(float fov, float aspect, float near, float far) {
    if (!finite({fov, aspect, near, far}) || fov <= 0 || fov >= std::numbers::pi_v<float> ||
        aspect <= 0 || near <= 0 || far <= near)
        throw std::invalid_argument("Invalid perspective camera");
    return rowMajor(glm::perspectiveRH_NO(fov, aspect, near, far));
}
Matrix orthographic(float width, float height, float near, float far) {
    if (!finite({width, height, near, far}) || width <= 0 || height <= 0 || far <= near)
        throw std::invalid_argument("Invalid orthographic camera");
    return rowMajor(glm::orthoRH_NO(-width / 2, width / 2, -height / 2, height / 2, near, far));
}
} // namespace mojive
