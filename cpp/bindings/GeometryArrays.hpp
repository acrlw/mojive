#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <memory>
#include <vector>

namespace mojive::geometry2d {
template <class Scalar, class Value>
nanobind::ndarray<nanobind::numpy, Scalar> ownedArray(std::vector<Value> values,
                                                      size_t columns = 0) {
    auto storage = std::make_unique<std::vector<Value>>(std::move(values));
    size_t shape[]{storage->size(), columns};
    nanobind::capsule owner(storage.get(), [](void *pointer) noexcept {
        delete static_cast<std::vector<Value> *>(pointer);
    });
    auto *data = reinterpret_cast<Scalar *>(storage.release()->data());
    return {data, columns ? 2u : 1u, shape, owner};
}
} // namespace mojive::geometry2d
