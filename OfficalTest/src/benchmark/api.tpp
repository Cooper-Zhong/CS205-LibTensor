#include "api.hpp"

// include your implementation's header file here, e.g.
// #include "../my_tensor.hpp"

namespace bm {

    template<typename T>
    ts::Tensor<T> create_with_data(const std::vector<size_t> &shape, const T *data) {
        // do necessary conversions and call your implementation, e.g.

        // int dim_ = static_cast<int>(shape.size());
        // int shape_[dim_];
        // for (int i = 0; i < dim_; i++) {
        //     shape_[i] = static_cast<int>(shape[i]);
        // }
        // return ts::Tensor<T>(dim_, shape_, data);

        // TODO
    }

    template<typename T>
    ts::Tensor<T> rand(const std::vector<size_t> &shape) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> zeros(const std::vector<size_t> &shape) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> ones(const std::vector<size_t> &shape) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> full(const std::vector<size_t> &shape, const T &value) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> eye(size_t rows, size_t cols) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> slice(const ts::Tensor<T> &tensor, const std::vector<std::pair<size_t, size_t>> &slices) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> concat(const std::vector<ts::Tensor<T>> &tensors, size_t axis) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> tile(const ts::Tensor<T> &tensor, const std::vector<size_t> &shape) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> transpose(const ts::Tensor<T> &tensor, size_t dim1, size_t dim2) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> permute(const ts::Tensor<T> &tensor, const std::vector<size_t> &permutation) {
        // TODO
    }

    template<typename T>
    T at(const ts::Tensor<T> &tensor, const std::vector<size_t> &indices) {
        // TODO
    }

    template<typename T>
    void set_at(ts::Tensor<T> &tensor, const std::vector<size_t> &indices, const T &value) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> pointwise_add(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> pointwise_sub(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> pointwise_mul(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> pointwise_div(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> pointwise_log(const ts::Tensor<T> &tensor) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> reduce_sum(const ts::Tensor<T> &tensor, size_t axis) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> reduce_mean(const ts::Tensor<T> &tensor, size_t axis) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> reduce_max(const ts::Tensor<T> &tensor, size_t axis) {
        // TODO
    }

    template<typename T>
    ts::Tensor<T> reduce_min(const ts::Tensor<T> &tensor, size_t axis) {
        // TODO
    }

    // you may modify the following functions' implementation if necessary

    template<typename T>
    ts::Tensor<bool> eq(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        return a == b;
    }

    template<typename T>
    ts::Tensor<bool> ne(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        return a != b;
    }

    template<typename T>
    ts::Tensor<bool> gt(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        return a > b;
    }

    template<typename T>
    ts::Tensor<bool> ge(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        return a >= b;
    }

    template<typename T>
    ts::Tensor<bool> lt(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        return a < b;
    }

    template<typename T>
    ts::Tensor<bool> le(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        return a <= b;
    }
}
