#include "api.hpp"

#include "../tensor.h"

namespace bm
{

    std::vector<int> int_shape(const std::vector<size_t> &shape)
    {
        std::vector<int> ret;
        for (auto &i : shape)
        {
            ret.push_back(static_cast<int>(i));
        }
        return ret;
    }

    std::vector<std::vector<int>> int_slices(const std::vector<std::pair<size_t, size_t>> &slices)
    {
        std::vector<std::vector<int>> ret;

        for (auto &i : slices)
        {
            ret.push_back({static_cast<int>(i.first), static_cast<int>(i.second)});
        }
        return ret;
    }

    template <typename T>
    ts::Tensor<T> create_with_data(const std::vector<size_t> &shape, const T *data)
    {
        // do necessary conversions and call your implementation, e.g.

        // int dim_ = static_cast<int>(shape.size());
        // int shape_[dim_];
        // for (int i = 0; i < dim_; i++) {
        //     shape_[i] = static_cast<int>(shape[i]);
        // }
        // return ts::Tensor<T>(dim_, shape_, data);

        return ts::Tensor<T>(data, int_shape(shape));
    }

    template <typename T>
    ts::Tensor<T> rand(const std::vector<size_t> &shape)
    {
        return ts::Tensor<T>::rand_tensor(int_shape(shape));
    }

    template <typename T>
    ts::Tensor<T> zeros(const std::vector<size_t> &shape)
    {
        return ts::Tensor<T>::zeros_tensor(int_shape(shape));
    }

    template <typename T>
    ts::Tensor<T> ones(const std::vector<size_t> &shape)
    {
        return ts::Tensor<T>::ones_tensor(int_shape(shape));
    }

    template <typename T>
    ts::Tensor<T> full(const std::vector<size_t> &shape, const T &value)
    {
        return ts::Tensor<T>::full_tensor(int_shape(shape), value);
    }

    template <typename T>
    ts::Tensor<T> eye(size_t rows, size_t cols)
    {
        return ts::Tensor<T>::eye_tensor({static_cast<int>(rows), static_cast<int>(cols)});
    }

    template <typename T>
    ts::Tensor<T> slice(const ts::Tensor<T> &tensor, const std::vector<std::pair<size_t, size_t>> &slices)
    {
        return tensor.slicing(int_slices(slices));
    }

    template <typename T>
    ts::Tensor<T> concat(const std::vector<ts::Tensor<T>> &tensors, size_t axis)
    {
        ts::Tensor<T> result = tensors[0];
        for (size_t i = 1; i < tensors.size(); i++)
        {
            result = result.cat(tensors[i], axis);
        }
        return result;
    }

    template <typename T>
    ts::Tensor<T> tile(const ts::Tensor<T> &tensor, const std::vector<size_t> &shape)
    {
        return tensor.tile(int_shape(shape));
    }

    template <typename T>
    ts::Tensor<T> transpose(const ts::Tensor<T> &tensor, size_t dim1, size_t dim2)
    {
        return tensor.transpose(dim1, dim2);
    }

    template <typename T>
    ts::Tensor<T> permute(const ts::Tensor<T> &tensor, const std::vector<size_t> &permutation)
    {
        return tensor.permute(int_shape(permutation));
    }

    template <typename T>
    T at(const ts::Tensor<T> &tensor, const std::vector<size_t> &indices)
    {
        return tensor.at(int_shape(indices));
    }

    template <typename T>
    void set_at(ts::Tensor<T> &tensor, const std::vector<size_t> &indices, const T &value)
    {
        tensor[int_shape(indices)] = value;
    }

    template <typename T>
    ts::Tensor<T> pointwise_add(const ts::Tensor<T> &a, const ts::Tensor<T> &b)
    {
        return a + b;
    }

    template <typename T>
    ts::Tensor<T> pointwise_sub(const ts::Tensor<T> &a, const ts::Tensor<T> &b)
    {
        return a - b;
    }

    template <typename T>
    ts::Tensor<T> pointwise_mul(const ts::Tensor<T> &a, const ts::Tensor<T> &b)
    {
        return a * b;
    }

    template <typename T>
    ts::Tensor<T> pointwise_div(const ts::Tensor<T> &a, const ts::Tensor<T> &b)
    {
        return a / b;
    }

    template <typename T>
    ts::Tensor<T> pointwise_log(const ts::Tensor<T> &tensor)
    {
        return tensor.log();
    }

    template <typename T>
    ts::Tensor<T> reduce_sum(const ts::Tensor<T> &tensor, size_t axis)
    {
        return tensor.sum(static_cast<int>(axis));
    }

    template <typename T>
    ts::Tensor<T> reduce_mean(const ts::Tensor<T> &tensor, size_t axis)
    {
        return tensor.mean(static_cast<int>(axis));
    }

    template <typename T>
    ts::Tensor<T> reduce_max(const ts::Tensor<T> &tensor, size_t axis)
    {
        return tensor.max(static_cast<int>(axis));
    }

    template <typename T>
    ts::Tensor<T> reduce_min(const ts::Tensor<T> &tensor, size_t axis)
    {
        return tensor.min(static_cast<int>(axis));
    }

    // you may modify the following functions' implementation if necessary

    template <typename T>
    ts::Tensor<bool> eq(const ts::Tensor<T> &a, const ts::Tensor<T> &b)
    {
        return a == b;
    }

    template <typename T>
    ts::Tensor<bool> ne(const ts::Tensor<T> &a, const ts::Tensor<T> &b)
    {
        return a != b;
    }

    template <typename T>
    ts::Tensor<bool> gt(const ts::Tensor<T> &a, const ts::Tensor<T> &b)
    {
        return a > b;
    }

    template <typename T>
    ts::Tensor<bool> ge(const ts::Tensor<T> &a, const ts::Tensor<T> &b)
    {
        return a >= b;
    }

    template <typename T>
    ts::Tensor<bool> lt(const ts::Tensor<T> &a, const ts::Tensor<T> &b)
    {
        return a < b;
    }

    template <typename T>
    ts::Tensor<bool> le(const ts::Tensor<T> &a, const ts::Tensor<T> &b)
    {
        return a <= b;
    }
}
