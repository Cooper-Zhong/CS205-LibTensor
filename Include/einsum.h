#ifndef EINSUM_H_
#define EINSUM_H_

#include "tensor.h"

namespace ts {

    template<typename T>
    // 返回从start到end的一维tensor
    Tensor<T> arange(const int start, const int end, const int step = 1) {
        int total_size = (end - start) / step;
        Tensor<T> res = Tensor<T>({total_size});
        for (int i = 0; i < total_size; i++) {
            res.get_data()[i] = start + i * step;
        }
        return res;
    }

    template<typename T>
    T &Tensor<T>::enisum_indexing(const std::vector<int> &indices) const // 用于einsum, 根据indices获取tensor的值
    {
        int id = offset;
        for (int i = 0; i < ndim; i++) {
            id += indices[i] * stride[i];
        }
        return data.get()[id];
    }


    /*
    reference: https://github.com/Akito-UzukiP/libtensor : einsum
    */
    template<typename T>
    Tensor<T> einsum(const std::string &equation, const std::vector<Tensor<T>> &tensors) {
        std::map<char, int> dims;                  // 每个维度的大小,dims['i'] 指i对应的维度大小
        std::map<char, int> cur_index;             // 每个维度每一轮遍历的下标
        std::vector<char> dim_iter_order;          // 遍历的维度顺序
        std::vector<std::vector<char>> input_dims; // 每个输入Tensor的维度对应的字母 "ijk,ikl->ij" -> {{'i','j','k'},{'i','k','l'}}
        std::vector<char> output_part;             // 输出Tensor对应的字母
        std::vector<char> input_part; // 临时存储输入Tensor的维度对应的字母
        bool only1dim = false;

        // 解析表达式
        const auto arrow_pos = equation.find(">");
        const auto lhs = equation.substr(0, arrow_pos);
        const auto rhs = equation.substr(arrow_pos + 1);
        for (char c: lhs) {
            if (c == ' ') {
                continue;
            }
            if (c == ',' || c == '-') {
                input_dims.push_back(input_part); // 一个输入Tensor的维度对应的字母
                input_part.clear();
            } else {
                input_part.push_back(c);
            }
        }
        for (char c: rhs) {
            if (c == ' ') {
                continue;
            } else {
                output_part.push_back(c);
            }
        }

        if (input_dims.size() != tensors.size()) {
            throw std::invalid_argument("The number of input tensors != the number of input parts in equation.");
        }

        only1dim = output_part.size() == 0; // -> 后面没有字母，说明输出是标量

        for (int i = 0; i < input_dims.size(); i++) { // "ijk,ikl->ij" => {{'i','j','k'},{'i','k','l'}}
            for (int j = 0; j < input_dims[i].size(); j++) {
                char c = input_dims[i][j];
                if (dims.find(c) == dims.end()) { // 如果dims中没有这个维度
                    dim_iter_order.push_back(c);
                    dims[c] = tensors[i].get_shape()[j]; // 将这个维度的大小加入dims, 例 维度'i'的大小为 dims[i]
                    cur_index[c] = 0;                    // 下标初始化为0
                } else {
                    if (dims[c] != tensors[i].get_shape()[j]) {
                        throw std::invalid_argument("The shape of input tensors is not valid.");
                    }
                }
            }
        }
        std::vector<int> result_shape;
        if (only1dim) { // 标量
            result_shape.push_back(1);
        } else {
            for (char c: output_part) {
                result_shape.push_back(dims[c]);
            }
        }
        Tensor<T> result(result_shape);

        while (true) {
            // sum_result += np_a[i, j, k] * np_b[i, k, l]
            T term = 1; // 每个tensor对应项的乘积
            std::vector<int> indices;
            for (int i = 0; i < tensors.size(); i++) {
                indices.clear();
                for (char dim: input_dims[i]) {
                    indices.push_back(cur_index[dim]); // 某个维度当前遍历到的索引
                }
                term *= tensors[i].enisum_indexing(indices); // 使用indices获取两个/多个tensor的值
            }

            if (only1dim) {
                result.enisum_indexing({0}) += term;//累加
            } else {
                indices.clear();
                for (char dim: output_part) {
                    indices.push_back(cur_index[dim]);
                }
                result.enisum_indexing(indices) += term;//累加
            }

            // 更新下标
            for (int i = dim_iter_order.size() - 1; i >= 0; i--) {
                int id = dim_iter_order[i];
                if (cur_index[id] < dims[id] - 1) {// 当前维度的下标还没有到达这个维度的最大值
                    cur_index[id]++;
                    break;
                } else {
                    cur_index[id] = 0; // 将当前索引重置为0, 继续更新前一个维度的索引
                    if (i == 0) { // 如果当前索引已经到达了最后一个维度的最大值
                        return result;
                    }
                }
            }
        }
    }
} // namespace ts

#endif