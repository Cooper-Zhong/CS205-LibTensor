#pragma once
#include "tensor.h"

namespace ts
{

    template <typename T>
    // 返回从start到end的等差数列tensor
    Tensor<T> arange(const int start, const int end, const int step = 1)
    {
        int total_size = (end - start) / step;
        T *data = new T[total_size];
        for (int i = 0; i < total_size; i++)
        {
            data[i] = start + i * step;
        }
        Tensor<T> res = Tensor<T>(data, {total_size});
        delete[] data;
        return res;
    }

    template <typename T>
    T &Tensor<T>::enisum_indexing(const std::vector<int> &indices) const // 用于einsum, 根据indices获取tensor的值
    {
        int id = offset;
        for (int i = 0; i < ndim; i++)
        {
            id += indices[i] * stride[i];
        }
        return data.get()[id];
    }

    // 返回两个tensor广播后的形状
    template <typename U>
    std::vector<int> get_broadcast_shape(const std::vector<Tensor<U>> &tensors)
    {
        std::vector<int> result_shape;
        for (const auto &tensor : tensors)
        {
            if (tensor.get_ndim() > result_shape.size())
            {
                result_shape = tensor.get_shape();
            }
        }
        for (const auto &tensor : tensors)
        {
            std::vector<int> cur_shape = tensor.get_shape();
            int diff = result_shape.size() - cur_shape.size(); // 当前tensor和result_shape的维度差

            for (int i = result_shape.size() - 1; i >= 0; --i)
            {                       // 从后往前遍历
                int dim = i - diff; // 当前维度在tensor中的索引
                if (dim >= 0)
                { // 如果当前维度在tensor中存在
                    if (cur_shape[dim] > result_shape[i])
                    {
                        result_shape[i] = cur_shape[dim];
                    }
                    else if (cur_shape[dim] != 1 && cur_shape[dim] != result_shape[i])
                    {
                        throw std::invalid_argument("The tensors cannot be broadcasted.");
                    }
                }
            }
        }
        return result_shape;
    }

    template <typename U> // get_broadcast_shape( {tensor1, tensor2, ...})
    std::vector<int> get_broadcast_shape(const std::initializer_list<Tensor<U>> &tensors)
    {
        std::vector<Tensor<U>> temp = tensors;
        return get_broadcast_shape(temp);
    }

    // template <typename U>
    // Tensor<U> broadcast(const Tensor<U> &cur, const std::vector<int> &broadcast_shape)
    // {

    // }

    template <typename T>
    Tensor<T> einsum(const std::string &equation, const std::vector<Tensor<T>> &tensors)
    {
        std::map<char, int> dims;                  // 用于存储每个维度的大小
        std::map<char, int> cur_index;             // 用于迭代的当前索引
        std::vector<char> dim_iter_order;          // 用于迭代的维度顺序
        std::vector<std::vector<char>> input_dims; // 每个输入Tensor的维度对应的字母 "ijk,ikl->ij" -> {{'i','j','k'},{'i','k','l'}}
        std::vector<char> output_part;             // 输出Tensor的维度对应的字母
        bool is_input_part = true;
        bool is_scalar_output = false;
        std::vector<char> input_part; // 临时存储输入Tensor的维度对应的字母

        // 解析表达式
        const auto arrow_pos = equation.find(">");
        const auto lhs = equation.substr(0, arrow_pos);
        const auto rhs = equation.substr(arrow_pos + 1);
        for (char c : lhs)
        {
            if (c == ' ')
            {
                continue;
            }
            if (c == ',' || c == '-')
            {
                // for(char c : input_part)
                // {
                //     std::cout << c << " ";
                // }
                input_dims.push_back(input_part); // 一个输入Tensor的维度对应的字母
                input_part.clear();
            }
            else
            {
                input_part.push_back(c);
            }
        }
        for (char c : rhs)
        {
            if (c == ' ')
            {
                continue;
            }
            else
            {
                output_part.push_back(c);
            }
        }
        is_scalar_output = output_part.size() == 0; // -> 后面没有字母，说明输出是标量

        for (int i = 0; i < input_dims.size(); i++)
        { // "ijk,ikl->ij" => {{'i','j','k'},{'i','k','l'}}
            for (int j = 0; j < input_dims[i].size(); j++)
            {
                char c = input_dims[i][j];
                if (dims.find(c) == dims.end())
                { // 如果dims中没有这个维度
                    dim_iter_order.push_back(c);
                    dims[c] = tensors[i].get_shape()[j]; // 将这个维度的大小加入dims, 例 维度 i 的大小为 dims[i]
                    cur_index[c] = 0;                    // 将这个维度的当前索引初始化为0
                }
                else
                {
                    // continue;
                    // assert(dims[c] == tensors[i].get_shape()[j]);
                    // 如果dims中已经有这个维度了，那么这个维度的大小应该和当前输入Tensor的维度大小相同
                }
            }
        }
        // 在这里可以创建output矩阵
        std::vector<int> result_shape;
        if (is_scalar_output)
        { // 标量
            result_shape.push_back(1);
        }
        else
        {
            for (char c : output_part)
            {
                result_shape.push_back(dims[c]);
            }
        }
        Tensor<T> result(result_shape);
        // debug
        // for (size_t i = 0; i < input_dims.size(); i++)
        // {
        //     for (size_t j = 0; j < input_dims[i].size(); j++)
        //     {
        //         std::cout << input_dims[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // std::cout << "debug 1 " << std::endl;

        bool done = false;
        while (!done)
        {
            // 创建一个用于存储当前索引的vector
            std::vector<T> index_for_each_input;

            for (int i = 0; i < input_dims.size(); i++)
            {
                std::vector<int> indices;
                for (char dim : input_dims[i])
                {
                    indices.push_back(cur_index[dim]);
                }
                // int start = tensors[i].get_offset();
                // for (int i = 0; i < tensors[i].get_ndim(); i++)
                // {
                //     start += indices[i] * tensors[i].get_stride()[i];
                // }
                index_for_each_input.push_back(tensors[i].enisum_indexing(indices)); // 使用indices获取tensor的值
                // std ::cout << tensors[i].enisum_indexing(indices) << " " << std::endl;
            }
            // std::cout << "debug 2 " << std::endl;
            // for (size_t i = 0; i < index_for_each_input.size(); i++)
            // {
            //     std::cout << index_for_each_input[i] << " ";
            // }

            // 执行计算
            T temp_result = 1; // 假设是乘法操作
            for (auto value : index_for_each_input)
            {
                temp_result *= value;
            }
            // std::cout << temp_result << " " << std::endl;
            // std::cout << "debug 3 " << std::endl;

            if (is_scalar_output)
            {
                // std::cout << "debug 4 " << std::endl;
                // result({0}) = result({0}) + temp_result;
                result.enisum_indexing({0}) = result.enisum_indexing({0}) + temp_result;

                // std::cout << "debug 5 " << std::endl;
            }
            else
            {
                // 根据output_part计算输出索引
                std::vector<int> output_indices;
                for (char dim : output_part)
                {
                    output_indices.push_back(cur_index[dim]);
                }
                // std::cout << "debug 6 " << std::endl;

                // 将结果累加到输出Tensor的相应位置
                // result.enisum_indexing(output_indices) = result.enisum_indexing(output_indices) + temp_result;
                result.enisum_indexing(output_indices) += temp_result;
                // result(output_indices) = result(output_indices) + temp_result;
                // std::cout << "debug 7 " << std::endl;
            }

            // 更新索引
            for (int i = dim_iter_order.size() - 1; i >= 0; i--)
            {
                if (cur_index[dim_iter_order[i]] < dims[dim_iter_order[i]] - 1)
                {                                   // 如果当前索引还没有到达这个维度的最大值
                    cur_index[dim_iter_order[i]]++; // 将当前索引加1
                    break;
                }
                else
                {
                    cur_index[dim_iter_order[i]] = 0; // 将当前索引重置为0
                    if (i == 0)
                    { // 如果当前索引已经到达了最后一个维度的最大值
                        done = true;
                    }
                }
            }
        }
        return result;
    }
} // namespace ts
