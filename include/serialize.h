#pragma once
#include "tensor.h"
#include <fstream>

namespace ts
{
    template <typename T>
    void Tensor<T>::serialize(std::string filename) {
        std::ofstream os(filename);
        // 序列化数据成员
        os.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
        os.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        os.write(reinterpret_cast<const char*>(&data_length), sizeof(data_length));
        os.write(reinterpret_cast<const char*>(shape.data()), ndim * sizeof(int));
        os.write(reinterpret_cast<const char*>(stride.data()), ndim * sizeof(int));
        os.write(reinterpret_cast<const char*>(data.get()), data_length * sizeof(T));

    }

    template <typename T>
    Tensor<T> Tensor<T>::deserialize(std::string filename){
        std::ifstream is(filename);

        if(is.fail()){
            std::cout << "Error opening file" << std::endl;
            exit(1);
        }
        Tensor<T> tensor;
        // 反序列化数据成员
        is.read(reinterpret_cast<char*>(&tensor.offset), sizeof(tensor.offset));
        is.read(reinterpret_cast<char*>(&tensor.ndim), sizeof(tensor.ndim));
        is.read(reinterpret_cast<char*>(&tensor.data_length), sizeof(tensor.data_length));
        tensor.shape.resize(tensor.ndim);
        tensor.stride.resize(tensor.ndim);
        tensor.data.reset(new T[tensor.data_length]);

        is.read(reinterpret_cast<char*>(tensor.shape.data()), tensor.ndim * sizeof(int));
        is.read(reinterpret_cast<char*>(tensor.stride.data()), tensor.ndim * sizeof(int));
        is.read(reinterpret_cast<char*>(tensor.data.get()), tensor.data_length * sizeof(T));

        return tensor;
    }
}