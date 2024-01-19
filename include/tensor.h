#include <vector>
#include <typeinfo> // typeid
#include <string>   // std::string
#include <memory>
#include <stdexcept>
#include <stack>
#include <random>
#include <iostream>
#include <cmath>
#include <map>
#include <cassert>
#include "math.h"
#include "einsum.h"
#include "transform.h"

#ifndef TENSOR_H_
#define TENSOR_H_

namespace ts
{
    // Tensor class
    template <typename T>
    class Tensor
    {
    private:
        std::shared_ptr<T[]> data;
        int offset;
        int ndim;
        int data_length;
        std::vector<int> shape;
        std::vector<int> stride;

    public:
        // Constructor
        Tensor();

        Tensor(const std::vector<int> &_shape);

        Tensor(const T *data, const std::vector<int> &_shape);

        static Tensor<T> rand_tensor(const std::vector<int> &_shape);

        static Tensor<T> zeros_tensor(const std::vector<int> &_shape);

        static Tensor<T> ones_tensor(const std::vector<int> &_shape);

        static Tensor<T> full_tensor(const std::vector<int> &_shape, T t);

        static Tensor<T> eye_tensor(const std::vector<int> &_shape);

        // Destructor
        ~Tensor();

        std::string get_type() const;

        std::shared_ptr<T[]> get_data() const;

        int get_offset() const;

        int get_ndim() const;

        int get_data_length() const;

        const std::vector<int> &get_shape() const;

        const std::vector<int> &get_stride() const;

        Tensor<T> slicing(const std::vector<std::vector<int>> &indices) const;

        // use -1 to indicate all elements
        Tensor<T> indexing(const std::vector<int> &indices) const;

        // alias for slicing
        Tensor<T> operator()(const std::vector<std::vector<int>> &indices) const;

        // alias for indexing
        Tensor<T> operator()(const std::vector<int> &indices) const;

        T &enisum_indexing(const std::vector<int> &indices) const;

        Tensor<T> permute(const std::vector<int> &axes);

        Tensor<T> transpose(const int &dim1, const int &dim2);

        Tensor<T> cat(const Tensor<T> &other, const int &dim);

        bool is_contiguous() const;

        Tensor<T> contiguous() const;

        Tensor<T> reshape(const std::vector<int> &shape);

        Tensor<T> squeeze();

        Tensor<T> unsqueeze(int new_dim);

        Tensor<T> tile(std::vector<int> dims);

        void view(const std::vector<int> &shape);

        template <typename T1>
        friend std::ostream &operator<<(std::ostream &o, Tensor<T1> &t);

        /**
         * @brief 重载运算符
         */
        Tensor<T> &operator=(const Tensor<T> &t);    // copy
        Tensor<bool> operator==(const Tensor<T> &t); // 等于
        Tensor<bool> operator!=(const Tensor<T> &t); // 不等于
        Tensor<bool> operator>(const Tensor<T> &t);  // 大于
        Tensor<bool> operator<(const Tensor<T> &t);  // 小于
        Tensor<bool> operator>=(const Tensor<T> &t); // 大于等于
        Tensor<bool> operator<=(const Tensor<T> &t); // 小于等于
        Tensor<bool> eq(const Tensor<T> &t);         // 等于
        Tensor<bool> ne(const Tensor<T> &t);         // 不等于
        Tensor<bool> gt(const Tensor<T> &t);         // 大于
        Tensor<bool> lt(const Tensor<T> &t);         // 小于
        Tensor<bool> ge(const Tensor<T> &t);         // 大于等于
        Tensor<bool> le(const Tensor<T> &t);         // 小于等于

        static void checkShape(const Tensor<T> &t1, const Tensor<T> &t2); // 检查两个张量的dataType, dim, shape是否相同

        template <typename U>
        friend Tensor<T> einsum(const std::string &equation, const std::vector<Tensor<T>> &tensors); // einsum

        /**
         * @brief reduction operation: sum, mean, max, min
         */

        Tensor<T> sum(const int &dim) const;

        Tensor<T> mean(const int &dim) const;

        Tensor<T> max(const int &dim) const;

        Tensor<T> min(const int &dim) const;

        template <typename U>
        friend Tensor<U> sum(const Tensor<U> &t, const int &dim);

        template <typename U>
        friend Tensor<bool> ne(const Tensor<T> &t1, const Tensor<T> &t2); // 不等于
        template <typename U>
        friend Tensor<bool> gt(const Tensor<T> &t1, const Tensor<T> &t2); // 大于
        template <typename U>
        friend Tensor<bool> lt(const Tensor<T> &t1, const Tensor<T> &t2); // 小于
        template <typename U>
        friend Tensor<U> sum(const Tensor<U> &t, const int &dim);

        template <typename U>
        friend Tensor<U> mean(const Tensor<U> &t, const int &dim);

        template <typename U>
        friend Tensor<U> max(const Tensor<U> &t, const int &dim);

        template <typename U>
        friend Tensor<U> min(const Tensor<U> &t, const int &dim);

        static Tensor<T> re_construct(const Tensor<T> &t);

        Tensor<T> add(const Tensor<T> &t);
        Tensor<T> add(T value);
        Tensor<T> operator+(const Tensor<T> &t);
        Tensor<T> operator+(T value);
        template <typename Y>
        friend Tensor<Y> add(const Tensor<Y> &t1, const Tensor<Y> &t2);
        template <typename Y>
        friend Tensor<Y> add(const Tensor<Y> &t1, Y value);

        Tensor<T> sub(const Tensor<T> &t);
        Tensor<T> sub(T value);
        Tensor<T> operator-(const Tensor<T> &t);
        Tensor<T> operator-(T value);
        template <typename Y>
        friend Tensor<Y> sub(const Tensor<Y> &t1, const Tensor<Y> &t2);
        template <typename Y>
        friend Tensor<Y> sub(const Tensor<Y> &t1, Y value);

        Tensor<T> mul(const Tensor<T> &t);
        Tensor<T> mul(T value);
        Tensor<T> operator*(const Tensor<T> &t);
        Tensor<T> operator*(T value);
        template <typename Y>
        friend Tensor<Y> mul(const Tensor<Y> &t1, const Tensor<Y> &t2);
        template <typename Y>
        friend Tensor<Y> mul(const Tensor<Y> &t1, Y value);

        Tensor<T> div(const Tensor<T> &t);
        Tensor<T> div(T value);
        Tensor<T> operator/(const Tensor<T> &t);
        Tensor<T> operator/(T value);
        template <typename Y>
        friend Tensor<Y> mul(const Tensor<Y> &t1, const Tensor<Y> &t2);
        template <typename Y>
        friend Tensor<Y> mul(const Tensor<Y> &t1, Y value);

        Tensor<T> log(const Tensor<T> &t);
        Tensor<T> log(T value);
        template <typename Y>
        friend Tensor<Y> log(const Tensor<Y> &t1, const Tensor<Y> &t2);
        template <typename Y>
        friend Tensor<Y> log(const Tensor<Y> &t1, Y value);
        };

    // Default Constructor
    template <typename T>
    Tensor<T>::Tensor() : offset(0), ndim(0), data_length(0), shape(std::vector<int>()), stride(std::vector<int>())
    {
        // Initialize data as nullptr
        data = nullptr;
    }

    // Constructor with shape
    template <typename T>
    Tensor<T>::Tensor(const std::vector<int> &_shape) : offset(0), ndim(_shape.size()), shape(_shape),
                                                        stride(std::vector<int>())
    {
        // Calculate data_length and allocate memory for data
        data_length = 1;
        for (int axis : shape)
        {
            data_length *= axis;
        }

        int data_l = data_length;

        for (int i = 0; i < ndim; i++)
        {
            data_l /= shape[i];
            stride.push_back(data_l);
        }

        data = std::shared_ptr<T[]>(new T[data_length]);
        for (int i = 0; i < data_length; i++)
        {
            data[i] = 0;
        }
    }

    // Constructor with existing data and shape
    template <typename T>
    Tensor<T>::Tensor(const T *_data, const std::vector<int> &_shape) : offset(0), ndim(_shape.size()), shape(_shape),
                                                                        stride(std::vector<int>())
    {
        // Calculate data_length and allocate memory for data
        data_length = 1;
        for (int axis : shape)
        {
            data_length *= axis;
        }

        int data_l = data_length;

        for (int i = 0; i < ndim; i++)
        {
            data_l /= shape[i];
            stride.push_back(data_l);
        }

        // Allocate memory for data and copy the content
        data = std::shared_ptr<T[]>(new T[data_length]);
        std::copy(_data, _data + data_length, data.get());
    }

    // rand_tensor method implementation
    template <typename T>
    Tensor<T> Tensor<T>::rand_tensor(const std::vector<int> &_shape)
    {
        Tensor<T> random_tensor(_shape);

        // Generate random numbers for real number types
        std::random_device rd;
        std::mt19937 gen(rd());

        // std::uniform_real_distribution<T> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        std::uniform_real_distribution<T> dis(-10, 10);
        for (int i = 0; i < random_tensor.data_length; i++)
        {
            random_tensor.data[i] = dis(gen);
        }

        return random_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::zeros_tensor(const std::vector<int> &_shape)
    {
        return Tensor<T>(_shape);
    }

    template <typename T>
    Tensor<T> Tensor<T>::ones_tensor(const std::vector<int> &_shape)
    {
        Tensor<T> ones_tensor(_shape);

        // Initialize all elements to one
        for (int i = 0; i < ones_tensor.data_length; i++)
        {
            ones_tensor.data[i] = 1;
        }

        return ones_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::full_tensor(const std::vector<int> &_shape, T t)
    {
        Tensor<T> full_tensor(_shape);

        // Initialize all elements to the specified value 't'
        for (int i = 0; i < full_tensor.data_length; i++)
        {
            full_tensor.data[i] = t;
        }

        return full_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::eye_tensor(const std::vector<int> &_shape)
    {
        Tensor<T> eye_tensor(_shape);

        // Check if the shape is square
        if (_shape.size() < 2 ||
            !std::equal(_shape.begin(), _shape.end() - 1, _shape.begin() + 1, std::equal_to<int>()))
        {
            throw std::invalid_argument("eye_tensor is only supported for square tensors.");
        }

        // Initialize as an identity matrix
        T *data = eye_tensor.get_data().get();
        T *current = data;
        auto stride = eye_tensor.get_stride();

        for (int i = 0; i < _shape[0]; i++)
        {
            current = data;
            for (int j = 0; j < _shape.size(); j++)
            {
                current += i * stride[j];
            }
            *current = 1;
        }

        return eye_tensor;
    }

    // Destructor
    template <typename T>
    Tensor<T>::~Tensor()
    {
        // data is automatically deallocated when the shared_ptr goes out of scope
    }

    // Get the type of the tensor
    template <typename T>
    std::string Tensor<T>::get_type() const
    {
        return typeid(T).name();
    }

    // Get the shared pointer to the data
    template <typename T>
    std::shared_ptr<T[]> Tensor<T>::get_data() const
    {
        return data;
    }

    // Get the offset
    template <typename T>
    int Tensor<T>::get_offset() const
    {
        return offset;
    }

    // Get the number of dimensions
    template <typename T>
    int Tensor<T>::get_ndim() const
    {

        return ndim;
    }

    // Get the data length
    template <typename T>
    int Tensor<T>::get_data_length() const
    {
        return data_length;
    }

    // Get the shape
    template <typename T>
    const std::vector<int> &Tensor<T>::get_shape() const
    {
        return shape;
    }

    // Get the stride
    template <typename T>
    const std::vector<int> &Tensor<T>::get_stride() const
    {
        return stride;
    }

    template <typename T>
    void
    recurse_print(const std::vector<int> &shape, const std::vector<int> &stride, int layer, const T *data, int ndim)
    {
        std::ostream &o = std::cout;
        for (int i = 0; i < layer; i++)
        {
            o << " ";
        }

        o << "[";
        if (layer + 1 == ndim)
        {
            for (int i = 0; i < shape[layer]; i++)
            {
                o << data[i * stride[layer]];
                if (i + 1 < shape[layer])
                {
                    o << ", ";
                }
            }
            o << "]";
            if (layer > 0)
            {
                o << ",";
            }
            o << std::endl;
        }
        else
        {
            o << std::endl;
            for (int i = 0; i < shape[layer]; i++)
            {
                recurse_print(shape, stride, layer + 1, data + stride[layer] * i, ndim);
            }
            for (int i = 0; i < layer; i++)
            {
                o << " ";
            }

            o << "]";
            if (layer > 0)
            {
                o << ",";
            }
            o << std::endl;
        }
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &o, Tensor<T> &t)
    {
        // o << "Data pointer: " << t.get_data() << std::endl;
        // o << "Data length: " << t.get_data_length() << std::endl;
        // o << "Type: " << t.get_type() << std::endl;
        o << "Offset: " << t.get_offset() << " ";
        o << "Ndim:" << t.get_ndim() << " ";
        auto t_shape = t.get_shape();
        o << "Shape: [";
        for (int i = 0; i < t_shape.size(); i++)
        {
            o << t_shape[i];
            if (i + 1 < t_shape.size())
            {
                o << ", ";
            }
        }
        o << "]"
          << " ";

        auto t_stride = t.get_stride();
        o << "Stride: [";
        for (int i = 0; i < t_stride.size(); i++)
        {
            o << t_stride[i];
            if (i + 1 < t_stride.size())
            {
                o << ", ";
            }
        }
        o << "]" << std::endl;

        T *t_data = t.get_data().get() + t.get_offset();

        recurse_print(t_shape, t_stride, 0, t_data, t.get_ndim());

        return o;
    }

    template <typename T>
    Tensor<T> &Tensor<T>::operator=(const Tensor<T> &t)
    {

        if (this->shape.size() != t.shape.size() ||
            !std::equal(this->shape.begin(), this->shape.end(), t.shape.begin(), std::equal_to<int>()))
        {
            throw std::invalid_argument("copy is only supported for tensors of the same shape.");
        }
        else
        {
            std::vector<int> index = std::vector<int>(this->ndim, 0);
            int top = this->ndim - 1;
            T *data_this = this->data.get() + this->offset;
            T *data_copy = t.data.get() + t.offset;
            T *current_this = data_this;
            T *current_copy = data_copy;

            while (index[0] < this->shape[0])
            {

                current_this = data_this;
                current_copy = data_copy;
                for (int i = 0; i < this->ndim; i++)
                {
                    current_this += index[i] * this->stride[i];
                    current_copy += index[i] * t.stride[i];
                }

                *current_this = *current_copy;
                index[top]++;
                while (index[top] == this->shape[top])
                {
                    top--;
                    if (top < 0)
                    {
                        break;
                    }
                    index[top]++;
                }
                if (top < 0)
                {
                    break;
                }
                while (top < this->ndim - 1)
                {
                    top++;
                    index[top] = 0;
                }
            }
        }

        return *this;
    }

} // namespace ts

#endif