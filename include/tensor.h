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

    template <typename T>
    Tensor<T> Tensor<T>::slicing(const std::vector<std::vector<int>> &indices) const
    {
        // Check if the number of indices is equal to the number of dimensions
        if (indices.size() != ndim)
        {
            throw std::invalid_argument("The number of indices must be equal to the number of dimensions.");
        }

        // Check if the indices are valid
        for (int i = 0; i < ndim; i++)
        {
            if (indices[i].size() != 2)
            {
                throw std::invalid_argument("The number of indices for each dimension must be equal to 2.");
            }
            if (indices[i][0] < 0 || indices[i][0] > shape[i] || indices[i][1] < 0 || indices[i][1] > shape[i])
            {
                throw std::invalid_argument("Slicing: The indices are out of range.");
            }
        }

        // Calculate the new shape
        std::vector<int> new_shape;
        for (int i = 0; i < ndim; i++)
        {
            new_shape.push_back(indices[i][1] - indices[i][0]);
        }

        // Calculate the new offset
        int new_offset = offset;
        for (int i = 0; i < ndim; i++)
        {
            new_offset += indices[i][0] * stride[i];
        }

        // Create a new tensor
        Tensor<T> new_tensor;
        new_tensor.data = data;
        new_tensor.offset = new_offset;
        new_tensor.ndim = ndim;
        new_tensor.data_length = data_length;
        new_tensor.shape = new_shape;
        new_tensor.stride = stride;

        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator()(const std::vector<std::vector<int>> &indices) const
    {
        return slicing(indices);
    }

    template <typename T>
    Tensor<T> Tensor<T>::indexing(const std::vector<int> &indices) const
    {
        // Check if the number of indices is equal to the number of dimensions
        if (indices.size() != ndim)
        {
            throw std::invalid_argument("The number of indices must be equal to the number of dimensions.");
        }

        // Check if the indices are valid
        for (int i = 0; i < ndim; i++)
        {
            if (indices[i] < -1 || indices[i] >= shape[i]) // -1 means all elements
            {
                throw std::invalid_argument("The indices are out of range.");
            }
        }

        // Calculate the new shape
        std::vector<int> new_shape;
        for (int i = 0; i < ndim; i++)
        {
            if (indices[i] == -1)
            {
                new_shape.push_back(shape[i]);
            }
            else
            {
                new_shape.push_back(1);
            }
        }

        // Calculate the new offset
        int new_offset = offset;
        for (int i = 0; i < ndim; i++)
        {
            if (indices[i] != -1)
            {
                new_offset += indices[i] * stride[i];
            }
        }

        // Create a new tensor
        Tensor<T> new_tensor;
        new_tensor.data = data;
        new_tensor.offset = new_offset;
        new_tensor.ndim = ndim;
        new_tensor.data_length = data_length;
        new_tensor.shape = new_shape;
        new_tensor.stride = stride;

        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator()(const std::vector<int> &indices) const
    {
        return indexing(indices);
    }

    template <typename T>
    Tensor<T> Tensor<T>::permute(const std::vector<int> &axes)
    {
        // Check if the number of axes is equal to the number of dimensions
        if (axes.size() != ndim)
        {
            throw std::invalid_argument("The number of axes must be equal to the number of dimensions.");
        }

        // Check if the axes are valid
        std::vector<bool> check(ndim, false);
        for (int axis : axes)
        {
            if (axis < 0 || axis >= ndim)
            {
                throw std::invalid_argument("The axes are out of range.");
            }
            if (check[axis])
            {
                throw std::invalid_argument("The axes must be unique.");
            }
            check[axis] = true;
        }

        // Calculate the new shape
        std::vector<int> new_shape;
        for (int axis : axes)
        {
            new_shape.push_back(shape[axis]);
        }

        // Calculate the new stride
        std::vector<int> new_stride;
        for (int axis : axes)
        {
            new_stride.push_back(stride[axis]);
        }

        // Create a new tensor
        Tensor<T> new_tensor;
        new_tensor.data = data;
        new_tensor.offset = offset;
        new_tensor.ndim = ndim;
        new_tensor.data_length = data_length;
        new_tensor.shape = new_shape;
        new_tensor.stride = new_stride;

        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::transpose(const int &dim1, const int &dim2)
    {
        // Check if the number of axes is equal to the number of dimensions
        if (dim1 < 0 || dim1 >= ndim || dim2 < 0 || dim2 >= ndim)
        {
            throw std::invalid_argument("The axes are out of range.");
        }

        // Calculate the new shape
        std::vector<int> new_shape = shape;
        std::swap(new_shape[dim1], new_shape[dim2]);

        // Calculate the new stride
        std::vector<int> new_stride = stride;
        std::swap(new_stride[dim1], new_stride[dim2]);

        // Create a new tensor
        Tensor<T> new_tensor;
        new_tensor.data = data;
        new_tensor.offset = offset;
        new_tensor.ndim = ndim;
        new_tensor.data_length = data_length;
        new_tensor.shape = new_shape;
        new_tensor.stride = new_stride;

        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::cat(const Tensor<T> &other, const int &dim)
    {
        // Check if the number of dimensions is equal
        if (ndim != other.ndim)
        {
            throw std::invalid_argument("The number of dimensions must be equal.");
        }

        // Check if the dimension is valid
        if (dim < 0 || dim >= ndim)
        {
            throw std::invalid_argument("The dimension is out of range.");
        }

        // Check if the shapes are valid
        std::vector<int> new_shape = shape;
        std::vector<int> other_shape = other.shape;
        for (int i = 0; i < ndim; i++)
        {
            if (i != dim)
            {
                if (new_shape[i] != other_shape[i])
                {
                    throw std::invalid_argument("The shapes are not valid.");
                }
            }
        }
        new_shape[dim] += other_shape[dim];

        Tensor<T> new_tensor(new_shape);
        std::vector<std::vector<int>> slicing_indices_current;
        std::vector<std::vector<int>> slicing_indices_other;
        for (int i = 0; i < ndim; i++)
        {
            if (i != dim)
            {
                slicing_indices_current.push_back({0, shape[i]});
                slicing_indices_other.push_back({0, other.shape[i]});
            }
            else
            {
                slicing_indices_current.push_back({0, shape[i]});
                slicing_indices_other.push_back({shape[i], shape[i] + other.shape[i]});
            }
        }

        new_tensor(slicing_indices_current) = *this;
        new_tensor(slicing_indices_other) = other;

        return new_tensor;
    }

    template <typename T>
    bool Tensor<T>::is_contiguous() const
    {
        size_t true_data_length = 1;
        for (int axis : shape)
        {
            true_data_length *= axis;
        }

        if (true_data_length != data_length)
        {
            return false;
        }

        int stride = 1;
        for (int i = ndim - 1; i >= 0; i--)
        {
            if (stride != this->stride[i])
            {
                return false;
            }
            stride *= shape[i];
        }
        return true;
    }

    template <typename T>
    Tensor<T> Tensor<T>::contiguous() const
    {
        // Check contiguous of the tensor
        if (is_contiguous())
        {
            return *this;
        }

        // Create a new tensor
        Tensor<T> new_tensor = Tensor<T>(shape);

        // Copy data element by element
        new_tensor = *this;

        new_tensor = *this;

        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::reshape(const std::vector<int> &shape)
    {
        // check contiguous of the tensor
        if (!is_contiguous())
        {
            throw std::invalid_argument("The tensor is not contiguous.");
        }

        // Check if the number of elements is equal
        int num_elements = 1;
        for (int axis : shape)
        {
            num_elements *= axis;
        }
        if (num_elements != data_length) // contiguous tensor must hava true data_length
        {
            throw std::invalid_argument("The number of elements must be equal.");
        }

        // Create a new tensor
        Tensor<T> new_tensor(shape);
        new_tensor.data = data;
        new_tensor.offset = offset;
        new_tensor.data_length = data_length;

        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::squeeze()
    {
        // Check if the tensor is a vector
        if (ndim == 1)
        {
            throw std::invalid_argument("The tensor is already a vector.");
        }

        // Calculate the new shape
        std::vector<int> new_shape;
        for (int axis : shape)
        {
            if (axis != 1)
            {
                new_shape.push_back(axis);
            }
        }

        // Create a new tensor
        Tensor<T> new_tensor;
        new_tensor.data = data;
        new_tensor.offset = offset;
        new_tensor.ndim = new_shape.size();
        new_tensor.data_length = data_length;
        new_tensor.shape = new_shape;

        // Calculate the new stride
        std::vector<int> new_stride;
        int data_l = data_length;
        for (int i = 0; i < ndim; i++)
        {
            data_l /= shape[i];
            new_stride.push_back(data_l);
        }
        new_tensor.stride = new_stride;

        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::unsqueeze(int new_dim)
    {
        // Check if the dimension is valid
        if (new_dim < 0 || new_dim < ndim)
        {
            throw std::invalid_argument("The dimension is out of range.");
        }

        if (new_dim == ndim)
        {
            return *this;
        }

        // Calculate the new shape
        std::vector<int> new_shape;
        int dim_diff = new_dim - ndim;
        for (int i = 0; i < new_dim; i++)
        {
            if (i < ndim)
            {
                new_shape.push_back(1);
            }
            else
            {
                new_shape.push_back(shape[i - dim_diff]);
            }
        }

        // Calculate the new stride
        std::vector<int> new_stride;
        for (int i = new_dim - 1; i >= 0; i--)
        {
            if (i - dim_diff >= 0)
            {
                new_stride.push_back(stride[i - dim_diff]);
            }
            else
            {
                new_stride.push_back(new_stride[i + 1] * new_shape[i + 1]);
            }
        }

        // Create a new tensor
        Tensor<T> new_tensor;
        new_tensor.data = data;
        new_tensor.offset = offset;
        new_tensor.ndim = new_shape.size();
        new_tensor.data_length = data_length;
        new_tensor.shape = new_shape;
        new_tensor.stride = new_stride;
        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::tile(std::vector<int> dims)
    {
        // Check if the dimensions are valid
        for (int i = 0; i < ndim; i++)
        {
            if (dims[i] < 1)
            {
                throw std::invalid_argument("The dimensions must be greater than 0.");
            }
        }
        Tensor<T> tensor = *this;
        // Check if the number of dimensions is equal
        if (dims.size() < ndim)
        {
            dims.insert(dims.begin(), ndim - dims.size(), 1);
        }
        else if (dims.size() > ndim)
        {
            tensor = this->unsqueeze(dims.size());
        }

        // Calculate the new shape
        std::vector<int> new_shape;
        for (int i = 0; i < ndim; i++)
        {
            new_shape.push_back(shape[i] * dims[i]);
        }

        Tensor<T> new_tensor = Tensor<T>(new_shape);

        // fill in new data
        std::vector<int> index = std::vector<int>(dims.size(), 0);
        int top = dims.size() - 1;
        while (true)
        {
            // copy at here
            std::vector<std::vector<int>> slicing_indices;

            for (int i = 0; i < dims.size(); i++)
            {
                slicing_indices.push_back({index[i] * shape[i], (index[i] + 1) * shape[i]});
            }
            new_tensor(slicing_indices) = tensor;

            index[top]++;
            while (index[top] == dims[top])
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
            while (top < dims.size() - 1)
            {
                top++;
                index[top] = 0;
            }
        }

        return new_tensor;
    }

    template <typename T>
    void Tensor<T>::view(const std::vector<int> &shape)
    {
        Tensor<T> reshaped_tensor = reshape(shape);
        // output
        std::cout << reshaped_tensor << std::endl;
    }

    /**
     * @brief 重载运算符
     */

    template <typename T>
    void Tensor<T>::checkShape(const Tensor<T> &t1, const Tensor<T> &t2)
    {
        if (t1.get_type() != t2.get_type())
        {
            throw std::runtime_error("Tensor type mismatch");
        }
        if (t1.ndim != t2.ndim)
        {
            throw std::runtime_error("Tensor dimension mismatch");
        }
        for (int i = 0; i < t1.ndim; i++)
        {
            if (t1.shape[i] != t2.shape[i])
            {
                throw std::runtime_error("Tensor shape mismatch");
            }
        }
    }

    template <typename T>
    Tensor<bool> Tensor<T>::operator==(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> temp1 = contiguous();
        Tensor<T> temp2 = t.contiguous();

        Tensor<bool> result = Tensor<bool>(temp1.get_shape());
        for (int i = 0; i < temp1.get_data_length(); i++)
        {
            result.get_data()[i] = (temp1.get_data()[i] == temp2.get_data()[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::operator!=(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> temp1 = contiguous();
        Tensor<T> temp2 = t.contiguous();

        Tensor<bool> result = Tensor<bool>(temp1.get_shape());
        for (int i = 0; i < temp1.get_data_length(); i++)
        {
            result.get_data()[i] = (temp1.get_data()[i] != temp2.get_data()[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::operator>(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> temp1 = contiguous();
        Tensor<T> temp2 = t.contiguous();

        Tensor<bool> result = Tensor<bool>(temp1.get_shape());
        for (int i = 0; i < temp1.get_data_length(); i++)
        {
            result.get_data()[i] = (temp1.get_data()[i] > temp2.get_data()[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::operator<(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> temp1 = contiguous();
        Tensor<T> temp2 = t.contiguous();

        Tensor<bool> result = Tensor<bool>(temp1.get_shape());
        for (int i = 0; i < temp1.get_data_length(); i++)
        {
            result.get_data()[i] = (temp1.get_data()[i] < temp2.get_data()[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::operator>=(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> temp1 = contiguous();
        Tensor<T> temp2 = t.contiguous();

        Tensor<bool> result = Tensor<bool>(temp1.get_shape());
        for (int i = 0; i < temp1.get_data_length(); i++)
        {
            result.get_data()[i] = (temp1.get_data()[i] >= temp2.get_data()[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::operator<=(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> temp1 = contiguous();
        Tensor<T> temp2 = t.contiguous();

        Tensor<bool> result = Tensor<bool>(temp1.get_shape());
        for (int i = 0; i < temp1.get_data_length(); i++)
        {
            result.get_data()[i] = (temp1.get_data()[i] <= temp2.get_data()[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::eq(const Tensor<T> &t)
    {
        return *this == t;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::ne(const Tensor<T> &t)
    {
        return *this != t;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::gt(const Tensor<T> &t)
    {
        return *this > t;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::lt(const Tensor<T> &t)
    {
        return *this < t;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::ge(const Tensor<T> &t)
    {
        return *this >= t;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::le(const Tensor<T> &t)
    {
        return *this <= t;
    }

    template <typename T>
    Tensor<T> Tensor<T>::sum(const int &dim) const
    {
        if (dim < 0 || dim >= ndim)
        {
            throw std::invalid_argument("Dimension out of range.");
        }
        // 计算结果张量的形状
        std::vector<int> result_shape(ndim - 1);
        for (int i = 0, j = 0; i < shape.size(); ++i)
        {
            if (i != dim)
            {
                result_shape[j++] = shape[i];
            }
        }
        Tensor result = Tensor<T>(result_shape);
        for (int i = 0; i < result.data_length; ++i)
        {
            for (int j = 0; j < stride[dim]; ++j)
            {
                T sum = 0;
                for (int k = 0; k < shape[dim]; ++k)
                {
                    if (dim == 0)
                    {
                        sum += data[k * stride[dim] + j + offset];
                    }
                    else
                    {
                        sum += data[i * stride[dim - 1] + k * stride[dim] + j + offset];
                    }
                }
                result.data[i * stride[dim] + j] = sum;
            }
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::mean(const int &dim) const
    {
        Tensor<T> result = sum(dim);
        for (int i = 0; i < result.data_length; ++i)
        {
            result.data[i] /= shape[dim];
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::max(const int &dim) const
    {
        if (dim < 0 || dim >= ndim)
        {
            throw std::invalid_argument("Dimension out of range.");
        }
        // 计算结果张量的形状
        std::vector<int> result_shape(ndim - 1);
        for (int i = 0, j = 0; i < shape.size(); ++i)
        {
            if (i != dim)
            {
                result_shape[j++] = shape[i];
            }
        }
        Tensor result = Tensor<T>(result_shape);
        for (int i = 0; i < result.data_length; ++i)
        {
            for (int j = 0; j < stride[dim]; ++j)
            {
                T max = std::numeric_limits<T>::min();
                for (int k = 0; k < shape[dim]; ++k)
                {
                    if (dim == 0)
                    {
                        if (data[k * stride[dim] + j + offset] > max)
                        {
                            max = data[k * stride[dim] + j + offset];
                        }
                    }
                    else
                    {
                        if (data[i * stride[dim - 1] + k * stride[dim] + j + offset] > max)
                        {
                            max = data[i * stride[dim - 1] + k * stride[dim] + j + offset];
                        }
                    }
                }
                result.data[i * stride[dim] + j] = max;
            }
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::min(const int &dim) const
    {
        if (dim < 0 || dim >= ndim)
        {
            throw std::invalid_argument("Dimension out of range.");
        }
        // 计算结果张量的形状
        std::vector<int> result_shape(ndim - 1);
        for (int i = 0, j = 0; i < shape.size(); ++i)
        {
            if (i != dim)
            {
                result_shape[j++] = shape[i];
            }
        }
        Tensor result = Tensor<T>(result_shape);
        for (int i = 0; i < result.data_length; ++i)
        {
            for (int j = 0; j < stride[dim]; ++j)
            {
                T min = std::numeric_limits<T>::max();
                for (int k = 0; k < shape[dim]; ++k)
                { // 对dim维度上的元素求和
                    if (dim == 0)
                    {
                        if (data[k * stride[dim] + j + offset] < min)
                        {
                            min = data[k * stride[dim] + j + offset];
                        }
                    }
                    else
                    {
                        if (data[i * stride[dim - 1] + k * stride[dim] + j + offset] < min)
                        {
                            min = data[i * stride[dim - 1] + k * stride[dim] + j + offset];
                        }
                    }
                }
                result.data[i * stride[dim] + j] = min;
            }
        }
        return result;
    }

    template <typename U>
    Tensor<U> sum(const Tensor<U> &t, const int &dim)
    {
        return t.sum(dim);
    }

    template <typename U>
    Tensor<U> mean(const Tensor<U> &t, const int &dim)
    {
        return t.mean(dim);
    }

    template <typename U>
    Tensor<U> max(const Tensor<U> &t, const int &dim)
    {
        return t.max(dim);
    }

    template <typename U>
    Tensor<U> min(const Tensor<U> &t, const int &dim)
    {
        return t.min(dim);
    }

    // ==================================== Cooper

    template <typename T>
    Tensor<T> re_construct(const Tensor<T> &t)
    {
        Tensor<T> result = Tensor<T>(t.shape);
        result = t;
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::add(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> t1 = this->contiguous();
        Tensor<T> t2 = t.contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] + t2.data[i];
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::add(T value)
    {
        Tensor<T> t1 = this->contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] + value;
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T> &t)
    {
        return this->add(t);
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator+(T value)
    {
        return this->add(value);
    }

    template <typename Y>
    Tensor<Y> add(const Tensor<Y> &t1, const Tensor<Y> &t2)
    {
        return t1.add(t2);
    }

    template <typename Y>
    Tensor<Y> add(const Tensor<Y> &t1, Y value)
    {
        return t1.add(value);
    }

    template <typename T>
    Tensor<T> Tensor<T>::sub(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> t1 = this->contiguous();
        Tensor<T> t2 = t.contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] + t2.data[i];
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::sub(T value)
    {
        Tensor<T> t1 = this->contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] - value;
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T> &t)
    {
        return this->sub(t);
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator-(T value)
    {
        return this->sub(value);
    }

    template <typename Y>
    Tensor<Y> sub(const Tensor<Y> &t1, const Tensor<Y> &t2)
    {
        return t1.sub(t2);
    }

    template <typename Y>
    Tensor<Y> sub(const Tensor<Y> &t1, Y value)
    {
        return t1.sub(value);
    }

    template <typename T>
    Tensor<T> Tensor<T>::mul(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> t1 = this->contiguous();
        Tensor<T> t2 = t.contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] * t2.data[i];
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::mul(T value)
    {
        Tensor<T> t1 = this->contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] * value;
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T> &t)
    {
        return this->mul(t);
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator*(T value)
    {
        return this->mul(value);
    }

    template <typename Y>
    Tensor<Y> mul(const Tensor<Y> &t1, const Tensor<Y> &t2)
    {
        return t1.mul(t2);
    }

    template <typename Y>
    Tensor<Y> mul(const Tensor<Y> &t1, Y value)
    {
        return t1.mul(value);
    }

    template <typename T>
    Tensor<T> Tensor<T>::div(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> t1 = this->contiguous();
        Tensor<T> t2 = t.contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] / t2.data[i];
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::div(T value)
    {
        Tensor<T> t1 = this->contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] / value;
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T> &t)
    {
        return this->div(t);
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator/(T value)
    {
        return this->div(value);
    }

    template <typename Y>
    Tensor<Y> div(const Tensor<Y> &t1, const Tensor<Y> &t2)
    {
        return t1.div(t2);
    }

    template <typename Y>
    Tensor<Y> div(const Tensor<Y> &t1, Y value)
    {
        return t1.div(value);
    }

    template <typename T>
    Tensor<T> Tensor<T>::log(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> t1 = this->contiguous();
        Tensor<T> t2 = t.contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = std::log(t1.data[i]) / std::log(t2.data[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::log(T value)
    {
        Tensor<T> t1 = this->contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = std::log(t1.data[i]) / std::log(value);
        }
        return result;
    }

    template <typename Y>
    Tensor<Y> log(const Tensor<Y> &t1, const Tensor<Y> &t2)
    {
        return t1.log(t2);
    }

    template <typename Y>
    Tensor<Y> log(const Tensor<Y> &t1, Y value)
    {
        return t1.log(value);
    }

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

#endif