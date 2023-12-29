#include <vector>
#include <typeinfo> // typeid
#include <string>   // std::string
#include <memory>

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

        // Destructor
        ~Tensor();

        std::string get_type() const;

        std::shared_ptr<T[]> get_data();

        int get_offset() const;

        int get_ndim() const;

        int get_data_length() const;

        const std::vector<int> &get_shape() const;

        const std::vector<int> &get_stride() const;

        /**
         * @brief 重载运算符
         */
        Tensor<bool> operator==(const Tensor<T> &t); // 等于
        Tensor<bool> operator!=(const Tensor<T> &t); // 不等于
        Tensor<bool> operator>(const Tensor<T> &t);  // 大于
        Tensor<bool> operator<(const Tensor<T> &t);  // 小于
        Tensor<bool> operator>=(const Tensor<T> &t); // 大于等于
        Tensor<bool> operator<=(const Tensor<T> &t); // 小于等于
        template <typename U>
        friend Tensor<bool> eq(const Tensor<T> &t1, const Tensor<T> &t2); // 等于
        template <typename U>
        friend Tensor<bool> ne(const Tensor<T> &t1, const Tensor<T> &t2); // 不等于
        template <typename U>
        friend Tensor<bool> gt(const Tensor<T> &t1, const Tensor<T> &t2); // 大于
        template <typename U>
        friend Tensor<bool> lt(const Tensor<T> &t1, const Tensor<T> &t2); // 小于
        template <typename U>
        friend Tensor<bool> ge(const Tensor<T> &t1, const Tensor<T> &t2); // 大于等于
        template <typename U>
        friend Tensor<bool> le(const Tensor<T> &t1, const Tensor<T> &t2); // 小于等于

        static void checkShape(Tensor<T> &t1, Tensor<T> &t2); // 检查两个张量的dataType, dim, shape是否相同
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
    Tensor<T>::Tensor(const std::vector<int> &_shape) : offset(0), ndim(_shape.size()), shape(_shape), stride(std::vector<int>())
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
    }

    // Constructor with existing data and shape
    template <typename T>
    Tensor<T>::Tensor(const T *_data, const std::vector<int> &_shape) : offset(0), ndim(_shape.size()), shape(_shape), stride(std::vector<int>())
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
    std::shared_ptr<T[]> Tensor<T>::get_data()
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

    // Cooper ====================================

    /**
     * @brief 重载运算符
     */

    template <typename T>
    void checkShape(Tensor<T> &t1, Tensor<T> &t2)
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
        Tensor<bool> result;
        result.ndim = this->ndim;
        result.shape = this->shape;
        result.stride = this->stride;
        result.data_length = this->data_length;
        result.data = std::shared_ptr<bool[]>(new bool[this->data_length]);

        for (int i = 0; i < this->data_length; i++)
        {
            result.data[i] = (this->data[i + this->offset] == t.data[i + t.offset]);
        }
        return result;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::operator!=(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<bool> result;
        result.ndim = this->ndim;
        result.shape = this->shape;
        result.stride = this->stride;
        result.data_length = this->data_length;
        result.data = std::shared_ptr<bool[]>(new bool[this->data_length]);

        for (int i = 0; i < this->data_length; i++)
        {
            result.data[i] = (this->data[i + this->offset] != t.data[i + t.offset]);
        }
        return result;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::operator>(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<bool> result;
        result.ndim = this->ndim;
        result.shape = this->shape;
        result.stride = this->stride;
        result.data_length = this->data_length;
        result.data = std::shared_ptr<bool[]>(new bool[this->data_length]);

        for (int i = 0; i < this->data_length; i++)
        {
            result.data[i] = (this->data[i + this->offset] > t.data[i + t.offset]);
        }
        return result;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::operator<(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<bool> result;
        result.ndim = this->ndim;
        result.shape = this->shape;
        result.stride = this->stride;
        result.data_length = this->data_length;
        result.data = std::shared_ptr<bool[]>(new bool[this->data_length]);

        for (int i = 0; i < this->data_length; i++)
        {
            result.data[i] = (this->data[i + this->offset] < t.data[i + t.offset]);
        }
        return result;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::operator>=(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<bool> result;
        result.ndim = this->ndim;
        result.shape = this->shape;
        result.stride = this->stride;
        result.data_length = this->data_length;
        result.data = std::shared_ptr<bool[]>(new bool[this->data_length]);

        for (int i = 0; i < this->data_length; i++)
        {
            result.data[i] = (this->data[i + this->offset] >= t.data[i + t.offset]);
        }
        return result;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::operator<=(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<bool> result;
        result.ndim = this->ndim;
        result.shape = this->shape;
        result.stride = this->stride;
        result.data_length = this->data_length;
        result.data = std::shared_ptr<bool[]>(new bool[this->data_length]);

        for (int i = 0; i < this->data_length; i++)
        {
            result.data[i] = (this->data[i + this->offset] <= t.data[i + t.offset]);
        }
        return result;
    }

    template <typename T>
    Tensor<bool> eq(const Tensor<T> &t1, const Tensor<T> &t2)
    {
        return t1 == t2;
    }

    template <typename T>
    Tensor<bool> ne(const Tensor<T> &t1, const Tensor<T> &t2)
    {
        return t1 != t2;
    }

    template <typename T>
    Tensor<bool> gt(const Tensor<T> &t1, const Tensor<T> &t2)
    {
        return t1 > t2;
    }

    template <typename T>
    Tensor<bool> lt(const Tensor<T> &t1, const Tensor<T> &t2)
    {
        return t1 < t2;
    }

    template <typename T>
    Tensor<bool> ge(const Tensor<T> &t1, const Tensor<T> &t2)
    {
        return t1 >= t2;
    }

    template <typename T>
    Tensor<bool> le(const Tensor<T> &t1, const Tensor<T> &t2)
    {
        return t1 <= t2;
    }

    // ==================================== Cooper

} // namespace ts

#endif