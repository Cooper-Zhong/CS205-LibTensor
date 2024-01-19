#pragma once
#include "tensor.h"
namespace ts
{
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
}