
#ifndef MATH_H
#define MATH_H

#include "tensor.h"
#include "omp.h"


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
    Tensor<bool> Tensor<T>::operator==(const Tensor<T> &t) const
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
    Tensor<bool> Tensor<T>::operator!=(const Tensor<T> &t) const
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
    Tensor<bool> Tensor<T>::operator>(const Tensor<T> &t) const
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
    Tensor<bool> Tensor<T>::operator<(const Tensor<T> &t) const
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
    Tensor<bool> Tensor<T>::operator>=(const Tensor<T> &t) const
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
    Tensor<bool> Tensor<T>::operator<=(const Tensor<T> &t) const
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
    Tensor<bool> Tensor<T>::eq(const Tensor<T> &t) const
    {
        return *this == t;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::ne(const Tensor<T> &t) const
    {
        return *this != t;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::gt(const Tensor<T> &t) const
    {
        return *this > t;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::lt(const Tensor<T> &t) const
    {
        return *this < t;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::ge(const Tensor<T> &t) const
    {
        return *this >= t;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::le(const Tensor<T> &t) const
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
        std::vector<int> result_shape(ndim);
        for (int j = 0; j < shape.size(); ++j)
        {
            if (j == dim)
            {
                result_shape[j] = 1; // 要reduce的维度
            }
            else
            {
                result_shape[j] = shape[j];
            }
        }
        Tensor result = Tensor<T>(result_shape);
        std::vector<int> index = std::vector<int>(this->ndim, 0);
        int top = this->ndim - 1;
        T *data_this = this->data.get() + this->offset;
        T *data_result = result.data.get() + result.offset;
        T *current_this = data_this;
        T *current_result = data_result;
        while (index[0] < this->shape[0])
        {
            std::vector<int> result_index = index;
            result_index[dim] = 0;
            result.enisum_indexing(result_index) = this->enisum_indexing(index) + result.enisum_indexing(result_index);
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
        return result.squeeze(dim);
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
        std::vector<int> result_shape(ndim);
        for (int j = 0; j < shape.size(); ++j)
        {
            if (j == dim)
            {
                result_shape[j] = 1; // 要reduce的维度
            }
            else
            {
                result_shape[j] = shape[j];
            }
        }
        Tensor result = Tensor<T>(result_shape);
        for (int i = 0; i < result.data_length; ++i)
        {
            result.data[i] = -std::numeric_limits<T>::infinity();
        }
        std::vector<int> index = std::vector<int>(this->ndim, 0);
        int top = this->ndim - 1;
        T *data_this = this->data.get() + this->offset;
        T *data_result = result.data.get() + result.offset;
        T *current_this = data_this;
        T *current_result = data_result;

        while (index[0] < this->shape[0])
        {
            std::vector<int> result_index = index;
            result_index[dim] = 0;
            result.enisum_indexing(result_index) = std::max(this->enisum_indexing(index), result.enisum_indexing(result_index));
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
        return result.squeeze(dim);
    }


     template <typename T>
    Tensor<T> Tensor<T>::min(const int &dim) const
    {
        if (dim < 0 || dim >= ndim)
        {
            throw std::invalid_argument("Dimension out of range.");
        }
        // 计算结果张量的形状
        std::vector<int> result_shape(ndim);
        for (int j = 0; j < shape.size(); ++j)
        {
            if (j == dim)
            {
                result_shape[j] = 1; // 要reduce的维度
            }
            else
            {
                result_shape[j] = shape[j];
            }
        }
        Tensor result = Tensor<T>(result_shape);
        for (int i = 0; i < result.data_length; ++i)
        {
            result.data[i] = std::numeric_limits<T>::infinity();
        }
        std::vector<int> index = std::vector<int>(this->ndim, 0);
        int top = this->ndim - 1;
        T *data_this = this->data.get() + this->offset;
        T *data_result = result.data.get() + result.offset;
        T *current_this = data_this;
        T *current_result = data_result;

        while (index[0] < this->shape[0])
        {

            std::vector<int> result_index = index;
            result_index[dim] = 0;
            
            result.enisum_indexing(result_index) = std::min(this->enisum_indexing(index), result.enisum_indexing(result_index));

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
        return result.squeeze(dim);
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
    Tensor<T> Tensor<T>::add(const Tensor<T> &t) const
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
    Tensor<T> Tensor<T>::add(T value) const
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
    Tensor<T> Tensor<T>::operator+(const Tensor<T> &t) const
    {
        return this->add(t);
    }
    template <typename T>
    Tensor<T> Tensor<T>::operator+(T value) const
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
    Tensor<T> Tensor<T>::sub(const Tensor<T> &t) const
    {
        checkShape(*this, t);
        Tensor<T> t1 = this->contiguous();
        Tensor<T> t2 = t.contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] - t2.data[i];
        }
        return result;
    }
    template <typename T>
    Tensor<T> Tensor<T>::sub(T value) const
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
    Tensor<T> Tensor<T>::operator-(const Tensor<T> &t) const
    {
        return this->sub(t);
    }
    template <typename T>
    Tensor<T> Tensor<T>::operator-(T value) const
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
    Tensor<T> Tensor<T>::mul(const Tensor<T> &t) const
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
    Tensor<T> Tensor<T>::mul(T value) const
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
    Tensor<T> Tensor<T>::operator*(const Tensor<T> &t) const
    {
        return this->mul(t);
    }
    template <typename T>
    Tensor<T> Tensor<T>::operator*(T value) const
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
    Tensor<T> Tensor<T>::div(const Tensor<T> &t) const
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
    Tensor<T> Tensor<T>::div(T value) const
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
    Tensor<T> Tensor<T>::operator/(const Tensor<T> &t) const
    {
        return this->div(t);
    }
    template <typename T>
    Tensor<T> Tensor<T>::operator/(T value) const
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
    Tensor<T> Tensor<T>::log(const Tensor<T> &t) const
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
    Tensor<T> Tensor<T>::log(T value) const
    {
        Tensor<T> t1 = this->contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = std::log(t1.data[i]) / std::log(value);
        }
        return result;
    }
    template <typename T>
    Tensor<T> Tensor<T>::log() const
    {
        Tensor<T> t1 = this->contiguous();
        Tensor<T> result = Tensor(this->shape);
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = std::log(t1.data[i]);
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

    // OpenMP Optimized Version

    template <typename T>
    Tensor<T> Tensor<T>::omp_add(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> t1 = this->contiguous();
        Tensor<T> t2 = t.contiguous();
        Tensor<T> result = Tensor(this->shape);

        # pragma omp parallel for
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] + t2.data[i];
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_add(T value)
    {
        Tensor<T> t1 = this->contiguous();
        Tensor<T> result = Tensor(this->shape);

        # pragma omp parallel for
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] + value;
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_sub(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> t1 = this->contiguous();
        Tensor<T> t2 = t.contiguous();
        Tensor<T> result = Tensor(this->shape);

        # pragma omp parallel for
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] + t2.data[i];
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_sub(T value)
    {
        Tensor<T> t1 = this->contiguous();
        Tensor<T> result = Tensor(this->shape);

        # pragma omp parallel for
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] - value;
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_mul(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> t1 = this->contiguous();
        Tensor<T> t2 = t.contiguous();
        Tensor<T> result = Tensor(this->shape);

        # pragma omp parallel for
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] * t2.data[i];
        }
        return result;
    }
    template <typename T>
    Tensor<T> Tensor<T>::omp_mul(T value)
    {
        Tensor<T> t1 = this->contiguous();
        Tensor<T> result = Tensor(this->shape);

        # pragma omp parallel for
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] * value;
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_div(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> t1 = this->contiguous();
        Tensor<T> t2 = t.contiguous();
        Tensor<T> result = Tensor(this->shape);

        # pragma omp parallel for
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] / t2.data[i];
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_div(T value)
    {
        Tensor<T> t1 = this->contiguous();
        Tensor<T> result = Tensor(this->shape);

        # pragma omp parallel for
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = t1.data[i] / value;
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_log(const Tensor<T> &t)
    {
        checkShape(*this, t);
        Tensor<T> t1 = this->contiguous();
        Tensor<T> t2 = t.contiguous();
        Tensor<T> result = Tensor(this->shape);

        # pragma omp parallel for
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = std::log(t1.data[i]) / std::log(t2.data[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_log(T value)
    {
        Tensor<T> t1 = this->contiguous();
        Tensor<T> result = Tensor(this->shape);

        # pragma omp parallel for
        for (int i = 0; i < result.data_length; i++)
        {
            result.data[i] = std::log(t1.data[i]) / std::log(value);
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_eq(const Tensor<T> &t){
        checkShape(*this, t);
        Tensor<T> temp1 = contiguous();
        Tensor<T> temp2 = t.contiguous();

        Tensor<bool> result = Tensor<bool>(temp1.get_shape());

        # pragma omp parallel for
        for (int i = 0; i < temp1.get_data_length(); i++)
        {
            result.get_data()[i] = (temp1.get_data()[i] == temp2.get_data()[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_ne(const Tensor<T> &t){
        checkShape(*this, t);
        Tensor<T> temp1 = contiguous();
        Tensor<T> temp2 = t.contiguous();

        Tensor<bool> result = Tensor<bool>(temp1.get_shape());

        # pragma omp parallel for
        for (int i = 0; i < temp1.get_data_length(); i++)
        {
            result.get_data()[i] = (temp1.get_data()[i] != temp2.get_data()[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_gt(const Tensor<T> &t){
        checkShape(*this, t);
        Tensor<T> temp1 = contiguous();
        Tensor<T> temp2 = t.contiguous();

        Tensor<bool> result = Tensor<bool>(temp1.get_shape());

        # pragma omp parallel for
        for (int i = 0; i < temp1.get_data_length(); i++)
        {
            result.get_data()[i] = (temp1.get_data()[i] > temp2.get_data()[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_ge(const Tensor<T> &t){
        checkShape(*this, t);
        Tensor<T> temp1 = contiguous();
        Tensor<T> temp2 = t.contiguous();

        Tensor<bool> result = Tensor<bool>(temp1.get_shape());

        # pragma omp parallel for
        for (int i = 0; i < temp1.get_data_length(); i++)
        {
            result.get_data()[i] = (temp1.get_data()[i] >= temp2.get_data()[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_lt(const Tensor<T> &t){
        checkShape(*this, t);
        Tensor<T> temp1 = contiguous();
        Tensor<T> temp2 = t.contiguous();

        Tensor<bool> result = Tensor<bool>(temp1.get_shape());

        # pragma omp parallel for
        for (int i = 0; i < temp1.get_data_length(); i++)
        {
            result.get_data()[i] = (temp1.get_data()[i] < temp2.get_data()[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::omp_le(const Tensor<T> &t){
        checkShape(*this, t);
        Tensor<T> temp1 = contiguous();
        Tensor<T> temp2 = t.contiguous();

        Tensor<bool> result = Tensor<bool>(temp1.get_shape());

        # pragma omp parallel for
        for (int i = 0; i < temp1.get_data_length(); i++)
        {
            result.get_data()[i] = (temp1.get_data()[i] <= temp2.get_data()[i]);
        }
        return result;
    }
}

#endif