#ifndef TRANSFORM_H_
#define TRANSFORM_H_

#include "tensor.h"

namespace ts
{
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
    Tensor<T> Tensor<T>::permute(const std::vector<int> &axes) const
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
    Tensor<T> Tensor<T>::transpose(const int &dim1, const int &dim2) const
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
    Tensor<T> Tensor<T>::cat(const Tensor<T> &other, const int &dim) const
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
        new_tensor.deepcopy_from(*this);

        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::reshape(const std::vector<int> &shape) const
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
            return *this;
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
        for (int i = 0; i < new_tensor.ndim; i++)
        {
            data_l /= new_shape[i];
            new_stride.push_back(data_l);
        }
        new_tensor.stride = new_stride;

        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::squeeze(int dim){
        // Check if the dimension is valid
        if (dim < 0 || dim >= ndim)
        {
            throw std::invalid_argument("The dimension is out of range.");
        }

        // Check if the tensor is a vector
        if (ndim == 1)
        {
            return *this;
        }

        // Check if the dimension is valid
        if (shape[dim] != 1)
        {
            throw std::invalid_argument("The dimension is not 1.");
        }

        // Calculate the new shape
        std::vector<int> new_shape;
        for (int i = 0; i < ndim; i++)
        {
            if (i != dim)
            {
                new_shape.push_back(shape[i]);
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
        for (int i = 0; i < new_tensor.ndim; i++)
        {
            data_l /= new_shape[i];
            new_stride.push_back(data_l);
        }
        new_tensor.stride = new_stride;

        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::unsqueeze(int new_dim) const
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
    Tensor<T> Tensor<T>::tile(std::vector<int> dims) const
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
            new_tensor(slicing_indices).deepcopy_from(tensor);

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
    Tensor<T> Tensor<T>::view(const std::vector<int> &shape) const
    {
        Tensor<T> reshaped_tensor = reshape(shape);
        return reshaped_tensor;
    }

    template <typename T>
    T& Tensor<T>::at(const std::vector<int> &indices) const{
        // Check if the tensor is a vector
        if(ndim == 1 && data_length == 1 && (indices.size() == 0 || indices.size() == 1)){
            return data[offset];
        }

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

        // Calculate the offset
        int offset = 0;
        for (int i = 0; i < ndim; i++)
        {
            if (indices[i] != -1)
            {
                offset += indices[i] * stride[i];
            }
        }

        return data[offset];
    }

    template <typename T>
    T& Tensor<T>::operator[](const std::vector<int> &indices) const{
        return at(indices);
    }

} // namespace ts
#endif