#include <vector>
#include <typeinfo>  // typeid
#include <string>    // std::string
#include <memory>
#include <stdexcept>


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

        Tensor(const T* data, const std::vector<int> &_shape);

        // Destructor
        ~Tensor();
        
        std::string get_type() const;

        std::shared_ptr<T[]> get_data();

        int get_offset() const;

        int get_ndim() const;

        int get_data_length() const;

        const std::vector<int> &get_shape() const;

        const std::vector<int> &get_stride() const;

        Tensor<T> slicing(const std::vector<std::vector<int>> &indices);

        // use -1 to indicate all elements
        Tensor<T> indexing(const std::vector<int> &indices);

        // alias for slicing
        Tensor<T> operator()(const std::vector<std::vector<int>> &indices);

        // alias for indexing
        Tensor<T> operator()(const std::vector<int> &indices);

        Tensor<T> permute(const std::vector<int> &axes);

        Tensor<T> transpose(const int& dim1, const int & dim2);

        Tensor<T> cat(const Tensor<T> &other, const int & dim);
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
            data_l/=shape[i];
            stride.push_back(data_l);
        }

        data = std::shared_ptr<T[]>(new T[data_length]);
    }

    // Constructor with existing data and shape
    template <typename T>
    Tensor<T>::Tensor(const T* _data, const std::vector<int> &_shape) : offset(0), ndim(_shape.size()), shape(_shape), stride(std::vector<int>())
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
            data_l/=shape[i];
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
    const std::vector<int>& Tensor<T>::get_shape() const
    {
        return shape;
    }

    // Get the stride
    template <typename T>
    const std::vector<int>& Tensor<T>::get_stride() const
    {
        return stride;
    }


    template <typename T>
    Tensor<T> Tensor<T>::slicing(const std::vector<std::vector<int>> & indices){
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
            if (indices[i][0] < 0 || indices[i][0] >= shape[i] || indices[i][1] < 0 || indices[i][1] > shape[i])
            {
                throw std::invalid_argument("The indices are out of range.");
            }
        }

        // Calculate the new shape
        std::vector<int> new_shape;
        for (int i = 0; i < ndim; i++)
        {
            new_shape.push_back(indices[i][1] - indices[i][0]);
        }

        // Calculate the new data length
        int new_data_length = 1;
        for (int axis : new_shape)
        {
            new_data_length *= axis;
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
        new_tensor.data_length = new_data_length;
        new_tensor.shape = new_shape;
        new_tensor.stride = stride;

        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator()(const std::vector<std::vector<int>> & indices){
        return slicing(indices);
    }

    template <typename T>
    Tensor<T> Tensor<T>::indexing(const std::vector<int> & indices){
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

        // Calculate the new data length
        int new_data_length = 1;
        for (int axis : new_shape)
        {
            new_data_length *= axis;
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
        new_tensor.data_length = new_data_length;
        new_tensor.shape = new_shape;
        new_tensor.stride = stride;

        return new_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator()(const std::vector<int> & indices){
        return indexing(indices);
    }


    template <typename T>
    Tensor<T> Tensor<T>::permute(const std::vector<int> & axes){
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
    Tensor<T> Tensor<T>::transpose(const int& dim1, const int & dim2){
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
    Tensor<T> Tensor<T>::cat(const Tensor<T> &other, const int & dim){
        
    }

} // namespace ts

#endif