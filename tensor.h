#include <vector>
#include <typeinfo>  // typeid
#include <string>    // std::string
#include <memory>
#include <stdexcept>
#include <random>


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

        static Tensor<T> rand_tensor(const std::vector<int> &_shape);

        static Tensor<T> zeros_tensor(const std::vector<int> &_shape);

        static Tensor<T> ones_tensor(const std::vector<int> &_shape);

        static Tensor<T> full_tensor(const std::vector<int> &_shape, T t);

        static Tensor<T> eye_tensor(const std::vector<int> &_shape);

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
        template <typename T1>
        friend std::ostream & operator<<(std::ostream & o, Tensor<T1> &t);
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
        for (int i = 0; i < data_length; i++)
        {
            data[i] = 0;
        }

        
        
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
        if (_shape.size() < 2 || !std::equal(_shape.begin(), _shape.end()-1, _shape.begin() + 1, std::equal_to<int>()))
        {
            throw std::invalid_argument("eye_tensor is only supported for square tensors.");
        }

        // Initialize as an identity matrix
        T * data=eye_tensor.get_data().get();
        T * current = data;
        auto stride = eye_tensor.get_stride();

        for (int i = 0; i < _shape[0]; i++)
        {
            current = data;
            for (int j = 0; j < _shape.size(); j++)
            {
                current += i*stride[j];
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
    void recurse_print(const std::vector<int>& shape,const std::vector<int>& stride, int layer,const T* data, int ndim){
        std::ostream& o = std::cout;
        for (int i = 0; i < layer; i++)
        {
            o << " ";
        }
        
        o << "[";
        if (layer+1 == ndim)
        {
            for (int i = 0; i < shape[layer]; i++)
            {
                o << data[i*stride[layer]];
                if (i+1<shape[layer])
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
        else{
            o << std::endl;
            for (int i = 0; i < shape[layer]; i++)
            {
                recurse_print(shape, stride, layer+1, data+stride[layer]*i, ndim);
            }
            for (int i = 0; i < layer; i++)
            {
                o << " ";
            }
            
            o << "]";
            if (layer > 0 )
            {
                o << ",";
            }
            o << std::endl;
            
        }        
    }

    template <typename T>
    std::ostream & operator<<(std::ostream & o, Tensor<T> & t)
    {
        o << "Data pointer: " << t.get_data() << std::endl;
        o << "Data length: " << t.get_data_length() << std::endl;
        o << "Type: " << t.get_type() << std::endl;
        o << "Offset: " << t.get_offset() << std::endl;
        o << "Ndim:" << t.get_ndim() << std::endl;
        auto t_shape = t.get_shape();
        o << "Shape: [";
        for (int i = 0; i < t_shape.size(); i++)
        {
            o << t_shape[i];
            if (i+1 < t_shape.size())
            {
                o << ", ";
            }
        }
        o << "]" << std::endl;


        auto t_stride = t.get_stride();
        o << "Stride: [";
        for (int i = 0; i < t_stride.size(); i++)
        {
            o << t_stride[i];
            if (i+1 < t_stride.size())
            {
                o << ", ";
            }
        }
        o << "]" << std::endl;
        
        T* t_data = t.get_data().get()+t.get_offset();

        recurse_print(t_shape, t_stride, 0, t_data, t.get_ndim());
        
        return o;
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