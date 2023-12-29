#include <vector>
#include <typeinfo>  // typeid
#include <string>    // std::string
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
            o << "]" << std::endl;
        }
        else{
            o << std::endl;
            for (int i = 0; i < shape[layer]; i++)
            {
                recurse_print(shape, stride, layer+1, data+stride[layer]*i, ndim);
            }
            o << "]";
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





} // namespace ts

#endif