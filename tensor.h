#include <vector>

#ifndef TENSOR_H_
#define TENSOR_H_

namespace ts
{
    // Tensor class
    template <typename T>
    class Tensor
    {
    private:
        std::vector<int> size;
        T *data_ptr;
        std::vector<int> sequence;

    public:
        // Constructor
        Tensor(const std::vector<int> &size) : size(size)
        {
            data_ptr = new T[data_length()];
            int shape_size=shape.size();
            for (int i = 0; i < shape_size; i++)
            {
                sequence.push_back(i);
            }
        }

        // Destructor
        ~Tensor()
        {
            delete[] data_ptr;
        }

        // Get the size of the tensor
        const std::vector<int> & size() const
        {
            return size;
        }

        // Get the type of the tensor
        std::string type() const
        {
            return typeid(T).name();
        }

        // Get the pointer to the data
        T * data_ptr()
        {
            return data_ptr;
        }

        const std::vector<int> & sequence() const
        {
            return sequence;
        }

        int data_length()
        {
            int length = 1;
            for (axis:size)
            {
                length*=axis;
            }
            return length;
        }

        
    };

    
} // namespace ts


#endif