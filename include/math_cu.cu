#include "tensor.h"

namespace ts
{
    template<typename T>
    __global__ void add_cu_kernal(T* result, const T* data1, const T* data2, int length){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length)
        {
            result[idx] = data1[idx] + data2[idx];
        }
    }

    template<typename T>
    Tensor<T> add_cu(const Tensor<T> &in1, const Tensor<T> &in2){
        Tensor<T> t1 = in1.contiguous();
        Tensor<T> t2 = in2.contiguous();
        Tensor<T> result = Tensor(in1.shape);


        T *dat, *data1, *data2;
        cudaMalloc((void**)&data1, t1.data_length*sizeof(T)); 
        cudaMalloc((void**)&data2, t1.data_length*sizeof(T)); 
        cudaMalloc((void**)&dat, t1.data_length*sizeof(T)); 
        
        cudaMemcpy(data1, t1.data.get(), t1.data_length*sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(data2, t2.data.get(), t1.data_length*sizeof(T), cudaMemcpyHostToDevice);
        
        int threadsPerBlock = 512;
        int numBlocks = (t1.data_length + threadsPerBlock-1) / threadsPerBlock;

        add_cu_kernal<<<numBlocks,threadsPerBlock>>>(dat, data1, data2, t1.data_length);

        cudaMemcpy(result.data.get(), dat, t1.data_length*sizeof(T),cudaMemcpyDeviceToHost);

        return result;
    }





}