
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
    Tensor<T> add_cu_f(const Tensor<T> &in1, const Tensor<T> &in2){
        Tensor<T> t1 = in1.contiguous();
        Tensor<T> t2 = in2.contiguous();
        Tensor<T> result = Tensor<T>(in1.shape);


        T *dat, *data1, *data2;
        cudaMalloc((void**)&data1, t1.data_length*sizeof(T)); 
        cudaMalloc((void**)&data2, t1.data_length*sizeof(T)); 
        cudaMalloc((void**)&dat, t1.data_length*sizeof(T)); 
        
        cudaMemcpy(data1, t1.data.get(), t1.data_length*sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(data2, t2.data.get(), t1.data_length*sizeof(T), cudaMemcpyHostToDevice);
        
        int threadsPerBlock = 1024;
        int numBlocks = (t1.data_length + threadsPerBlock-1) / threadsPerBlock;

        add_cu_kernal<<<numBlocks,threadsPerBlock>>>(dat, data1, data2, t1.data_length);

        cudaMemcpy(result.data.get(), dat, t1.data_length*sizeof(T),cudaMemcpyDeviceToHost);
        cudaFree(dat);
        cudaFree(data1);
        cudaFree(data2);

        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::add_cu(const Tensor<T>& t){
        return add_cu_f(*this,t);
    }
    template ts::Tensor<int> ts::Tensor<int>::add_cu(ts::Tensor<int> const&);
    template ts::Tensor<float> ts::Tensor<float>::add_cu(ts::Tensor<float> const&);
    template ts::Tensor<double> ts::Tensor<double>::add_cu(ts::Tensor<double> const&);
    
    template<typename T>
    void dataToDevice(T* dev, const T* hos, int length){
        assert(cudaSuccess==cudaMalloc((void**)&dev,length*sizeof(T)));
        assert(cudaSuccess==cudaMemcpy(dev,hos,length*sizeof(T),cudaMemcpyHostToDevice));
    }

    template<typename T>
    void dataToHost(T* dev, const T* hos, int length){
        assert(cudaSuccess==cudaMemcpy(hos,dev,length*sizeof(T),cudaMemcpyDeviceToHost));
        assert(cudaSuccess==cudaFree(dev));
        dev=nullptr;
    }

    template<typename T>
    void add_perf(T* result, const T* data1, const T* data2, int length) {
        int threadsPerBlock = 1024;
        int numBlocks = (length + threadsPerBlock-1) / threadsPerBlock;
        add_cu_kernal<<<numBlocks,threadsPerBlock>>>(result, data1, data2, length);
        cudaDeviceSynchronize();
    }

    
}