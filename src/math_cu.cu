
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
    __global__ void sub_cu_kernal(T* result, const T* data1, const T* data2, int length){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length)
        {
            result[idx] = data1[idx] - data2[idx];
        }
    }

    template<typename T>
    __global__ void mul_cu_kernal(T* result, const T* data1, const T* data2, int length){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length)
        {
            result[idx] = data1[idx] * data2[idx];
        }
    }

    template<typename T>
    __global__ void div_cu_kernal(T* result, const T* data1, const T* data2, int length){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length)
        {
            result[idx] = data1[idx] / data2[idx];
        }
    }

    template<typename T>
    __global__ void ein_cu_kernal(T* result, const T* data1, const T* data2, int t1_height, int t1_width, int t2_width) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= t1_height*t2_width)
        {
            return;
        }
        size_t row = idx/t2_width;
        size_t col = idx%t2_width;

        T ans = 0;
        for (size_t i = 0; i < t1_width; i++)
        {
            ans+=data1[row*t1_width + i] * data2[i*t2_width+col];
        }
        result[row*t2_width + col] = ans;
    }

    template<typename T>
    Tensor<T> Tensor<T>::cu_add(Tensor<T>& t){
        Tensor<T> result = Tensor<T>(shape);

        if(gpu_t == nullptr){
            gpu();
        }
        if(t.gpu_t == nullptr){
            t.gpu();
        }

        int threadsPerBlock = 1024;
        int numBlocks = (data_length + threadsPerBlock-1) / threadsPerBlock;

        result.gpu();

        add_cu_kernal<<<numBlocks,threadsPerBlock>>>(result.gpu_t ,gpu_t,t.gpu_t,data_length);

        result.cpu();
        result.gpu_free();

        return result;
    }
    template ts::Tensor<double> ts::Tensor<double>::cu_add(ts::Tensor<double>&);

    template<typename T>
    Tensor<T> Tensor<T>::cu_sub(Tensor<T>& t){
        Tensor<T> result = Tensor<T>(shape);

        if(gpu_t == nullptr){
            gpu();
        }
        if(t.gpu_t == nullptr){
            t.gpu();
        }

        int threadsPerBlock = 1024;
        int numBlocks = (data_length + threadsPerBlock-1) / threadsPerBlock;

        result.gpu();

        sub_cu_kernal<<<numBlocks,threadsPerBlock>>>(result.gpu_t ,gpu_t,t.gpu_t,data_length);

        result.cpu();
        result.gpu_free();

        return result;
    }
    template ts::Tensor<double> ts::Tensor<double>::cu_sub(ts::Tensor<double>&);

    template<typename T>
    Tensor<T> Tensor<T>::cu_mul(Tensor<T>& t){
        Tensor<T> result = Tensor<T>(shape);

        if(gpu_t == nullptr){
            gpu();
        }
        if(t.gpu_t == nullptr){
            t.gpu();
        }

        int threadsPerBlock = 1024;
        int numBlocks = (data_length + threadsPerBlock-1) / threadsPerBlock;

        result.gpu();

        mul_cu_kernal<<<numBlocks,threadsPerBlock>>>(result.gpu_t ,gpu_t,t.gpu_t,data_length);

        result.cpu();
        result.gpu_free();

        return result;
    }
    template ts::Tensor<double> ts::Tensor<double>::cu_mul(ts::Tensor<double>&);

    template<typename T>
    Tensor<T> Tensor<T>::cu_div(Tensor<T>& t){
        Tensor<T> result = Tensor<T>(shape);

        if(gpu_t == nullptr){
            gpu();
        }
        if(t.gpu_t == nullptr){
            t.gpu();
        }

        int threadsPerBlock = 1024;
        int numBlocks = (data_length + threadsPerBlock-1) / threadsPerBlock;

        result.gpu();

        div_cu_kernal<<<numBlocks,threadsPerBlock>>>(result.gpu_t ,gpu_t,t.gpu_t,data_length);

        result.cpu();
        result.gpu_free();

        return result;
    }
    template ts::Tensor<double> ts::Tensor<double>::cu_div(ts::Tensor<double>&);

    template<typename T>
    Tensor<T> Tensor<T>::cu_ein(Tensor<T>& t){
        std::vector<int> shap = {shape[0],t.shape[1]};
        Tensor<T> result = Tensor<T>(shap);
        if(gpu_t == nullptr){
            gpu();
        }
        if(t.gpu_t == nullptr){
            t.gpu();
        }

        int threadsPerBlock = 1024;
        int numBlocks = (data_length + threadsPerBlock-1) / threadsPerBlock;

        result.gpu();

        ein_cu_kernal<<<numBlocks,threadsPerBlock>>>(result.gpu_t,gpu_t,t.gpu_t,shape[0],shape[1],t.shape[1]);

        result.cpu();
        result.gpu_free();

        return result;

    }
    template ts::Tensor<double> ts::Tensor<double>::cu_ein(ts::Tensor<double>&);










    template<typename T>
    __global__ void eq_cu_kernal(bool* result, const T* data1, const T* data2, int length){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length)
        {
            result[idx] = (data1[idx] == data2[idx]);
        }
    }

    template<typename T>
    Tensor<bool> Tensor<T>::cu_eq(Tensor<T>& t){
        Tensor<bool> result = Tensor<bool>(shape);

        if(gpu_t == nullptr){
            gpu();
        }
        if(t.gpu_t == nullptr){
            t.gpu();
        }

        int threadsPerBlock = 1024;
        int numBlocks = (data_length + threadsPerBlock-1) / threadsPerBlock;

        result.gpu();

        eq_cu_kernal<<<numBlocks,threadsPerBlock>>>(result.get_gpu_t() ,gpu_t,t.gpu_t,data_length);

        result.cpu();
        result.gpu_free();

        return result;
    }
    template ts::Tensor<bool> ts::Tensor<double>::cu_eq(ts::Tensor<double>&);


    template<typename T>
    __global__ void ne_cu_kernal(bool* result, const T* data1, const T* data2, int length){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length)
        {
            result[idx] = (data1[idx] != data2[idx]);
        }
    }

    template<typename T>
    Tensor<bool> Tensor<T>::cu_ne(Tensor<T>& t){
        Tensor<bool> result = Tensor<bool>(shape);

        if(gpu_t == nullptr){
            gpu();
        }
        if(t.gpu_t == nullptr){
            t.gpu();
        }

        int threadsPerBlock = 1024;
        int numBlocks = (data_length + threadsPerBlock-1) / threadsPerBlock;

        result.gpu();

        ne_cu_kernal<<<numBlocks,threadsPerBlock>>>(result.get_gpu_t() ,gpu_t,t.gpu_t,data_length);

        result.cpu();
        result.gpu_free();

        return result;
    }
    template ts::Tensor<bool> ts::Tensor<double>::cu_ne(ts::Tensor<double>&);

    template<typename T>
    __global__ void gt_cu_kernal(bool* result, const T* data1, const T* data2, int length){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length)
        {
            result[idx] = (data1[idx] > data2[idx]);
        }
    }

    template<typename T>
    Tensor<bool> Tensor<T>::cu_gt(Tensor<T>& t){
        Tensor<bool> result = Tensor<bool>(shape);

        if(gpu_t == nullptr){
            gpu();
        }
        if(t.gpu_t == nullptr){
            t.gpu();
        }

        int threadsPerBlock = 1024;
        int numBlocks = (data_length + threadsPerBlock-1) / threadsPerBlock;

        result.gpu();

        gt_cu_kernal<<<numBlocks,threadsPerBlock>>>(result.get_gpu_t() ,gpu_t,t.gpu_t,data_length);

        result.cpu();
        result.gpu_free();

        return result;
    }
    template ts::Tensor<bool> ts::Tensor<double>::cu_gt(ts::Tensor<double>&);

    template<typename T>
    __global__ void ge_cu_kernal(bool* result, const T* data1, const T* data2, int length){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length)
        {
            result[idx] = (data1[idx] >= data2[idx]);
        }
    }

    template<typename T>
    Tensor<bool> Tensor<T>::cu_ge(Tensor<T>& t){
        Tensor<bool> result = Tensor<bool>(shape);

        if(gpu_t == nullptr){
            gpu();
        }
        if(t.gpu_t == nullptr){
            t.gpu();
        }

        int threadsPerBlock = 1024;
        int numBlocks = (data_length + threadsPerBlock-1) / threadsPerBlock;

        result.gpu();

        ge_cu_kernal<<<numBlocks,threadsPerBlock>>>(result.get_gpu_t() ,gpu_t,t.gpu_t,data_length);

        result.cpu();
        result.gpu_free();

        return result;
    }
    template ts::Tensor<bool> ts::Tensor<double>::cu_ge(ts::Tensor<double>&);

    template<typename T>
    __global__ void lt_cu_kernal(bool* result, const T* data1, const T* data2, int length){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length)
        {
            result[idx] = (data1[idx] < data2[idx]);
        }
    }

    template<typename T>
    Tensor<bool> Tensor<T>::cu_lt(Tensor<T>& t){
        Tensor<bool> result = Tensor<bool>(shape);

        if(gpu_t == nullptr){
            gpu();
        }
        if(t.gpu_t == nullptr){
            t.gpu();
        }

        int threadsPerBlock = 1024;
        int numBlocks = (data_length + threadsPerBlock-1) / threadsPerBlock;

        result.gpu();

        lt_cu_kernal<<<numBlocks,threadsPerBlock>>>(result.get_gpu_t() ,gpu_t,t.gpu_t,data_length);

        result.cpu();
        result.gpu_free();

        return result;
    }
    template ts::Tensor<bool> ts::Tensor<double>::cu_lt(ts::Tensor<double>&);

    template<typename T>
    __global__ void le_cu_kernal(bool* result, const T* data1, const T* data2, int length){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length)
        {
            result[idx] = (data1[idx] <= data2[idx]);
        }
    }

    template<typename T>
    Tensor<bool> Tensor<T>::cu_le(Tensor<T>& t){
        Tensor<bool> result = Tensor<bool>(shape);

        if(gpu_t == nullptr){
            gpu();
        }
        if(t.gpu_t == nullptr){
            t.gpu();
        }

        int threadsPerBlock = 1024;
        int numBlocks = (data_length + threadsPerBlock-1) / threadsPerBlock;

        result.gpu();

        le_cu_kernal<<<numBlocks,threadsPerBlock>>>(result.get_gpu_t() ,gpu_t,t.gpu_t,data_length);

        result.cpu();
        result.gpu_free();

        return result;
    }
    template ts::Tensor<bool> ts::Tensor<double>::cu_le(ts::Tensor<double>&);















    template<typename T>
    void Tensor<T>::gpu(){
        if(gpu_t != nullptr){
            cudaFree(gpu_t);
            gpu_t = nullptr;
        }
        cudaMalloc((void**)&gpu_t, data_length*sizeof(T)); 
        cudaMemcpy(gpu_t, data.get(), data_length*sizeof(T), cudaMemcpyHostToDevice);
    }
    template void Tensor<double>::gpu();
    template void Tensor<bool>::gpu();


    template<typename T>
    void Tensor<T>::cpu(){
        cudaMemcpy(data.get(), gpu_t, data_length*sizeof(T),cudaMemcpyDeviceToHost);
    }
    template void Tensor<double>::cpu();
    template void Tensor<bool>::cpu();



    template<typename T>
    void Tensor<T>::gpu_free(){
        if(gpu_t != nullptr){
            cudaFree(gpu_t);
            gpu_t = nullptr;
        }
    }
    template void Tensor<double>::gpu_free();
    template void Tensor<bool>::gpu_free();



    template<typename T>
    T* Tensor<T>::get_gpu_t(){
        return gpu_t;
    }
    template double* Tensor<double>::get_gpu_t();
    template bool* Tensor<bool>::get_gpu_t();


    
}