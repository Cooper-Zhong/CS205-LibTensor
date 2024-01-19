
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
    __global__ void ein_cu_kernal(T* result, const T* data1, const T* data2, int t1_height, int t1_width, int t2_width) {
        size_t idx = blockIdx.x * blockDim.x + threaIdx.x;
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

    template<typename T>
    Tensor<T> Tensor<T>::cu_ein(Tensor<T>& t){
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

        ein_cu_kernal<<<numBlocks,threadsPerBlock>>>(result.gpu_t,gpu_t,t.gpu_t,shape[0],shape[1],t.shape[1]);

        result.cpu();
        result.gpu_free();

        return result;

    }

    template ts::Tensor<double> ts::Tensor<double>::cu_add(ts::Tensor<double>&);

    template ts::Tensor<int> ts::Tensor<int>::add_cu(ts::Tensor<int> const&);
    template ts::Tensor<float> ts::Tensor<float>::add_cu(ts::Tensor<float> const&);
    template ts::Tensor<double> ts::Tensor<double>::add_cu(ts::Tensor<double> const&);

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


    template<typename T>
    void Tensor<T>::cpu(){
        cudaMemcpy(data.get(), gpu_t, data_length*sizeof(T),cudaMemcpyDeviceToHost);
    }
    template void Tensor<double>::cpu();


    template<typename T>
    void Tensor<T>::gpu_free(){
        if(gpu_t != nullptr){
            cudaFree(gpu_t);
            gpu_t = nullptr;
        }
    }
    template void Tensor<double>::gpu_free();


    template<typename T>
    T* Tensor<T>::get_gpu_t(){
        return gpu_t;
    }

    template double* Tensor<double>::get_gpu_t();
    
    // template<typename T>
    // void dataToDevice(T** dev, const T** hos, int length){
    //     assert(cudaSuccess==cudaMalloc((void**)dev,length*sizeof(T)));
    //     assert(cudaSuccess==cudaMemcpy(*dev,*hos,length*sizeof(T),cudaMemcpyHostToDevice));
    // }

    // template void dataToDevice<double>(double**dev, const double** hos, int length);

    // template<typename T>
    // void dataToHost(T** dev, T** hos, int length){
    //     assert(cudaSuccess==cudaMemcpy(*hos,*dev,length*sizeof(T),cudaMemcpyDeviceToHost));
    //     assert(cudaSuccess==cudaFree(*dev));
    //     dev=nullptr;
    // }

    // template void dataToHost<double>(double**dev, double** hos, int length);

    // template<typename T>
    // void add_perf(T* result, const T* data1, const T* data2, int length) {
    //     int threadsPerBlock = 1024;
    //     int numBlocks = (length + threadsPerBlock-1) / threadsPerBlock;
    //     add_cu_kernal<<<numBlocks,threadsPerBlock>>>(result, data1, data2, length);
    //     cudaDeviceSynchronize();
    // }

    // template void add_perf<double>(double* result, const double* data1, const double* data2, int length);


    
}