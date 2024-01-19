#include <benchmark/benchmark.h>
#include "tensor.h"
#include "test.h"

#define ADD_MATH_BENCHMARK(FUNC, BENCHMARK_NAME)                       \
    static void BENCHMARK_NAME(benchmark::State &state)                \
    {                                                                  \
        int N = state.range(0);                                        \
        state.SetComplexityN(state.range(0));                          \
        vector<int> shape = {N, N, N};                                 \
        ts::Tensor<double> a = ts::Tensor<double>::rand_tensor(shape); \
        ts::Tensor<double> b = ts::Tensor<double>::rand_tensor(shape); \
                                                                       \
        for (auto _ : state)                                           \
        {                                                              \
            auto result = a.FUNC(b);                                   \
            benchmark::DoNotOptimize(result);                          \
            benchmark::ClobberMemory();                                \
        }                                                              \
    }                                                                  \
    BENCHMARK(BENCHMARK_NAME)->DenseRange(16, 512, 16)->Complexity(benchmark::oNCubed);

#define ADD_CUDA_MATH_BENCHMARK(FUNC, BENCHMARK_NAME)                       \
    static void BENCHMARK_NAME(benchmark::State &state)                     \
    {                                                                       \
        int N = state.range(0);                                             \
        state.SetComplexityN(state.range(0));                               \
        vector<int> shape = {N, N, N};                                      \
        ts::Tensor<double> a = ts::Tensor<double>::rand_tensor(shape);      \
        ts::Tensor<double> b = ts::Tensor<double>::rand_tensor(shape);      \
        double *result, *a_data, *b_data;                                   \
        datatToDevice(a_data, a.get_data(), a.get_data_length());           \
        datatToDevice(b_data, b.get_data(), b.get_data_length());           \
        cudaMalloc((void **)&result, a.get_data_length() * sizeof(double)); \
                                                                            \
        for (auto _ : state)                                                \
        {                                                                   \
            FUNC(result, a_data, b_data, a.get_data_length());          \
            benchmark::DoNotOptimize(result);                               \
            benchmark::ClobberMemory();                                     \
        }                                                                   \
        datatToHost(a.get_data(), a_data, a.get_data_length());             \
        datatToHost(b.get_data(), b_data, b.get_data_length());             \
        cudaFree(result);                                                   \
    }                                                                       \
    BENCHMARK(BENCHMARK_NAME)->DenseRange(16, 512, 16)->Complexity(benchmark::oNCubed);


ADD_MATH_BENCHMARK(add_perf, BM_add_perf_benchmark);
ADD_MATH_BENCHMARK(add, BM_add_benchmark);
ADD_MATH_BENCHMARK(omp_add, BM_omp_add_benchmark);

// ADD_MATH_BENCHMARK(mul, BM_mul_benchmark);
// ADD_MATH_BENCHMARK(omp_mul, BM_omp_mul_benchmark);

// ADD_MATH_BENCHMARK(div, BM_div_benchmark);
// ADD_MATH_BENCHMARK(omp_div, BM_omp_div_benchmark);

BENCHMARK_MAIN();