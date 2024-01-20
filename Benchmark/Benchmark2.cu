#include <benchmark/benchmark.h>
#include "tensor.h"
#include "test.h"

#define ADD_MATH_BENCHMARK(FUNC, BENCHMARK_NAME)                       \
    static void BENCHMARK_NAME(benchmark::State &state)                \
    {                                                                  \
        int N = state.range(0);                                        \
        state.SetComplexityN(state.range(0));                          \
        vector<int> shape = {512, 512, 512};                           \
        ts::Tensor<double> a = ts::Tensor<double>::rand_tensor(shape); \
        ts::Tensor<double> b = ts::Tensor<double>::rand_tensor(shape); \
                                                                       \
        for (auto _ : state)                                           \
        {                                                              \
            for (int i = 0; i < N; i++)                                \
            {                                                          \
                auto result = a.FUNC(b);                               \
                benchmark::DoNotOptimize(result);                      \
                benchmark::ClobberMemory();                            \
            }                                                          \
        }                                                              \
    }                                                                  \
    BENCHMARK(BENCHMARK_NAME)->DenseRange(16, 512, 16)->Complexity(benchmark::oAuto);

#define ADD_CUDA_MATH_BENCHMARK(FUNC, BENCHMARK_NAME)                  \
    using namespace ts;                                                \
    static void BENCHMARK_NAME(benchmark::State &state)                \
    {                                                                  \
        int N = state.range(0);                                        \
        state.SetComplexityN(state.range(0));                          \
        vector<int> shape = {512, 512, 512};                           \
        ts::Tensor<double> a = ts::Tensor<double>::rand_tensor(shape); \
        ts::Tensor<double> b = ts::Tensor<double>::rand_tensor(shape); \
        ts::Tensor<double> result = ts::Tensor<double>(shape);         \
        for (auto _ : state)                                           \
        {                                                              \
            for (int i = 0; i < N; i++)                                \
            {                                                          \
                result = a.FUNC(b);                                    \
                benchmark::DoNotOptimize(result);                      \
                benchmark::ClobberMemory();                            \
            }                                                          \
        }                                                              \
        a.gpu_free();                                                  \
        b.gpu_free();                                                  \
        result.gpu_free();                                             \
    }                                                                  \
    BENCHMARK(BENCHMARK_NAME)->DenseRange(16, 512, 16)->Complexity(benchmark::oAuto);

ADD_MATH_BENCHMARK(add, BM_add_benchmark);

ADD_MATH_BENCHMARK(omp_add, BM_omp_add_benchmark);
ADD_CUDA_MATH_BENCHMARK(cu_add, BM_cu_add_benchmark);

// ADD_MATH_BENCHMARK(mul, BM_mul_benchmark);
// ADD_MATH_BENCHMARK(omp_mul, BM_omp_mul_benchmark);

// ADD_MATH_BENCHMARK(div, BM_div_benchmark);
// ADD_MATH_BENCHMARK(omp_div, BM_omp_div_benchmark);

BENCHMARK_MAIN();