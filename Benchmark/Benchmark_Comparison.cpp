#include <benchmark/benchmark.h>
#include <vector>
#include "tensor.h"
#include "test.h"


static void BM_comparison(benchmark::State &state)
{
    int N = state.range(0);
    state.SetComplexityN(state.range(0));
    vector<int> shape = {N, N};
    ts::Tensor<double> a = ts::Tensor<double>::rand_tensor(shape);
    ts::Tensor<double> b = ts::Tensor<double>::rand_tensor(shape);

    for (auto _ : state)
    {
        auto result1 = a == b;
        auto result2 = a != b;
        auto result3 = a > b;
        auto result4 = a >= b;
        auto result5 = a < b;
        auto result6 = a <= b;
        benchmark::DoNotOptimize(result1);
        benchmark::DoNotOptimize(result2);
        benchmark::DoNotOptimize(result3);
        benchmark::DoNotOptimize(result4);
        benchmark::DoNotOptimize(result5);
        benchmark::DoNotOptimize(result6);
        benchmark::ClobberMemory();
    }
}

static void BM_cu_comparison(benchmark::State &state)
{
    int N = state.range(0);
    state.SetComplexityN(state.range(0));
    vector<int> shape = {N, N};
    ts::Tensor<double> a = ts::Tensor<double>::rand_tensor(shape);
    ts::Tensor<double> b = ts::Tensor<double>::rand_tensor(shape);
    for (auto _ : state)
    {
        auto result1 = a.cu_eq(b);
        auto result2 = a.cu_ne(b);
        auto result3 = a.cu_gt(b);
        auto result4 = a.cu_ge(b);
        auto result5 = a.cu_lt(b);
        auto result6 = a.cu_le(b);
        benchmark::DoNotOptimize(result1);
        benchmark::DoNotOptimize(result2);
        benchmark::DoNotOptimize(result3);
        benchmark::DoNotOptimize(result4);
        benchmark::DoNotOptimize(result5);
        benchmark::DoNotOptimize(result6);
        benchmark::ClobberMemory();
    }
    
    a.gpu_free();
    b.gpu_free();
}


static void BM_omp_comparison(benchmark::State &state)
{
    int N = state.range(0);
    state.SetComplexityN(state.range(0));
    vector<int> shape = {N, N};
    ts::Tensor<double> a = ts::Tensor<double>::rand_tensor(shape);
    ts::Tensor<double> b = ts::Tensor<double>::rand_tensor(shape);

    for (auto _ : state)
    {
        auto result1 = a.omp_eq(b);
        auto result2 = a.omp_ne(b);
        auto result3 = a.omp_gt(b);
        auto result4 = a.omp_ge(b);
        auto result5 = a.omp_lt(b);
        auto result6 = a.omp_le(b);
        benchmark::DoNotOptimize(result1);
        benchmark::DoNotOptimize(result2);
        benchmark::DoNotOptimize(result3);
        benchmark::DoNotOptimize(result4);
        benchmark::DoNotOptimize(result5);
        benchmark::DoNotOptimize(result6);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_cu_comparison)->DenseRange(16, 512, 16)->Complexity(benchmark::oAuto);
BENCHMARK(BM_comparison)->DenseRange(16, 512, 16)->Complexity(benchmark::oAuto);
BENCHMARK(BM_omp_comparison)->DenseRange(16, 512, 16)->Complexity(benchmark::oAuto);

BENCHMARK_MAIN();