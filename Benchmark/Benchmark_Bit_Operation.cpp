#include <benchmark/benchmark.h>
#include "tensor.h"
#include "test.h"



static void BM_cu_bit_operation(benchmark::State &state)
{
    int N = state.range(0);
    state.SetComplexityN(state.range(0));
    vector<int> shape = {N, N};
    ts::Tensor<double> a = ts::Tensor<double>::rand_tensor(shape);
    ts::Tensor<double> b = ts::Tensor<double>::rand_tensor(shape);

    for (auto _ : state)
    {
        auto result1 = a.cu_add(b);
        auto result2 = a.cu_sub(b);
        auto result3 = a.cu_mul(b);
        auto result4 = a.cu_div(b);
        auto result5 = a.cu_log();
        auto result6 = b.cu_log();
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

BENCHMARK(BM_cu_bit_operation)->DenseRange(16, 512, 16)->Complexity(benchmark::oAuto);


static void BM_bit_operation(benchmark::State &state)
{
    int N = state.range(0);
    state.SetComplexityN(state.range(0));
    vector<int> shape = {N, N};
    ts::Tensor<double> a = ts::Tensor<double>::rand_tensor(shape);
    ts::Tensor<double> b = ts::Tensor<double>::rand_tensor(shape);

    for (auto _ : state)
    {
        auto result1 = a + b;
        auto result2 = a - b;
        auto result3 = a * b;
        auto result4 = a / b;
        auto result5 = a.log();
        auto result6 = b.log();
        benchmark::DoNotOptimize(result1);
        benchmark::DoNotOptimize(result2);
        benchmark::DoNotOptimize(result3);
        benchmark::DoNotOptimize(result4);
        benchmark::DoNotOptimize(result5);
        benchmark::DoNotOptimize(result6);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_bit_operation)->DenseRange(16, 512, 16)->Complexity(benchmark::oAuto);

static void BM_omp_bit_operation(benchmark::State &state)
{
    int N = state.range(0);
    state.SetComplexityN(state.range(0));
    vector<int> shape = {N, N};
    ts::Tensor<double> a = ts::Tensor<double>::rand_tensor(shape);
    ts::Tensor<double> b = ts::Tensor<double>::rand_tensor(shape);

    for (auto _ : state)
    {
        auto result1 = a.omp_add(b);
        auto result2 = a.omp_sub(b);
        auto result3 = a.omp_mul(b);
        auto result4 = a.omp_div(b);
        auto result5 = a.omp_log();
        auto result6 = b.omp_log();
        benchmark::DoNotOptimize(result1);
        benchmark::DoNotOptimize(result2);
        benchmark::DoNotOptimize(result3);
        benchmark::DoNotOptimize(result4);
        benchmark::DoNotOptimize(result5);
        benchmark::DoNotOptimize(result6);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_omp_bit_operation)->DenseRange(16, 512, 16)->Complexity(benchmark::oAuto);

BENCHMARK_MAIN();