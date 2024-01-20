#include <benchmark/benchmark.h>
#include "tensor.h"
#include "test.h"

static void BM_matrix_product(benchmark::State &state)
{
    int N = state.range(0);
    state.SetComplexityN(state.range(0));
    vector<int> shape = {N, N};
    ts::Tensor<double> a = ts::Tensor<double>::rand_tensor(shape);
    ts::Tensor<double> b = ts::Tensor<double>::rand_tensor(shape);

    for (auto _ : state)
    {
        auto result = ts::einsum<double>("ij,jk->ik", {a, b});
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_matrix_product)->DenseRange(8, 256, 8)->Complexity(benchmark::oAuto);

static void BM_cu_matrix_product(benchmark::State &state)
{
    int N = state.range(0);
    state.SetComplexityN(state.range(0));
    vector<int> shape = {N, N};
    ts::Tensor<double> a = ts::Tensor<double>::rand_tensor(shape);
    ts::Tensor<double> b = ts::Tensor<double>::rand_tensor(shape);

    for (auto _ : state)
    {
        auto result = a.cu_ein(b);
        result.gpu_free();
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
    a.gpu_free();
    b.gpu_free();
}

BENCHMARK(BM_cu_matrix_product)->DenseRange(16, 1024, 16)->Complexity(benchmark::oAuto);

BENCHMARK_MAIN();