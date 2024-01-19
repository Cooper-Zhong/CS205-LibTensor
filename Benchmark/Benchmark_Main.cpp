#include <benchmark/benchmark.h>

static void BM_StringCreation(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation)->DenseRange(16, 1024, 16)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13);

// Define another benchmark
static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  for (auto _ : state)
    std::string copy(x);
}
BENCHMARK(BM_StringCopy)->DenseRange(16, 1024, 16)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13);

BENCHMARK_MAIN();