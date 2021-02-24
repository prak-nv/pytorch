
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include "utils.h"

using namespace torch::jit::fuser::cuda;

static TensorView* setupTranspose(
    Fusion* fusion,
    TensorView* input) {
  FusionGuard fg(fusion);

  auto tv0 = input;
  auto tv1 = transpose(tv0, {{0, 1}});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  const int BS = 32;
  const int BDIM = 256;

  // CTA tiling by BS*BS
  tv1->split(1, BS);
  tv1->split(0, BS);
  tv1->reorder({{1, 2}});
  // tv1: [I1/BS, I0/BS, BS(I1), BS(I0)]

  // Create a smem buffer to cache each tile
  auto tv0_cache = tv0->cache_after();
  tv0_cache->setMemoryType(MemoryType::Shared);

  tv0->computeAt(tv1, 2);
  // tv0: [I0, I1]
  // tv0_cache: [I1/BS, I0/BS, BS*BS/BDIM, BDIM]
  // tv1: [I1/BS, I0/BS, BS*BS/BDIM, BDIM]

  // Tranform the tile axes for 1D thread mapping
  tv1->merge(-2, -1);
  tv1->split(-1, BDIM);
  // tv1: [I1/BS, I0/BS, BS*BS/BDIM, BDIM]

  // Transform the cache similarly but apply swizzle to the 2D tile axes.
  tv0_cache->reorder({{-2, -1}});
  tv0_cache->swizzle(SwizzleType::Transpose, {2, 3});
  tv0_cache->merge(-2, -1);
  tv0_cache->split(-1, BDIM);
  // tv0: [I1/BS, I0/BS, BS*BS/BDIM, BDIM]

  // Assign each thread block to a tile
  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(1)->parallelize(ParallelType::BIDx);

  // Thread mapping for each tile.
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);

  return tv1;
}


static TensorView* setupTranspose4D(
    Fusion* fusion,
    TensorView* input) {
  FusionGuard fg(fusion);

  auto tv0 = input;
  auto tv1 = transpose(tv0, {{0, 0},{1,3},{2,1},{3,2}});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  const int BS = 32;
  const int BDIM = 256;

  tv1->merge(1,2);

  // CTA tiling by BS*BS
  tv1->split(2, BS);
  tv1->split(1, BS);
  tv1->reorder({{2, 3}});
  // tv1: [I1/BS, I0/BS, BS(I1), BS(I0)]

  // Create a smem buffer to cache each tile
  auto tv0_cache = tv0->cache_after();
  tv0_cache->setMemoryType(MemoryType::Shared);

  tv0->computeAt(tv1, 3);

   // Tranform the tile axes for 1D thread mapping
  tv1->merge(-2, -1);
  tv1->split(-1, BDIM);
  // tv1: [I1/BS, I0/BS, BS*BS/BDIM, BDIM]

  // Transform the cache similarly but apply swizzle to the 2D tile axes.
  tv0_cache->reorder({{-2, -1}});
  tv0_cache->swizzle(SwizzleType::Transpose, {3, 4});
  tv0_cache->merge(-2, -1);
  tv0_cache->split(-1, BDIM);
  // tv0: [I1/BS, I0/BS, BS*BS/BDIM, BDIM]

  // Assign each thread block to a tile
  tv1->axis(0)->parallelize(ParallelType::BIDz);
  tv1->axis(1)->parallelize(ParallelType::BIDy);
  tv1->axis(2)->parallelize(ParallelType::BIDx);

  // Thread mapping for each tile.
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);
  
  return tv1;
}

static TensorView* setupPointwise(
    Fusion* fusion,
    TensorView* input) {
  FusionGuard fg(fusion);

  auto tv0 = input;
  auto tv1 = add(tv0,new Double(1));
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  return tv1;
}


inline void timeFusionRun(
    Fusion& fusion,
    std::vector<int64_t> & input_shape,
    benchmark::State& benchmark_state
){
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  std::vector<c10::IValue> inputs({at_x});

  // outputs
  std::vector<at::Tensor> outputs;

  FusionExecutor executor;
  executor.setMeasureKernelTimeFlag(true);
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(
        c10::ArrayRef<c10::IValue>(inputs));
    benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
    cudaDeviceSynchronize();
  }

  benchmark_state.SetBytesProcessed(8*benchmark_state.iterations()*std::accumulate(std::begin(input_shape), std::end(input_shape), 1, std::multiplies<int64_t>()));
}

static void Transpose2D(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{benchmark_state.range(0),benchmark_state.range(1)};
  
  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  
  auto output = setupTranspose(&fusion, input);

  timeFusionRun(fusion,input_shape,benchmark_state);
}


static void Transpose4D(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{benchmark_state.range(0),benchmark_state.range(1),benchmark_state.range(2),benchmark_state.range(3)};
  
  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  
  auto output = setupTranspose4D(&fusion, input);

  timeFusionRun(fusion,input_shape,benchmark_state);
}

static void SingleAddBaseline2D(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{benchmark_state.range(0), benchmark_state.range(1)};
  
  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  
  auto output = setupPointwise(&fusion, input);
  scheduleFusion(&fusion);
  
  timeFusionRun(fusion,input_shape,benchmark_state);
}

static void SingleAddBaseline4D(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{benchmark_state.range(0), benchmark_state.range(1),benchmark_state.range(2), benchmark_state.range(3)};
  
  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  
  auto output = setupPointwise(&fusion, input);
  scheduleFusion(&fusion);
  
  timeFusionRun(fusion,input_shape,benchmark_state);
}


BENCHMARK(Transpose2D)
    ->RangeMultiplier(2)
    ->Ranges({{128<<3, 128 << 5},{128<<3, 128 << 5}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Transpose4D)
    ->Ranges({{32,64},{3,3},{16,32},{16,32}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();


BENCHMARK(SingleAddBaseline2D)
    ->RangeMultiplier(2)
    ->Ranges({{128<<3, 128 << 5},{128<<3, 128 << 5}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(SingleAddBaseline4D)
    ->Ranges({{32,64},{3,3},{16,32},{16,32}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();