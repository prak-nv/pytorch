
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
    TensorView* input,
    const std::unordered_map<int, int>& old2new) {
  FusionGuard fg(fusion);

    auto tv0 = input;
    auto tv1 = transpose(tv0, old2new);
    fusion->addInput(tv0);
    fusion->addOutput(tv1);

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

// Input axes are outter-middle-inner (OMI)
enum class PERMUTE3D{
  OIM,
  MIO,
  MOI,
  IMO,
  IOM
};

// Input axes are NCHW
enum class PERMUTE4D{
  NHWC
};

std::unordered_map<int,int> getPermute3D(
  PERMUTE3D index
){
  switch(index){
    case PERMUTE3D::OIM:
      return {{1,2}};
    case PERMUTE3D::MIO:
      return {{0,2},{1,0},{2,1}};
    case PERMUTE3D::MOI:
      return {{0,1}};
    case PERMUTE3D::IMO:
      return {{0,2}};
    case PERMUTE3D::IOM:
      return {{0,1},{1,2},{2,0}};
    default:
      TORCH_INTERNAL_ASSERT(false,"unreachable");
  }
  return {};
}

std::unordered_map<int,int> getPermute4D(
  PERMUTE4D index
){
  switch(index){
    case PERMUTE4D::NHWC:
      return {{1,3},{2,1},{3,2}};
    default:
      TORCH_INTERNAL_ASSERT(false,"unreachable");
  }
  return {};
}

static void transposeHelper(benchmark::State& benchmark_state, size_t ndims, std::unordered_map<int,int> old2new) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape(ndims);
  for(int i=0;i<ndims;i++){
    input_shape[i] = benchmark_state.range(i);
  }
   // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();

  auto output = setupTranspose(&fusion,input,old2new);

  scheduleTranspose(&fusion,output);

  timeFusionRun(fusion,input_shape,benchmark_state);
}

static void runTranspose(benchmark::State& benchmark_state, PERMUTE3D permute){
  transposeHelper(benchmark_state,3,getPermute3D(permute));
}

static void runTranspose(benchmark::State& benchmark_state, PERMUTE4D permute){
  transposeHelper(benchmark_state,4,getPermute4D(permute));
}

static void runSingleAdd(benchmark::State& benchmark_state, size_t ndims) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape(ndims);
  for(int i=0;i<ndims;i++){
    input_shape[i] = benchmark_state.range(i);
  }
  
  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  
  auto output = setupPointwise(&fusion, input);
  scheduleFusion(&fusion);
  
  timeFusionRun(fusion,input_shape,benchmark_state);
}

template <class ...ExtraArgs>
static void BenchmarkTranspose(benchmark::State& benchmark_state, ExtraArgs&&... extra_args) {
  runTranspose(benchmark_state,extra_args...);
}

template <class ...ExtraArgs>
static void BenchmarkSingleAdd(benchmark::State& benchmark_state, ExtraArgs&&... extra_args) {
  runSingleAdd(benchmark_state,extra_args...);
}


BENCHMARK_CAPTURE(BenchmarkSingleAdd,ADD2D,2)
    ->RangeMultiplier(2)
    ->Ranges({{128<<3, 128 << 5},{128<<3, 128 << 5}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK_CAPTURE(BenchmarkSingleAdd,ADD3D,3)
    ->RangeMultiplier(8)
    ->Ranges({{8,512},{8,512},{8,512}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK_CAPTURE(BenchmarkSingleAdd,ADD4D,4)
    ->Ranges({{32,64},{3,3},{16,32},{16,32}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK_CAPTURE(BenchmarkTranspose,Transpose3DOIM,PERMUTE3D::OIM)
    ->RangeMultiplier(8)
    ->Ranges({{8,512},{8,512},{8,512}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK_CAPTURE(BenchmarkTranspose,Transpose3DMIO,PERMUTE3D::MIO)
    ->RangeMultiplier(8)
    ->Ranges({{8,512},{8,512},{8,512}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK_CAPTURE(BenchmarkTranspose,Transpose3DMOI,PERMUTE3D::MOI)
    ->RangeMultiplier(8)
    ->Ranges({{8,512},{8,512},{8,512}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();


BENCHMARK_CAPTURE(BenchmarkTranspose,Transpose3DIMO,PERMUTE3D::IMO)
    ->RangeMultiplier(8)
    ->Ranges({{8,512},{8,512},{8,512}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK_CAPTURE(BenchmarkTranspose,Transpose3DIOM,PERMUTE3D::IOM)
    ->RangeMultiplier(8)
    ->Ranges({{8,512},{8,512},{8,512}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();


BENCHMARK_CAPTURE(BenchmarkTranspose,Transpose4DNHWC,PERMUTE4D::NHWC)
    ->Ranges({{32,64},{3,3},{16,32},{16,32}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
