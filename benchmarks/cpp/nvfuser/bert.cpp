#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <sstream>

#include "operators.h"
#include "utils.h"

using namespace torch::jit::fuser::cuda;

// Return reduction tensor view and output of reduction
static void setupDivMaxSoftmaxDropoutForward(
    Fusion* fusion,
    DataType dtype) {

  FusionGuard fg(fusion);

  bool is_fp16 = dtype == DataType::Half;

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(4)
                        .dtype(dtype)
                        .contiguity({true, true, true, true})
                        .shape({-1, 1, 1, -1})
                        .build();
  fusion->addInput(tv0);

  TensorView* tv1 = TensorViewBuilder()
                        .ndims(4)
                        .dtype(dtype)
                        .contiguity({true, true, true, true})
                        .build();
  fusion->addInput(tv1);

  auto d16 = new Double(1.0);

  if (is_fp16) {
    tv0 = castOp(DataType::Float, tv0);
  }

  if (is_fp16) {
    tv1 = castOp(DataType::Half, tv1);
  }

  auto tv2 = div(tv1, d16);
  auto tv3 = add(tv2, tv0);

  auto tv10 = setupSoftmax(fusion, tv3, 4, 3);
  auto dropout_tvs = setupDropout(fusion, tv10, 0.9);
  auto tv12 = dropout_tvs.first;
  auto tv14 = dropout_tvs.second;

  if(is_fp16){
    tv14 = castOp(DataType::Float, tv14);
    tv12 = castOp(DataType::Float, tv12);
    tv10 = castOp(DataType::Float, tv10);
    tv3 = castOp(DataType::Float, tv3);
  }

  fusion->addOutput(tv14);
  fusion->addOutput(tv12);
  fusion->addOutput(tv10);
  fusion->addOutput(tv3);
}

static void setupDivMaxSoftmaxDropoutBackward(
    Fusion* fusion,
    DataType dtype,
    int red_axis) {
  TensorView* tv0 = TensorViewBuilder()
                        .ndims(4)
                        .dtype(dtype)
                        .contiguity({true, true, true, true})
                        .build();
  fusion->addInput(tv0);
  TensorView* tv1 = TensorViewBuilder()
                        .ndims(4)
                        .dtype(dtype)
                        .contiguity({true, true, true, true})
                        .build();
  fusion->addInput(tv1);
  TensorView* tv2 = TensorViewBuilder()
                        .ndims(4)
                        .dtype(dtype)
                        .contiguity({true, true, true, true})
                        .build();
  fusion->addInput(tv2);
  TensorView* tv3 = TensorViewBuilder()
                        .ndims(4)
                        .dtype(dtype)
                        .contiguity({true, true, true, true})
                        .build();
  fusion->addInput(tv3);
  auto d32 = new Double(1.0);
  fusion->addInput(d32);
  auto d33 = new Double(2.0);
  fusion->addInput(d33);

  auto tv4 = mul(tv2, tv3);
  auto tv5 = mul(tv4, d33);
  auto tv6 = mul(tv5, tv0);
  auto tv7 = sum(tv6, {-1});
  auto tv8 = broadcast(tv7, {false, false, false, true});
  auto tv9 = mul(tv0, tv8);
  auto tv10 = sub(tv6, tv9);
  auto tv11 = div(tv10, d32);
  fusion->addOutput(tv11);
  fusion->addOutput(tv10);
}

static void MagicScheduler_DivMaxSoftDropFwd(benchmark::State& benchmark_state,
  DataType dtype) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto w = benchmark_state.range(0);
  auto x = benchmark_state.range(1);
  auto y = benchmark_state.range(2);
  auto z = benchmark_state.range(3);

  setupDivMaxSoftmaxDropoutForward(&fusion, dtype);

  auto tvs = scheduler_utils::allTvs(&fusion);

  std::vector<TensorView*> reduction_tvs;
  std::copy_if(
      tvs.begin(),
      tvs.end(),
      std::back_inserter(reduction_tvs),
      [](TensorView* tv) { return tv->hasReduction(); });

  std::vector<TensorView*> other_tvs;
  std::copy_if(
      tvs.begin(),
      tvs.end(),
      std::back_inserter(other_tvs),
      [](TensorView* tv) { return !tv->hasReduction(); });

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({w, 1, 1, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  auto norm_params = getNormalizationHeuristics(&fusion, {t0, t1}, reduction_tvs);

  auto rparams = norm_params.value();
  auto lparams = rparams.lparams;

  TORCH_CHECK(norm_params.has_value(), "Norm scheduler can't be used!");
  scheduleNormalization(&fusion, rparams, reduction_tvs, other_tvs);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.setMeasureKernelTimeFlag(true);
  // Sync everything up before we start
  std::vector<at::Tensor> cg_outputs;
  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    cg_outputs = fe.runFusion({t0, t1}, lparams);
    benchmark_state.SetIterationTime(fe.kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  cudaDeviceSynchronize();
  
  int64_t bytes = 0;
  for(auto tensor : std::vector<at::Tensor>({t0, t1})){
    bytes +=
        tensor.numel() * (int64_t) dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  for(auto tensor : cg_outputs){
    bytes +=
        tensor.numel() * (int64_t) dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  benchmark_state.SetBytesProcessed(bytes * int64_t(benchmark_state.iterations()) );
}

static void MagicScheduler_fp32_DivMaxSoftDropFwd(benchmark::State& benchmark_state) {
  MagicScheduler_DivMaxSoftDropFwd(benchmark_state, DataType::Float);
}

BENCHMARK(MagicScheduler_fp32_DivMaxSoftDropFwd)
    ->RangeMultiplier(8)
    ->Ranges({{8, 8}, {16, 16}, {128, 128}, {128, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
